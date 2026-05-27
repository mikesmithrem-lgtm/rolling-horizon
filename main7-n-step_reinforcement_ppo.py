"""PPO variant of the n-step reinforcement training pipeline.

Mirrors the structure of ``main6-n-step_reinforcement.py`` but:
  * uses the PPO-modified env in ``L2S_PPO`` (raw makespan-difference reward);
  * uses the PPO-modified actor in ``L2S_PPO`` (adds a critic head and an
    ``evaluate``-style forward path);
  * replaces the REINFORCE update with a clipped PPO update (GAE advantages,
    multi-epoch minibatch updates, value + entropy losses).

The original ``L2S`` package and the original main6 script are not touched.
"""

import time
import random
import os
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from types import SimpleNamespace

from L2S_PPO.parameters import args
from L2S_PPO.env.environment import JsspWindow, BatchGraph
from L2S_PPO.env.window_utils import _init_worker_nonzero_cpu_affinity
from L2S_PPO.model.actor import Actor
from L2S_PPO.env.generateJSP import uni_instance_gen


def _bind_main_process_away_from_cpu0():
    """Prefer non-zero CPUs for the training process when affinity control is available."""
    if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
        return None
    current = set(os.sched_getaffinity(0))
    allowed = {cpu for cpu in current if cpu != 0}
    target = allowed if allowed else current
    os.sched_setaffinity(0, target)
    return set(os.sched_getaffinity(0))


def _pack_batch_states(states):
    x, edge_index_pc, edge_index_mc, batch = states
    return SimpleNamespace(
        x=x,
        edge_index_pc=edge_index_pc,
        edge_index_mc=edge_index_mc,
        batch=batch,
    )


# ===== validation worker (single-instance, CPU policy) ========================
# A pool of worker processes each owns one CPU policy + one single-instance env.
# The env is built with cpu_budget=1 so its internal CP-solver pool is bypassed
# and stays serial inside the worker — all parallelism comes from the outer pool.

_WORKER_POLICY = None
_WORKER_ENV = None
_WORKER_INIT_TYPE = None
_WORKER_VAL_TRANSIT = None


def _validation_worker_init(actor_kwargs, env_kwargs, state_dict, init_type, val_transit, torch_threads):
    global _WORKER_POLICY, _WORKER_ENV, _WORKER_INIT_TYPE, _WORKER_VAL_TRANSIT
    _init_worker_nonzero_cpu_affinity()
    torch.set_num_threads(max(1, int(torch_threads)))
    policy = Actor(**actor_kwargs)
    policy.load_state_dict(state_dict)
    policy.eval()
    _WORKER_POLICY = policy
    _WORKER_ENV = JsspWindow(**env_kwargs)
    _WORKER_INIT_TYPE = init_type
    _WORKER_VAL_TRANSIT = val_transit


def _validation_worker_run(instance):
    st = time.time()
    dev = torch.device("cpu")
    instances_np = np.asarray(instance)[None, ...]
    with torch.no_grad():
        states, batch_window_states, feasible_actions, _ = _WORKER_ENV.reset(
            instances=instances_np,
            init_type=_WORKER_INIT_TYPE,
            device=dev,
        )
        while _WORKER_ENV.itr < _WORKER_VAL_TRANSIT:
            batch_states = _pack_batch_states(states)
            actions, _ = _WORKER_POLICY(
                batch_states=batch_states,
                batch_window_states=batch_window_states,
                feasible_actions=feasible_actions,
            )
            states, batch_window_states, _, feasible_actions, _ = _WORKER_ENV.step(actions, dev)
    ed = time.time()
    print(f"Validation 1 instance for {ed - st} s")
    return float(_WORKER_ENV.current_objs.mean().cpu().item())


class PPO_L2S_CPJSSP:
    """PPO trainer for the CP-window JSSP environment."""

    def __init__(self):
        self.window_size = getattr(args, 'window_size', args.j * args.m)
        self.cp_solver_time = getattr(args, 'cp_solver_time', 1)
        self.cp_solver_cpu = getattr(args, 'cp_solver_cpu', 1)
        self.cpu_budget = getattr(args, 'cpu_budget', 16)
        self.tensorboard_root = Path(getattr(args, 'tensorboard_log_dir', './tensorboard_log_ppo'))
        self.training_env_log_path = getattr(args, 'window_training_log_path', None)
        self.validation_env_log_path = getattr(args, 'window_validation_log_path', None)

        self.env_training = JsspWindow(
            n_job=args.j,
            n_mch=args.m,
            low=args.l,
            high=args.h,
            cp_solver_time=self.cp_solver_time,
            cp_solver_cpu=self.cp_solver_cpu,
            cpu_budget=self.cpu_budget,
            window_size=self.window_size,
        )
        self.env_validation = JsspWindow(
            n_job=args.j,
            n_mch=args.m,
            low=args.l,
            high=args.h,
            cp_solver_time=self.cp_solver_time,
            cp_solver_cpu=self.cp_solver_cpu,
            cpu_budget=self.cpu_budget,
            window_size=self.window_size,
        )
        self.eps = np.finfo(np.float32).eps.item()
        self.current_validation_result = np.inf

        # === Monitoring state for PPO debug scalars ===
        # learn() 中累计 critic/advantage 统计
        self._last_critic_stats = {}
        # learn() 中累计 PPO 内部状态（ratio/KL/clip）
        self._last_ratio_max = 0.0
        self._last_approx_kl = 0.0
        self._last_clip_frac = 0.0
        self._last_ppo_inner_steps = 0

        self.dghan_param_for_saved_model = '{}_{}'.format(args.heads, args.drop_out)

        train_data_path = Path(
            "./L2S_PPO/training_data/training_instance_{}x{}[{},{}].npy".format(
                args.j, args.m, args.l, args.h
            )
        )
        train_data_path.parent.mkdir(parents=True, exist_ok=True)
        if train_data_path.is_file():
            self.training_data = np.load(str(train_data_path))
        else:
            self.training_data = np.array([
                uni_instance_gen(n_j=args.j, n_m=args.m, low=args.l, high=args.h)
                for _ in range(args.training_set_size)
            ])
            np.save(str(train_data_path), self.training_data)

        validation_data_path = Path(
            './L2S_PPO/validation_data/validation_instance_{}x{}[{},{}].npy'.format(
                args.j, args.m, args.l, args.h
            )
        )
        validation_data_path.parent.mkdir(parents=True, exist_ok=True)
        if validation_data_path.is_file():
            self.validation_data = np.load(str(validation_data_path))[:args.batch_size * 2, ...]
        else:
            print('No validation data for {}x{}[{},{}], generating new one.'.format(
                args.j, args.m, args.l, args.h
            ))
            self.validation_data = np.array([
                uni_instance_gen(n_j=args.j, n_m=args.m, low=args.l, high=args.h)
                for _ in range(100)
            ])
            np.save(str(validation_data_path), self.validation_data)

    # ----- utility tags / names -----------------------------------------------
    def _cp_run_tag(self):
        return 'cp{}_{}_wb{}_ws{}'.format(
            self.cp_solver_time,
            self.cp_solver_cpu,
            self.cpu_budget,
            self.window_size,
        )

    def _tb_run_name(self):
        return ('ppo_l2scp_'
                '{}x{}[{},{}]_'
                '{}_{}_{}_'
                '{}_{}_{}_'
                '{}_'
                '{}_{}_{}_'
                '{}_{}_{}'
                .format(args.j, args.m, args.l, args.h,
                        args.init_type, args.gamma, self._cp_run_tag(),
                        args.hidden_dim, args.embedding_layer, args.policy_layer,
                        self.dghan_param_for_saved_model,
                        args.lr, args.steps_learn, args.transit,
                        args.batch_size, args.episodes, args.step_validation))

    # ----- policy rollout helpers ---------------------------------------------
    def _policy_act(self, policy, states, batch_window_states, feasible_actions):
        """Sample actions and also return value + entropy for PPO storage."""
        batch_states = _pack_batch_states(states)
        actions, log_ps, entropy, values = policy(
            batch_states=batch_states,
            batch_window_states=batch_window_states,
            feasible_actions=feasible_actions,
            return_value=True,
        )
        return actions, log_ps, entropy, values, batch_states

    def _policy_evaluate(self, policy, batch_states, batch_window_states,
                        feasible_actions, chosen_actions):
        """Recompute log-probs / values / entropy for a stored transition."""
        _, log_ps, entropy, values = policy(
            batch_states=batch_states,
            batch_window_states=batch_window_states,
            feasible_actions=feasible_actions,
            chosen_actions=chosen_actions,
        )
        return log_ps, entropy, values
    
    def _log_ppo_update(self, writer, update_step,
                       policy_loss, value_loss, entropy_avg,
                       optimizer, avg_candidates):
        """Centralized logging of PPO update scalars."""
        writer.add_scalar("train/policy_loss", policy_loss, update_step)
        writer.add_scalar("train/value_loss", value_loss, update_step)
        writer.add_scalar("train/entropy", entropy_avg, update_step)

        for k, v in self._last_critic_stats.items():
            writer.add_scalar(f'critic/{k}', v, update_step)

        inner = max(self._last_ppo_inner_steps, 1)
        writer.add_scalar('ppo/approx_kl', self._last_approx_kl / inner, update_step)
        writer.add_scalar('ppo/clip_fraction', self._last_clip_frac / inner, update_step)
        writer.add_scalar('ppo/ratio_max', self._last_ratio_max, update_step)

        writer.add_scalar('lr/actor', optimizer.param_groups[0]['lr'], update_step)
        writer.add_scalar('lr/critic', optimizer.param_groups[1]['lr'], update_step)

        if avg_candidates > 1:
            max_entropy = float(np.log(avg_candidates))
            writer.add_scalar('train/entropy_normalized',
                            entropy_avg / max_entropy, update_step)

        # reset accumulators
        self._last_ratio_max = 0.0
        self._last_approx_kl = 0.0
        self._last_clip_frac = 0.0
        self._last_ppo_inner_steps = 0

    # ----- PPO learn step -----------------------------------------------------
    def learn(self, transitions, last_values, policy, optimizer, scheduler):
        """Run a PPO update over a chunk of stored transitions.

        ``transitions`` is a list of dicts (one entry per env step) with:
            states, batch_window_states, feasible_actions,
            actions, log_probs, values, rewards, dones
        Each tensor has leading batch dim B.
        ``last_values`` is V(s_{T}) used to bootstrap the final return; shape [B,1].
        """
        device = transitions[0]["values"].device
        gamma = args.gamma
        lam = args.gae_lambda
        clip = args.ppo_clip
        T = len(transitions)

        # ----- compute GAE advantages and returns (no grad) -----
        with torch.no_grad():
            advantages = [None] * T
            returns = [None] * T
            gae = torch.zeros_like(last_values)
            next_value = last_values
            # next_nonterminal for the *last* boundary is always 1.0 since we
            # are bootstrapping with last_values from a non-terminal state.
            next_nonterminal = torch.ones_like(last_values)
            for t in reversed(range(T)):
                reward = transitions[t]["rewards"]
                value = transitions[t]["values"]
                # Mask future returns by the post-step termination flag.
                post_done = transitions[t]["next_dones"].to(value.dtype)
                delta = reward + gamma * next_value * next_nonterminal - value
                gae = delta + gamma * lam * next_nonterminal * gae
                advantages[t] = gae
                returns[t] = gae + value
                next_value = value
                next_nonterminal = 1.0 - post_done

            adv_cat = torch.cat(advantages, dim=0)
            adv_mean = adv_cat.mean()
            adv_std = adv_cat.std(unbiased=False)
            adv_norm_const = adv_std + self.eps

            # === Critic / advantage 健康度统计（每次 learn 调用记录一次）===
            all_values = torch.cat([tr["values"] for tr in transitions])
            all_returns_cat = torch.cat(returns)
            all_rewards_cat = torch.cat([tr["rewards"] for tr in transitions])

            var_y = all_returns_cat.var()
            var_residual = (all_returns_cat - all_values).var()
            explained_variance = (1.0 - (var_residual / (var_y + 1e-8))).item()

            self._last_critic_stats = {
                'V_mean': all_values.mean().item(),
                'V_std': all_values.std().item(),
                'V_max': all_values.max().item(),
                'V_min': all_values.min().item(),
                'return_mean': all_returns_cat.mean().item(),
                'return_std': all_returns_cat.std().item(),
                'explained_variance': explained_variance,
                'adv_raw_mean': adv_cat.mean().item(),
                'adv_raw_std': adv_cat.std().item(),
                'reward_contribution': all_rewards_cat.mean().item(),
            }

        # ----- PPO optimization epochs -----
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_updates = 0

        for _ in range(args.ppo_epochs):
            perm = np.random.permutation(T)
            # 把 T 个 timestep 分成 chunks of size mb_steps
            mb_steps = max(1, args.ppo_minibatch_steps)  # 新加超参，建议 4~8
            for i in range(0, T, mb_steps):
                chunk_ts = perm[i:i + mb_steps]
                # 累积一组 timestep 的 loss，一次性 backward
                policy_loss_acc = 0.0
                value_loss_acc = 0.0
                entropy_acc = 0.0
                total_valid = 0.0

                optimizer.zero_grad()
                for t in chunk_ts:
                    tr = transitions[int(t)]
                    old_log_prob = tr["log_probs"].detach()
                    adv = (advantages[int(t)] - adv_mean) / adv_norm_const
                    ret = returns[int(t)]

                    new_log_prob, entropy, new_value = self._policy_evaluate(
                        policy,
                        tr["states"],
                        tr["batch_window_states"],
                        tr["feasible_actions"],
                        tr["actions"],
                    )

                    valid_mask = (~tr["dones"]).to(adv.dtype)
                    n_valid = valid_mask.sum()

                    ratio = torch.exp(new_log_prob - old_log_prob)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
                    # === PPO 内部诊断：ratio / KL / clip fraction ===
                    with torch.no_grad():
                        approx_kl_t = ((ratio - 1) - (new_log_prob - old_log_prob)).mean().item()
                        clip_frac_t = ((ratio - 1.0).abs() > clip).float().mean().item()
                        ratio_max_t = ratio.max().item()
                        self._last_ratio_max = max(self._last_ratio_max, ratio_max_t)
                        self._last_approx_kl += approx_kl_t
                        self._last_clip_frac += clip_frac_t
                        self._last_ppo_inner_steps += 1

                    policy_loss_t = -(torch.min(surr1, surr2) * valid_mask).sum()
                    value_loss_t = (((new_value - ret) ** 2) * valid_mask).sum()
                    entropy_t = (entropy * valid_mask).sum()

                    policy_loss_acc = policy_loss_acc + policy_loss_t
                    value_loss_acc = value_loss_acc + value_loss_t
                    entropy_acc = entropy_acc + entropy_t
                    total_valid = total_valid + n_valid

                total_valid = total_valid.clamp(min=1.0)
                policy_loss = policy_loss_acc / total_valid
                value_loss = value_loss_acc / total_valid
                entropy_bonus = entropy_acc / total_valid

                loss = (policy_loss
                        + args.value_coef * value_loss
                        - args.entropy_coef * entropy_bonus)

                loss.backward()
                if args.max_grad_norm and args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_policy_loss += float(policy_loss.detach().cpu().item())
                total_value_loss += float(value_loss.detach().cpu().item())
                total_entropy += float(entropy_bonus.detach().cpu().item())
                total_updates += 1

        if total_updates == 0:
            return 0.0, 0.0, 0.0
        return (total_policy_loss / total_updates,
                total_value_loss / total_updates,
                total_entropy / total_updates)

    # ----- validation ---------------------------------------------------------
    def validation(self, policy, dev):
        validation_start = time.time()
        policy.eval()

        val_transit = args.validation_transit if args.validation_transit > 0 else args.transit

        # Snapshot the policy on CPU; each worker forks a copy and loads it once.
        cpu_state_dict = {k: v.detach().cpu() for k, v in policy.state_dict().items()}

        actor_kwargs = dict(
            in_dim=3,
            hidden_dim=args.hidden_dim,
            embedding_l=args.embedding_layer,
            window_op_in_dim=10,
            window_mch_in_dim=8,
            policy_l=args.policy_layer,
            heads=args.heads,
            dropout=args.drop_out,
        )
        env_kwargs = dict(
            n_job=args.j,
            n_mch=args.m,
            low=args.l,
            high=args.h,
            cp_solver_time=self.cp_solver_time,
            cp_solver_cpu=1,
            cpu_budget=1,
            window_size=self.window_size,
        )

        n_instances = len(self.validation_data)
        num_workers = max(1, min(self.cpu_budget, n_instances))
        instances_list = [self.validation_data[i] for i in range(n_instances)]

        ctx = mp.get_context("fork")
        init_args = (
            actor_kwargs,
            env_kwargs,
            cpu_state_dict,
            args.init_type,
            val_transit,
            1,
        )
        with ctx.Pool(
            processes=num_workers,
            initializer=_validation_worker_init,
            initargs=init_args,
        ) as pool:
            final_objs = pool.map(_validation_worker_run, instances_list)

        validation_result = float(np.mean(final_objs))

        if validation_result < self.current_validation_result:
            print("Find better PPO CP-window model w.r.t final step objs, saving model...")
            save_dir = Path('./L2S_PPO/saved_model')
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / (
                "ppo-last-step-cp_"
                "{}x{}[{},{}]_"
                "{}_{}_{}_"
                "{}_{}_{}_"
                "{}_{}_{}_"
                "{}_{}_{}.pth".format(
                    args.j, args.m, args.l, args.h,
                    args.init_type, args.gamma, self._cp_run_tag(),
                    args.hidden_dim, args.policy_layer, self.dghan_param_for_saved_model,
                    args.lr, args.steps_learn, args.transit,
                    args.batch_size, args.episodes, args.step_validation,
                )
            )
            torch.save(policy.state_dict(), str(save_path))
            self.current_validation_result = validation_result

        validation_end = time.time()
        print(
            "Final step objs for PPO CP validation: {:.2f} | took {:.2f}s".format(
                validation_result, validation_end - validation_start
            )
        )

        policy.train()
        return validation_result

    # ----- main training loop -------------------------------------------------
    def train(self):
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(1)
        random.seed(1)
        np.random.seed(1)

        policy = Actor(
            in_dim=3,
            hidden_dim=args.hidden_dim,
            embedding_l=args.embedding_layer,
            window_op_in_dim=10,
            window_mch_in_dim=8,
            policy_l=args.policy_layer,
            heads=args.heads,
            dropout=args.drop_out,
        ).to(dev)

        actor_params = []
        critic_params = []
        for name, param in policy.named_parameters():
            if 'value_head' in name:
                critic_params.append(param)
            else:
                actor_params.append(param)

        optimizer = optim.Adam([
            {'params': actor_params, 'lr': args.lr},
            {'params': critic_params, 'lr': args.lr * 5},  # critic 学得更快
        ])

        # optimizer = optim.Adam(policy.parameters(), lr=args.lr)
        log_root = Path('./log_ppo')
        log_root.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(self.tensorboard_root / self._tb_run_name()))

        log = []
        validation_log = []
        train_step = 0
        batch_step = 0
        update_step = 0

        num_train = len(self.training_data)
        num_epochs = getattr(args, 'epochs', args.episodes)

        # 加入 scheduler
        total_updates = (num_train // args.batch_size) * num_epochs * (args.transit // args.steps_learn) * args.ppo_epochs
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: max(1.0 - step / total_updates, 0.0)
        )

        print()
        for epoch in range(1, num_epochs + 1):
            print(
                "Epoch {} | Training PPO CP-window environment with {} instances...".format(
                    epoch, num_train
                )
            )
            t1 = time.time()
            perm = np.random.permutation(num_train)
            epoch_final_objs = []
            epoch_reward_log = []
            epoch_policy_loss = []
            epoch_value_loss = []
            epoch_entropy = []

            for start in range(0, num_train, args.batch_size):
                batch_idx = perm[start:start + args.batch_size]
                instances = self.training_data[batch_idx]

                states, batch_window_states, feasible_actions, dones = self.env_training.reset(
                    instances=instances,
                    init_type=args.init_type,
                    device=dev,
                )

                policy.train()

                # Rollout buffer (list of per-step transition dicts).
                rollout = []

                while self.env_training.itr < args.transit:
                    # === 记录当前 state 的候选窗口数（平均/最大）===
                    num_candidates_per_batch = [len(fa) for fa in feasible_actions]
                    if num_candidates_per_batch:
                        avg_candidates = sum(num_candidates_per_batch) / len(num_candidates_per_batch)
                        max_candidates_val = max(num_candidates_per_batch)
                        writer.add_scalar('env/avg_num_candidates', avg_candidates, train_step)
                        writer.add_scalar('env/max_num_candidates', max_candidates_val, train_step)
                    else:
                        avg_candidates = 1.0

                    actions, log_ps, entropy, values, batch_states_pack = self._policy_act(
                        policy, states, batch_window_states, feasible_actions
                    )

                    next_states, next_batch_window_states, rewards, next_feasible_actions, next_dones = \
                        self.env_training.step(actions, dev)

                    reward_mean = rewards.mean().item()
                    current_obj_mean = self.env_training.current_objs.mean().item()
                    epoch_reward_log.append(reward_mean)
                    writer.add_scalar('train/reward_step', reward_mean, train_step)
                    writer.add_scalar('train/current_obj_step', current_obj_mean, train_step)
                    train_step += 1

                    # Store transition. Detach old log_probs/values — PPO uses
                    # them as targets in the loss. ``dones`` is the flag at the
                    # *start* of the step (used for masking placeholder
                    # log_probs in the loss); ``next_dones`` is the flag after
                    # the step (used for GAE bootstrap masking).
                    rollout.append({
                        "states": batch_states_pack,
                        "batch_window_states": batch_window_states,
                        "feasible_actions": feasible_actions,
                        "actions": list(actions),
                        "log_probs": log_ps.detach(),
                        "values": values.detach(),
                        "rewards": rewards.detach(),
                        "dones": dones,
                        "next_dones": next_dones,
                    })

                    states = next_states
                    batch_window_states = next_batch_window_states
                    feasible_actions = next_feasible_actions
                    dones = next_dones

                    if self.env_training.itr % args.steps_learn == 0 and len(rollout) > 0:
                        # Bootstrap with V(s_T) under the current policy.
                        with torch.no_grad():
                            batch_states_last = _pack_batch_states(states)
                            _, _, _, last_values = policy(
                                batch_states=batch_states_last,
                                batch_window_states=batch_window_states,
                                feasible_actions=feasible_actions,
                                return_value=True,
                            )
                        policy_loss, value_loss, entropy_avg = self.learn(
                            rollout, last_values.detach(), policy, optimizer, scheduler
                        )
                        epoch_policy_loss.append(policy_loss)
                        epoch_value_loss.append(value_loss)
                        epoch_entropy.append(entropy_avg)
                        self._log_ppo_update(
                            writer, update_step,
                            policy_loss, value_loss, entropy_avg,
                            optimizer, avg_candidates,
                        )
                        update_step += 1
                        update_step += 1
                        rollout = []

                # Trailing chunk shorter than steps_learn.
                if len(rollout) > 0:
                    with torch.no_grad():
                        batch_states_last = _pack_batch_states(states)
                        _, _, _, last_values = policy(
                            batch_states=batch_states_last,
                            batch_window_states=batch_window_states,
                            feasible_actions=feasible_actions,
                            return_value=True,
                        )
                    policy_loss, value_loss, entropy_avg = self.learn(
                        rollout, last_values.detach(), policy, optimizer, scheduler
                    )
                    epoch_policy_loss.append(policy_loss)
                    epoch_value_loss.append(value_loss)
                    epoch_entropy.append(entropy_avg)
                    self._log_ppo_update(
                        writer, update_step,
                        policy_loss, value_loss, entropy_avg,
                        optimizer, avg_candidates,
                    )
                    update_step += 1
                    update_step += 1

                batch_final_obj = self.env_training.current_objs.mean().item()
                epoch_final_objs.append(batch_final_obj)
                # === Batch 级性能指标（最重要的"是否在学"的判断指标）===
                initial_obj_mean = self.env_training.initial_objs.mean().item()
                best_obj_mean = self.env_training.best_objs.mean().item()
                improvement = initial_obj_mean - batch_final_obj
                improvement_ratio = improvement / max(initial_obj_mean, 1.0)

                writer.add_scalar('perf/initial_obj_per_batch', initial_obj_mean, batch_step)
                writer.add_scalar('perf/final_obj_per_batch', batch_final_obj, batch_step)
                writer.add_scalar('perf/best_obj_per_batch', best_obj_mean, batch_step)
                writer.add_scalar('perf/improvement_per_batch', improvement, batch_step)
                writer.add_scalar('perf/improvement_ratio_per_batch', improvement_ratio, batch_step)
                
                if batch_step % args.step_validation == 0:
                    validation_result = self.validation(policy, dev)
                    validation_log.append([validation_result])
                    writer.add_scalar("validation/final_obj_step", validation_result, batch_step)
                    print(
                        "Step {} | PPO CP-window validation final obj: {:.2f}".format(
                            batch_step, validation_result
                        )
                    )
                batch_step += 1

            t2 = time.time()

            epoch_obj_mean = float(np.mean(epoch_final_objs)) if epoch_final_objs else 0.0
            epoch_reward_mean = float(np.mean(epoch_reward_log)) if epoch_reward_log else 0.0
            epoch_reward_last = float(epoch_reward_log[-1]) if epoch_reward_log else 0.0
            epoch_policy_loss_mean = float(np.mean(epoch_policy_loss)) if epoch_policy_loss else 0.0
            epoch_value_loss_mean = float(np.mean(epoch_value_loss)) if epoch_value_loss else 0.0
            epoch_entropy_mean = float(np.mean(epoch_entropy)) if epoch_entropy else 0.0

            print(
                'Epoch {:4d} | time {:.2f}s | pi_loss {:.4f} | v_loss {:.4f} | ent {:.4f} | '
                'reward_mean {:.4f} | reward_last {:.4f} | final_obj_mean {:.2f}'.format(
                    epoch, t2 - t1,
                    epoch_policy_loss_mean, epoch_value_loss_mean, epoch_entropy_mean,
                    epoch_reward_mean, epoch_reward_last, epoch_obj_mean,
                )
            )

            log.append(epoch_obj_mean)
            writer.add_scalar('train/final_obj_epoch', epoch_obj_mean, epoch)
            writer.add_scalar('train/reward_epoch_mean', epoch_reward_mean, epoch)
            writer.add_scalar('train/policy_loss_epoch_mean', epoch_policy_loss_mean, epoch)
            writer.add_scalar('train/value_loss_epoch_mean', epoch_value_loss_mean, epoch)

            validation_result = self.validation(policy, dev)
            validation_log.append([validation_result])
            writer.add_scalar('validation/final_obj', validation_result, epoch)

            np.save(
                str(log_root / (
                    'training_log_ppo_cp_'
                    '{}x{}[{},{}]_{}_{}_{}_'
                    '{}_{}_{}_{}_{}_'
                    '{}_{}_{}_{}_{}_{}.npy'
                    .format(
                        args.j, args.m, args.l, args.h,
                        args.init_type, args.gamma, self._cp_run_tag(),
                        args.hidden_dim, args.embedding_layer, args.policy_layer,
                        getattr(args, 'embedding_type', 'na'),
                        self.dghan_param_for_saved_model,
                        args.lr, args.steps_learn, args.transit,
                        args.batch_size, num_epochs, args.step_validation,
                    )
                )),
                np.array(log),
            )

            np.save(
                str(log_root / (
                    'validation_log_ppo_cp_'
                    '{}x{}[{},{}]_{}_{}_{}_'
                    '{}_{}_{}_{}_{}_'
                    '{}_{}_{}_{}_{}_{}.npy'
                    .format(
                        args.j, args.m, args.l, args.h,
                        args.init_type, args.gamma, self._cp_run_tag(),
                        args.hidden_dim, args.embedding_layer, args.policy_layer,
                        getattr(args, 'embedding_type', 'na'),
                        self.dghan_param_for_saved_model,
                        args.lr, args.steps_learn, args.transit,
                        args.batch_size, num_epochs, args.step_validation,
                    )
                )),
                np.array(validation_log),
            )

        writer.close()


if __name__ == '__main__':
    affinity = _bind_main_process_away_from_cpu0()
    if affinity is not None:
        print("CPU affinity:", affinity)
    agent = PPO_L2S_CPJSSP()
    agent.train()
