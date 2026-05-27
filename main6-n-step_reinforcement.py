import time
import random
import os
import numpy as np
import numpy.random
from L2S.parameters import args
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from L2S.env.environment import JsspWindow, BatchGraph
from L2S.model.actor import Actor
from L2S.env.generateJSP import uni_instance_gen
from pathlib import Path
from types import SimpleNamespace


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
    """
    states from env:
        (x, edge_index_pc, edge_index_mc, batch)
    """
    x, edge_index_pc, edge_index_mc, batch = states
    return SimpleNamespace(
        x=x,
        edge_index_pc=edge_index_pc,
        edge_index_mc=edge_index_mc,
        batch=batch,
    )


class RL2SCPJSSP:
    def __init__(self):
        self.window_size = getattr(args, 'window_size', args.j * args.m)
        self.cp_solver_time = getattr(args, 'cp_solver_time', 1)
        self.cp_solver_cpu = getattr(args, 'cp_solver_cpu', 1)
        self.cpu_budget = getattr(args, 'cpu_budget', 16)
        self.tensorboard_root = Path(getattr(args, 'tensorboard_log_dir', './tensorboard_log'))
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

        self.dghan_param_for_saved_model = '{}_{}'.format(args.heads, args.drop_out)

        train_data_path = Path(
            "./L2S/training_data/training_instance_{}x{}[{},{}].npy".format(
                args.j, args.m, args.l, args.h
            )
        )
        if train_data_path.is_file():
            self.training_data = np.load(str(train_data_path))
        else:
            self.training_data = np.array([
                uni_instance_gen(n_j=args.j, n_m=args.m, low=args.l, high=args.h)
                for _ in range(args.training_set_size)
            ])
            np.save(str(train_data_path), self.training_data)

        validation_data_path = Path(
            './L2S/validation_data/validation_instance_{}x{}[{},{}].npy'.format(args.j, args.m, args.l, args.h))
        if validation_data_path.is_file():
            self.validation_data = np.load(
                './L2S/validation_data/validation_instance_{}x{}[{},{}].npy'.format(args.j, args.m, args.l, args.h))
        else:
            print('No validation data for {}x{}[{},{}], generating new one.'.format(args.j, args.m, args.l, args.h))
            self.validation_data = np.array(
                [uni_instance_gen(n_j=args.j, n_m=args.m, low=args.l, high=args.h) for _ in range(100)])
            np.save('./L2S/validation_data/validation_instance_{}x{}[{},{}].npy'.format(args.j, args.m, args.l, args.h),
                    self.validation_data)

    def _cp_run_tag(self):
        """Create a file-name-safe tag for the CP-window environment settings."""
        return 'cp{}_{}_wb{}_ws{}'.format(
            self.cp_solver_time,
            self.cp_solver_cpu,
            self.cpu_budget,
            self.window_size,
        )

    def _tb_run_name(self):
        return ('rl2scp_'
                '{}x{}[{},{}]_'
                '{}_{}_{}_'
                '{}_{}_{}_'
                '{}_'
                '{}_{}_{}_'
                '{}_{}_{}'
                .format(args.j, args.m, args.l, args.h, 
                        args.init_type,args.gamma, self._cp_run_tag(),
                        args.hidden_dim, args.embedding_layer, args.policy_layer, 
                        self.dghan_param_for_saved_model,
                        args.lr, args.steps_learn, args.transit, 
                        args.batch_size, args.episodes, args.step_validation))
    
    def _policy_act(self, policy, states, batch_window_states, feasible_actions):
        batch_states = _pack_batch_states(states)
        actions, log_ps = policy(
            batch_states=batch_states,
            batch_window_states=batch_window_states,
            feasible_actions=feasible_actions,
        )
        return actions, log_ps

    def _run_episode(self, env, policy, dev, instances):
        """Roll one full trajectory on the provided environment."""
        states, batch_window_states, feasible_actions, dones = env.reset(
            instances=instances,
            init_type=args.init_type,
            device=dev,
        )

        rewards_buffer = []
        log_probs_buffer = []
        dones_buffer = [dones]

        while env.itr < args.transit:
            actions, log_ps = self._policy_act(
                policy, states, batch_window_states, feasible_actions
            )
            states, batch_window_states, rewards, feasible_actions, dones = env.step(actions, dev)
            rewards_buffer.append(rewards)
            log_probs_buffer.append(log_ps)
            dones_buffer.append(dones)

        return rewards_buffer, log_probs_buffer, dones_buffer

    def learn(self, rewards, log_probs, dones, 
              # time_weights, 
              optimizer):
        R = torch.zeros_like(rewards[0], dtype=torch.float, device=rewards[0].device)
        returns = []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.cat(returns, dim=-1)
        dones = torch.cat(dones, dim=-1)
        log_probs = torch.cat(log_probs, dim=-1)
        # time_weights = torch.cat(time_weights, dim=-1)

        losses = []
        for b in range(returns.shape[0]):
            masked_R = torch.masked_select(returns[b], ~dones[b])
            masked_R = (masked_R - masked_R.mean()) / (torch.std(masked_R, unbiased=False) + self.eps)
            masked_log_prob = torch.masked_select(log_probs[b], ~dones[b])
            # masked_time_weight = torch.masked_select(time_weights[b], ~dones[b]) 
            loss = (- masked_log_prob * masked_R).sum()
            losses.append(loss)

        optimizer.zero_grad()
        mean_loss = torch.stack(losses).mean()
        mean_loss.backward()
        optimizer.step()
        return mean_loss.detach().cpu().item()

    def validation(self, policy, dev):
        validation_start = time.time()
        policy.eval()

        val_transit = args.validation_transit if args.validation_transit > 0 else args.transit

        with torch.no_grad():
            states_val, batch_window_states_val, feasible_actions_val, _ = self.env_validation.reset(
                instances=self.validation_data,
                init_type=args.init_type,
                device=dev,
            )

            while self.env_validation.itr < val_transit:
                actions_val, _ = self._policy_act(
                    policy,
                    states_val,
                    batch_window_states_val,
                    feasible_actions_val,
                )
                states_val, batch_window_states_val, _, feasible_actions_val, _ = self.env_validation.step(
                    actions_val,
                    dev,
                )

        validation_result = self.env_validation.current_objs.mean().cpu().item()

        if validation_result < self.current_validation_result:
            print("Find better CP-window model w.r.t final step objs, saving model...")
            torch.save(
                policy.state_dict(),
                "./L2S/saved_model/last-step-cp_"
                "{}x{}[{},{}]_" \
                "{}_{}_{}_" \
                "{}_{}_{}_" \
                "{}_{}_{}_" \
                "{}_{}_{}" \
                ".pth".format(
                    args.j, args.m, args.l, args.h,
                    args.init_type, args.gamma, self._cp_run_tag(),
                    args.hidden_dim, args.policy_layer, self.dghan_param_for_saved_model,
                    args.lr, args.steps_learn, args.transit,
                    args.batch_size, args.episodes, args.step_validation
                )
            )
            self.current_validation_result = validation_result

        validation_end = time.time()
        print(
            "Final step objs for CP validation are: {:.2f}".format(
                validation_result
            ),
            "validation takes:{:.2f}".format(validation_end - validation_start),
        )

        policy.train()
        return validation_result

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

        optimizer = optim.Adam(policy.parameters(), lr=args.lr)
        writer = SummaryWriter(log_dir=str(self.tensorboard_root / self._tb_run_name()))

        log = []
        validation_log = []
        train_step = 0
        batch_step = 0
        update_step = 0

        num_train = len(self.training_data)
        num_epochs = getattr(args, 'epochs', args.episodes)

        print()
        for epoch in range(1, num_epochs + 1):
            print(
                "Epoch {} | Training CP-window environment with {} training instances...".format(
                    epoch, num_train
                )
            )
            t1 = time.time()
            perm = np.random.permutation(num_train)
            # instances = np.array([
            #     uni_instance_gen(args.j, args.m, args.l, args.h)
            #     for _ in range(args.batch_size)
            # ])
            epoch_final_objs = []
            epoch_reward_log = []
            epoch_loss_log = []
            indices = np.random.permutation(len(self.training_data))
            for start in range(0, len(indices), args.batch_size):
                batch_idx = perm[start:start + args.batch_size]
                instances = self.training_data[batch_idx]

                states, batch_window_states, feasible_actions, dones = self.env_training.reset(
                    instances=instances,
                    init_type=args.init_type,
                    device=dev,
                )

                rewards_buffer = []
                log_probs_buffer = []
                dones_buffer = [dones]
                time_weight_buffer = []

                policy.train()

                while self.env_training.itr < args.transit:
                    actions, log_ps = self._policy_act(
                        policy,
                        states,
                        batch_window_states,
                        feasible_actions,
                    )

                    states, batch_window_states, rewards, feasible_actions, dones = self.env_training.step(
                        actions,
                        dev
                    )

                    progress = self.env_training.itr / max(args.transit - 1, 1)   # 0 ~ 1
                    # time_coef = 1.0 + args.time_weight_beta * progress            # 例如 beta=1.0，则权重 1~2
                    # time_coef_tensor = torch.full_like(rewards, fill_value=time_coef, dtype=torch.float)

                    reward_mean = rewards.mean().item()
                    current_obj_mean = self.env_training.current_objs.mean().item()
                    #time_coef_tensor_mean = time_coef_tensor.mean().item()

                    epoch_reward_log.append(reward_mean)
                    writer.add_scalar('train/reward_step', reward_mean, train_step)
                    writer.add_scalar('train/current_obj_step', current_obj_mean, train_step)
                    # writer.add_scalar('train/time_coef_step', time_coef_tensor_mean, train_step)
                    train_step += 1

                    rewards_buffer.append(rewards)
                    log_probs_buffer.append(log_ps)
                    dones_buffer.append(dones)
                    # time_weight_buffer.append(time_coef_tensor)

                    if self.env_training.itr % args.steps_learn == 0:
                        loss_value = self.learn(
                            rewards_buffer,
                            log_probs_buffer,
                            dones_buffer[:-1],
                            # time_weight_buffer, 
                            optimizer
                        )
                        epoch_loss_log.append(loss_value)
                        writer.add_scalar("train/loss", loss_value, update_step)
                        update_step += 1

                        rewards_buffer = []
                        log_probs_buffer = []
                        dones_buffer = [dones]
                        time_weight_buffer = []

                # 补上最后一个未满 steps_learn 的尾块
                if len(rewards_buffer) > 0:
                    loss_value = self.learn(
                        rewards_buffer,
                        log_probs_buffer,
                        dones_buffer[:-1],
                        optimizer
                    )
                    epoch_loss_log.append(loss_value)
                    writer.add_scalar("train/loss", loss_value, update_step)
                    update_step += 1
                
                batch_final_obj = self.env_training.current_objs.mean().item()
                epoch_final_objs.append(batch_final_obj)
                if batch_step % args.step_validation == 0:
                    validation_result= self.validation(policy, dev)
                    validation_log.append([validation_result])
                    writer.add_scalar("validation/final_obj_step", validation_result, batch_step)
                    print(
                        "Step {} | CP-window validation final obj: {:.2f}".format(
                            batch_step, validation_result
                        )
                    )
                batch_step += 1

            t2 = time.time()

            epoch_obj_mean = float(np.mean(epoch_final_objs)) if epoch_final_objs else 0.0
            epoch_reward_mean = float(np.mean(epoch_reward_log)) if epoch_reward_log else 0.0
            epoch_reward_last = float(epoch_reward_log[-1]) if epoch_reward_log else 0.0
            epoch_loss_mean = float(np.mean(epoch_loss_log)) if epoch_loss_log else 0.0
            epoch_loss_last = float(epoch_loss_log[-1]) if epoch_loss_log else 0.0

            print(
                'Epoch {:4d} | time {:.2f}s | loss_mean {:.4f} | loss_last {:.4f} | '
                'reward_mean {:.4f} | reward_last {:.4f} | final_obj_mean {:.2f}'.format(
                    epoch,
                    t2 - t1,
                    epoch_loss_mean,
                    epoch_loss_last,
                    epoch_reward_mean,
                    epoch_reward_last,
                    epoch_obj_mean,
                )
            )

            log.append(epoch_obj_mean)
            writer.add_scalar('train/final_obj_epoch', epoch_obj_mean, epoch)
            writer.add_scalar('train/reward_epoch_mean', epoch_reward_mean, epoch)
            writer.add_scalar('train/loss_epoch_mean', epoch_loss_mean, epoch)

            # epoch_test
            validation_result= self.validation(policy, dev)
            validation_log.append([validation_result])
            writer.add_scalar('validation/final_obj', validation_result, epoch)

            np.save(
                './log/training_log_cp_'
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
                    args.batch_size, num_epochs, args.step_validation
                ),
                np.array(log)
            )

            np.save(
                './log/validation_log_cp_'
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
                    args.batch_size, num_epochs, args.step_validation
                ),
                np.array(validation_log)
            )

        writer.close()

if __name__ == '__main__':
    affinity = _bind_main_process_away_from_cpu0()
    if affinity is not None:
        print("CPU affinity:", affinity)
    agent = RL2SCPJSSP()
    agent.train()
