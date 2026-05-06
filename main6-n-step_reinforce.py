import os
import time
import random
from pathlib import Path

import numpy as np
import numpy.random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from L2S.parameters import args
from L2S.env.environment import JsspN5, JsspWindow, BatchGraph
from L2S.model.actor import Actor
from L2S.env.generateJSP import uni_instance_gen


ROOT_DIR = Path(__file__).resolve().parent
L2S_DIR = ROOT_DIR / 'L2S'
VALIDATION_DIR = L2S_DIR / 'validation_data'
MODEL_DIR = L2S_DIR / 'saved_model'
LOG_DIR = ROOT_DIR / 'log'


def _ensure_dirs():
    """Create output directories used by training/validation if they do not exist."""
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _bind_main_process_away_from_cpu0():
    """Prefer non-zero CPUs for the training process when affinity control is available."""
    if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
        return None
    current = set(os.sched_getaffinity(0))
    allowed = {cpu for cpu in current if cpu != 0}
    target = allowed if allowed else current
    os.sched_setaffinity(0, target)
    return set(os.sched_getaffinity(0))


def _reward_tag():
    return '{}_zip{:g}'.format(args.reward_type, args.zero_improvement_penalty)


class RL2S4JSSP:
    def __init__(self):
        _ensure_dirs()
        self.env_training = JsspN5(
            n_job=args.j,
            n_mch=args.m,
            low=args.l,
            high=args.h,
            reward_type=args.reward_type,
            zero_improvement_penalty=args.zero_improvement_penalty,
        )
        self.env_validation = JsspN5(
            n_job=args.j,
            n_mch=args.m,
            low=args.l,
            high=args.h,
            reward_type=args.reward_type,
            zero_improvement_penalty=args.zero_improvement_penalty,
        )
        self.eps = np.finfo(np.float32).eps.item()
        self.tensorboard_root = Path(getattr(args, 'tensorboard_log_dir', './runs'))
        validation_data_path = VALIDATION_DIR / 'validation_instance_{}x{}[{},{}].npy'.format(
            args.j, args.m, args.l, args.h
        )
        if validation_data_path.is_file():
            self.validation_data = np.load(validation_data_path)
        else:
            print('No validation data for {}x{}[{},{}], generating new one.'.format(args.j, args.m, args.l, args.h))
            self.validation_data = np.array(
                [uni_instance_gen(n_j=args.j, n_m=args.m, low=args.l, high=args.h) for _ in range(100)]
            )
            np.save(validation_data_path, self.validation_data)
        self.incumbent_validation_result = np.inf
        self.current_validation_result = np.inf

        if args.embedding_type == 'gin':
            self.dghan_param_for_saved_model = 'NAN'
        elif args.embedding_type in ('dghan', 'gin+dghan'):
            self.dghan_param_for_saved_model = '{}_{}'.format(args.heads, args.drop_out)
        else:
            raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')

    def _tb_run_name(self):
        return (
            'rl2s_'
            '{}x{}[{},{}]_{}_{}_{}_'
            '{}_{}_{}_{}_{}_'
            '{}_{}_{}_{}_{}_{}'
        ).format(
            args.j, args.m, args.l, args.h, args.init_type, _reward_tag(), args.gamma,
            args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
            args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation,
        )

    def learn(self, rewards, log_probs, dones, optimizer):
        R = torch.zeros_like(rewards[0], dtype=torch.float, device=rewards[0].device)
        returns = []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.cat(returns, dim=-1)
        dones = torch.cat(dones, dim=-1)
        log_probs = torch.cat(log_probs, dim=-1)

        losses = []
        for b in range(returns.shape[0]):
            masked_R = torch.masked_select(returns[b], ~dones[b])
            masked_R = (masked_R - masked_R.mean()) / (torch.std(masked_R, unbiased=False) + self.eps)
            masked_log_prob = torch.masked_select(log_probs[b], ~dones[b])
            loss = (-masked_log_prob * masked_R).sum()
            losses.append(loss)

        optimizer.zero_grad()
        mean_loss = torch.stack(losses).mean()
        mean_loss.backward()
        optimizer.step()
        return mean_loss.detach().cpu().item()

    def validation(self, policy, dev):
        validation_start = time.time()
        validation_batch_data = BatchGraph()
        states_val, feasible_actions_val, _ = self.env_validation.reset(
            instances=self.validation_data,
            init_type=args.init_type,
            device=dev,
        )
        while self.env_validation.itr < args.transit:
            validation_batch_data.wrapper(*states_val)
            actions_val, _ = policy(validation_batch_data, feasible_actions_val)
            states_val, _, feasible_actions_val, _ = self.env_validation.step(actions_val, dev)

        validation_batch_data.clean()
        validation_result1 = self.env_validation.incumbent_objs.mean().cpu().item()
        validation_result2 = self.env_validation.current_objs.mean().cpu().item()
        if validation_result1 < self.incumbent_validation_result:
            print('Find better model w.r.t incumbent objs, saving model...')
            torch.save(
                policy.state_dict(),
                MODEL_DIR / (
                    'incumbent_'
                    '{}x{}[{},{}]_{}_{}_{}_'
                    '{}_{}_{}_{}_{}_'
                    '{}_{}_{}_{}_{}_{}'
                    '.pth'
                ).format(
                    args.j, args.m, args.l, args.h, args.init_type, _reward_tag(), args.gamma,
                    args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
                    args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation,
                ),
            )
            self.incumbent_validation_result = validation_result1
        if validation_result2 < self.current_validation_result:
            print('Find better model w.r.t final step objs, saving model...')
            torch.save(
                policy.state_dict(),
                MODEL_DIR / (
                    'last-step_'
                    '{}x{}[{},{}]_{}_{}_{}_'
                    '{}_{}_{}_{}_{}_'
                    '{}_{}_{}_{}_{}_{}'
                    '.pth'
                ).format(
                    args.j, args.m, args.l, args.h, args.init_type, _reward_tag(), args.gamma,
                    args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
                    args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation,
                ),
            )
            self.current_validation_result = validation_result2

        validation_end = time.time()
        print(
            'Incumbent objs and final step objs for validation are: {:.2f}  {:.2f}'.format(
                validation_result1, validation_result2
            ),
            'validation takes:{:.2f}'.format(validation_end - validation_start),
        )
        return validation_result1, validation_result2

    def train(self):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        torch.manual_seed(1)
        random.seed(1)
        np.random.seed(1)

        policy = Actor(
            in_dim=3,
            hidden_dim=args.hidden_dim,
            embedding_l=args.embedding_layer,
            policy_l=args.policy_layer,
            embedding_type=args.embedding_type,
            heads=args.heads,
            dropout=args.drop_out,
        ).to(dev)

        optimizer = optim.Adam(policy.parameters(), lr=args.lr)
        writer = SummaryWriter(log_dir=str(self.tensorboard_root / self._tb_run_name()))

        batch_data = BatchGraph()
        log = []
        validation_log = []
        train_step = 0
        update_step = 0

        print()
        for batch_i in range(1, args.episodes // args.batch_size + 1):
            t1 = time.time()
            instances = np.array([uni_instance_gen(args.j, args.m, args.l, args.h) for _ in range(args.batch_size)])
            states, feasible_actions, dones = self.env_training.reset(
                instances=instances,
                init_type=args.init_type,
                device=dev,
            )

            reward_log = []
            loss_log = []
            rewards_buffer = []
            log_probs_buffer = []
            dones_buffer = [dones]

            while self.env_training.itr < args.transit:
                batch_data.wrapper(*states)
                actions, log_ps = policy(batch_data, feasible_actions)
                states, rewards, feasible_actions, dones = self.env_training.step(actions, dev)
                reward_mean = rewards.mean().cpu().item()
                reward_log.append(reward_mean)
                writer.add_scalar('train/reward_step', reward_mean, train_step)
                writer.add_scalar('train/current_obj_step', self.env_training.current_objs.mean().cpu().item(), train_step)
                writer.add_scalar('train/incumbent_obj_step', self.env_training.incumbent_objs.mean().cpu().item(), train_step)
                train_step += 1

                rewards_buffer.append(rewards)
                log_probs_buffer.append(log_ps)
                dones_buffer.append(dones)

                if self.env_training.itr % args.steps_learn == 0:
                    loss_value = self.learn(rewards_buffer, log_probs_buffer, dones_buffer[:-1], optimizer)
                    loss_log.append(loss_value)
                    writer.add_scalar('train/loss', loss_value, update_step)
                    update_step += 1
                    rewards_buffer = []
                    log_probs_buffer = []
                    dones_buffer = [dones]

            t2 = time.time()
            reward_mean_epoch = float(np.mean(reward_log)) if reward_log else 0.0
            reward_last_epoch = float(reward_log[-1]) if reward_log else 0.0
            loss_mean_epoch = float(np.mean(loss_log)) if loss_log else 0.0
            loss_last_epoch = float(loss_log[-1]) if loss_log else 0.0
            print(
                'Epoch {:4d} | time {:.2f}s | loss_mean {:.4f} | loss_last {:.4f} | '
                'reward_mean {:.4f} | reward_last {:.4f} | obj {:.2f}'.format(
                    batch_i,
                    t2 - t1,
                    loss_mean_epoch,
                    loss_last_epoch,
                    reward_mean_epoch,
                    reward_last_epoch,
                    self.env_training.current_objs.cpu().mean().item(),
                )
            )
            log.append(self.env_training.current_objs.mean().cpu().item())
            writer.add_scalar('train/final_obj_batch', self.env_training.current_objs.mean().cpu().item(), batch_i)
            writer.add_scalar('train/incumbent_obj_batch', self.env_training.incumbent_objs.mean().cpu().item(), batch_i)
            if reward_log:
                writer.add_scalar('train/reward_batch_mean', float(np.mean(reward_log)), batch_i)

            if batch_i % args.step_validation == 0:
                validation_result1, validation_result2 = self.validation(policy, dev)
                validation_log.append([validation_result1, validation_result2])
                writer.add_scalar('validation/incumbent_obj', validation_result1, batch_i)
                writer.add_scalar('validation/final_obj', validation_result2, batch_i)

                np.save(
                    LOG_DIR / (
                        'training_log_'
                        '{}x{}[{},{}]_{}_{}_{}_'
                        '{}_{}_{}_{}_{}_'
                        '{}_{}_{}_{}_{}_{}.npy'
                    ).format(
                        args.j, args.m, args.l, args.h, args.init_type, _reward_tag(), args.gamma,
                        args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type,
                        self.dghan_param_for_saved_model,
                        args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation,
                    ),
                    np.array(log),
                )
                np.save(
                    LOG_DIR / (
                        'validation_log_'
                        '{}x{}[{},{}]_{}_{}_{}_'
                        '{}_{}_{}_{}_{}_'
                        '{}_{}_{}_{}_{}_{}.npy'
                    ).format(
                        args.j, args.m, args.l, args.h, args.init_type, _reward_tag(), args.gamma,
                        args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type,
                        self.dghan_param_for_saved_model,
                        args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation,
                    ),
                    np.array(validation_log),
                )
        writer.close()


class RL2SCPJSSP:
    def __init__(self):
        _ensure_dirs()
        self.window_size = getattr(args, 'window_size', args.j * args.m)
        self.cp_solver_time = getattr(args, 'cp_solver_time', 1)
        self.cp_solver_cpu = getattr(args, 'cp_solver_cpu', 1)
        self.cpu_budget = getattr(args, 'cpu_budget', getattr(args, 'cp_solver_budget', 16))
        self.tensorboard_root = Path(getattr(args, 'tensorboard_log_dir', './runs'))
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
            log_path=self.training_env_log_path,
            reward_type=args.reward_type,
            zero_improvement_penalty=args.zero_improvement_penalty,
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
            log_path=self.validation_env_log_path,
            reward_type=args.reward_type,
            zero_improvement_penalty=args.zero_improvement_penalty,
        )
        self.eps = np.finfo(np.float32).eps.item()
        validation_data_path = VALIDATION_DIR / 'validation_instance_{}x{}[{},{}].npy'.format(
            args.j, args.m, args.l, args.h
        )
        if validation_data_path.is_file():
            self.validation_data = np.load(validation_data_path)
        else:
            print('No validation data for {}x{}[{},{}], generating new one.'.format(args.j, args.m, args.l, args.h))
            self.validation_data = np.array(
                [uni_instance_gen(n_j=args.j, n_m=args.m, low=args.l, high=args.h) for _ in range(100)]
            )
            np.save(validation_data_path, self.validation_data)
        self.incumbent_validation_result = np.inf
        self.current_validation_result = np.inf

        if args.embedding_type == 'gin':
            self.dghan_param_for_saved_model = 'NAN'
        elif args.embedding_type in ('dghan', 'gin+dghan'):
            self.dghan_param_for_saved_model = '{}_{}'.format(args.heads, args.drop_out)
        else:
            raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')

    def _cp_run_tag(self):
        return 'cp{}_{}_wb{}_ws{}'.format(
            self.cp_solver_time,
            self.cp_solver_cpu,
            self.cpu_budget,
            self.window_size,
        )

    def _tb_run_name(self):
        return (
            'rl2scp_'
            '{}x{}[{},{}]_{}_{}_{}_{}_'
            '{}_{}_{}_{}_{}_'
            '{}_{}_{}_{}_{}_{}'
        ).format(
            args.j, args.m, args.l, args.h, args.init_type, _reward_tag(), args.gamma, self._cp_run_tag(),
            args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
            args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation,
        )

    def learn(self, rewards, log_probs, dones, optimizer):
        R = torch.zeros_like(rewards[0], dtype=torch.float, device=rewards[0].device)
        returns = []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.cat(returns, dim=-1)
        dones = torch.cat(dones, dim=-1)
        log_probs = torch.cat(log_probs, dim=-1)

        losses = []
        for b in range(returns.shape[0]):
            masked_R = torch.masked_select(returns[b], ~dones[b])
            masked_R = (masked_R - masked_R.mean()) / (torch.std(masked_R, unbiased=False) + self.eps)
            masked_log_prob = torch.masked_select(log_probs[b], ~dones[b])
            loss = (-masked_log_prob * masked_R).sum()
            losses.append(loss)

        optimizer.zero_grad()
        mean_loss = torch.stack(losses).mean()
        mean_loss.backward()
        optimizer.step()
        return mean_loss.detach().cpu().item()

    def validation(self, policy, dev):
        validation_start = time.time()
        validation_batch_data = BatchGraph()
        states_val, feasible_actions_val, _ = self.env_validation.reset(
            instances=self.validation_data,
            init_type=args.init_type,
            device=dev,
        )
        while self.env_validation.itr < args.transit:
            validation_batch_data.wrapper(*states_val)
            actions_val, _ = policy(validation_batch_data, feasible_actions_val)
            states_val, _, feasible_actions_val, _ = self.env_validation.step(actions_val, dev)

        validation_batch_data.clean()
        validation_result1 = self.env_validation.incumbent_objs.mean().cpu().item()
        validation_result2 = self.env_validation.current_objs.mean().cpu().item()
        if validation_result1 < self.incumbent_validation_result:
            print('Find better CP-window model w.r.t incumbent objs, saving model...')
            torch.save(
                policy.state_dict(),
                MODEL_DIR / (
                    'incumbent-cp_'
                    '{}x{}[{},{}]_{}_{}_{}_{}_'
                    '{}_{}_{}_{}_{}_'
                    '{}_{}_{}_{}_{}_{}'
                    '.pth'
                ).format(
                    args.j, args.m, args.l, args.h, args.init_type, _reward_tag(), args.gamma, self._cp_run_tag(),
                    args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
                    args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation,
                ),
            )
            self.incumbent_validation_result = validation_result1
        if validation_result2 < self.current_validation_result:
            print('Find better CP-window model w.r.t final step objs, saving model...')
            torch.save(
                policy.state_dict(),
                MODEL_DIR / (
                    'last-step-cp_'
                    '{}x{}[{},{}]_{}_{}_{}_{}_'
                    '{}_{}_{}_{}_{}_'
                    '{}_{}_{}_{}_{}_{}'
                    '.pth'
                ).format(
                    args.j, args.m, args.l, args.h, args.init_type, _reward_tag(), args.gamma, self._cp_run_tag(),
                    args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
                    args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation,
                ),
            )
            self.current_validation_result = validation_result2

        validation_end = time.time()
        print(
            'Incumbent objs and final step objs for CP validation are: {:.2f}  {:.2f}'.format(
                validation_result1, validation_result2
            ),
            'validation takes:{:.2f}'.format(validation_end - validation_start),
        )
        return validation_result1, validation_result2

    def train(self):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        torch.manual_seed(1)
        random.seed(1)
        np.random.seed(1)

        policy = Actor(
            in_dim=3,
            hidden_dim=args.hidden_dim,
            embedding_l=args.embedding_layer,
            policy_l=args.policy_layer,
            embedding_type=args.embedding_type,
            heads=args.heads,
            dropout=args.drop_out,
        ).to(dev)

        optimizer = optim.Adam(policy.parameters(), lr=args.lr)
        writer = SummaryWriter(log_dir=str(self.tensorboard_root / self._tb_run_name()))
        batch_data = BatchGraph()
        log = []
        validation_log = []
        train_step = 0
        update_step = 0

        print()
        for batch_i in range(1, args.episodes // args.batch_size + 1):
            t1 = time.time()
            instances = np.array([uni_instance_gen(args.j, args.m, args.l, args.h) for _ in range(args.batch_size)])
            states, feasible_actions, dones = self.env_training.reset(
                instances=instances,
                init_type=args.init_type,
                device=dev,
            )

            rewards_buffer = []
            log_probs_buffer = []
            dones_buffer = [dones]
            reward_log = []
            loss_log = []

            while self.env_training.itr < args.transit:
                batch_data.wrapper(*states)
                actions, log_ps = policy(batch_data, feasible_actions)
                states, rewards, feasible_actions, dones = self.env_training.step(actions, dev)
                # print('Batch {}, step {}, reward_each: {}'.format(batch_i, self.env_training.itr, rewards.cpu().numpy()))
                reward_mean = rewards.mean().cpu().item()
                reward_log.append(reward_mean)
                writer.add_scalar('train/reward_step', reward_mean, train_step)
                writer.add_scalar('train/current_obj_step', self.env_training.current_objs.mean().cpu().item(), train_step)
                writer.add_scalar('train/incumbent_obj_step', self.env_training.incumbent_objs.mean().cpu().item(), train_step)
                train_step += 1

                rewards_buffer.append(rewards)
                log_probs_buffer.append(log_ps)
                dones_buffer.append(dones)

                if self.env_training.itr % args.steps_learn == 0:
                    loss_value = self.learn(rewards_buffer, log_probs_buffer, dones_buffer[:-1], optimizer)
                    loss_log.append(loss_value)
                    writer.add_scalar('train/loss', loss_value, update_step)
                    update_step += 1
                    rewards_buffer = []
                    log_probs_buffer = []
                    dones_buffer = [dones]

            t2 = time.time()
            reward_mean_epoch = float(np.mean(reward_log)) if reward_log else 0.0
            reward_last_epoch = float(reward_log[-1]) if reward_log else 0.0
            loss_mean_epoch = float(np.mean(loss_log)) if loss_log else 0.0
            loss_last_epoch = float(loss_log[-1]) if loss_log else 0.0
            print(
                'CP Epoch {:4d} | time {:.2f}s | loss_mean {:.4f} | loss_last {:.4f} | '
                'reward_mean {:.4f} | reward_last {:.4f} | obj {:.2f}'.format(
                    batch_i,
                    t2 - t1,
                    loss_mean_epoch,
                    loss_last_epoch,
                    reward_mean_epoch,
                    reward_last_epoch,
                    self.env_training.current_objs.cpu().mean().item(),
                )
            )
            log.append(self.env_training.current_objs.mean().cpu().item())
            writer.add_scalar('train/final_obj_batch', self.env_training.current_objs.mean().cpu().item(), batch_i)
            writer.add_scalar('train/incumbent_obj_batch', self.env_training.incumbent_objs.mean().cpu().item(), batch_i)
            if reward_log:
                writer.add_scalar('train/reward_batch_mean', float(np.mean(reward_log)), batch_i)

            if batch_i % args.step_validation == 0:
                validation_result1, validation_result2 = self.validation(policy, dev)
                validation_log.append([validation_result1, validation_result2])
                writer.add_scalar('validation/incumbent_obj', validation_result1, batch_i)
                writer.add_scalar('validation/final_obj', validation_result2, batch_i)

                np.save(
                    LOG_DIR / (
                        'training_log_cp_'
                        '{}x{}[{},{}]_{}_{}_{}_{}_'
                        '{}_{}_{}_{}_{}_'
                        '{}_{}_{}_{}_{}_{}.npy'
                    ).format(
                        args.j, args.m, args.l, args.h, args.init_type, _reward_tag(), args.gamma, self._cp_run_tag(),
                        args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type,
                        self.dghan_param_for_saved_model,
                        args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation,
                    ),
                    np.array(log),
                )
                np.save(
                    LOG_DIR / (
                        'validation_log_cp_'
                        '{}x{}[{},{}]_{}_{}_{}_{}_'
                        '{}_{}_{}_{}_{}_'
                        '{}_{}_{}_{}_{}_{}.npy'
                    ).format(
                        args.j, args.m, args.l, args.h, args.init_type, _reward_tag(), args.gamma, self._cp_run_tag(),
                        args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type,
                        self.dghan_param_for_saved_model,
                        args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation,
                    ),
                    np.array(validation_log),
                )
        writer.close()


if __name__ == '__main__':
    affinity = _bind_main_process_away_from_cpu0()
    if affinity is not None:
        print('CPU affinity:', affinity)
    # agent = RL2S4JSSP()
    agent = RL2SCPJSSP()
    agent.train()
