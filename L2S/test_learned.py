import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from L2S.env.environment import BatchGraph, JsspWindow
from L2S.model.actor import Actor


L2S_DIR = ROOT_DIR / 'L2S'
MODEL_DIR = L2S_DIR / 'saved_model'
VALIDATION_DIR = L2S_DIR / 'validation_data'
TEST_DIR = L2S_DIR / 'test_data'


def parse_args():
    parser = argparse.ArgumentParser(description='Test learned policy on JsspWindow.')

    parser.add_argument('--j', type=int, default=20)
    parser.add_argument('--m', type=int, default=15)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--h', type=int, default=99)
    parser.add_argument('--init_type', type=str, default='fdd-divide-mwkr')
    parser.add_argument('--reward_type', type=str, default='yaoxin')
    parser.add_argument('--zero_improvement_penalty', type=float, default=-3.0)
    parser.add_argument('--gamma', type=float, default=1)

    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embedding_layer', type=int, default=4)
    parser.add_argument('--policy_layer', type=int, default=4)
    parser.add_argument('--embedding_type', type=str, default='gin+dghan')
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--drop_out', type=float, default=0.0)

    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--steps_learn', type=int, default=10)
    parser.add_argument('--transit', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--episodes', type=int, default=128000)
    parser.add_argument('--step_validation', type=int, default=10)

    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--cp_solver_time', type=float, default=1)
    parser.add_argument('--cp_solver_cpu', type=int, default=1)
    parser.add_argument('--cp_solver_budget', type=int, default=16)

    parser.add_argument('--model_type', type=str, default='incumbent-cp', choices=['incumbent-cp', 'last-step-cp'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--reference_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--chunk_size', type=int, default=0)
    parser.add_argument('--cap_horizon', type=int, default=None)
    parser.add_argument('--milestones', type=str, default='10,20,50,100')
    parser.add_argument('--result_type', type=str, default='incumbent', choices=['incumbent', 'current'])
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--show', action='store_true')

    return parser.parse_args()


def _bind_main_process_away_from_cpu0():
    if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
        return
    current = set(os.sched_getaffinity(0))
    allowed = {cpu for cpu in current if cpu != 0}
    os.sched_setaffinity(0, allowed if allowed else current)


def _dghan_param_for_saved_model(test_args):
    if test_args.embedding_type == 'gin':
        return 'NAN'
    if test_args.embedding_type in ('dghan', 'gin+dghan'):
        return '{}_{}'.format(test_args.heads, test_args.drop_out)
    raise ValueError('embedding_type should be one of "gin", "dghan", or "gin+dghan".')


def _cp_run_tag(test_args):
    return 'cp{}_{}_wb{}_ws{}'.format(
        test_args.cp_solver_time,
        test_args.cp_solver_cpu,
        test_args.cp_solver_budget,
        test_args.window_size,
    )


def _reward_tag(test_args):
    return '{}_zip{:g}'.format(test_args.reward_type, test_args.zero_improvement_penalty)


def _default_model_path(test_args):
    return MODEL_DIR / (
        '{}_{}x{}[{},{}]_{}_{}_{}_{}_'
        '{}_{}_{}_{}_{}_'
        '{}_{}_{}_{}_{}_{}'
        '.pth'
    ).format(
        test_args.model_type,
        test_args.j, test_args.m, test_args.l, test_args.h,
        test_args.init_type, _reward_tag(test_args), test_args.gamma, _cp_run_tag(test_args),
        test_args.hidden_dim, test_args.embedding_layer, test_args.policy_layer,
        test_args.embedding_type, _dghan_param_for_saved_model(test_args),
        test_args.lr, test_args.steps_learn, test_args.transit,
        test_args.batch_size, test_args.episodes, test_args.step_validation,
    )


def _resolve_data_path(test_args):
    if test_args.data_path is not None:
        return Path(test_args.data_path)

    benchmark_path = TEST_DIR / 'ft{}x{}.npy'.format(test_args.j, test_args.m)
    if benchmark_path.is_file():
        return benchmark_path

    validation_path = VALIDATION_DIR / 'validation_instance_{}x{}[{},{}].npy'.format(
        test_args.j, test_args.m, test_args.l, test_args.h
    )
    if validation_path.is_file():
        return validation_path

    raise FileNotFoundError('No test dataset found. Please provide --data_path explicitly.')


def _resolve_reference_path(test_args, data_path):
    if test_args.reference_path is not None:
        reference_path = Path(test_args.reference_path)
        return reference_path if reference_path.is_file() else None

    benchmark_result = data_path.with_name(data_path.stem + '_result.npy')
    return benchmark_result if benchmark_result.is_file() else None


def _parse_milestones(test_args):
    if not test_args.milestones.strip():
        return []
    milestones = sorted({int(item.strip()) for item in test_args.milestones.split(',') if item.strip()})
    cap_horizon = test_args.cap_horizon if test_args.cap_horizon is not None else test_args.transit
    return [step for step in milestones if step <= cap_horizon]


def _select_device(test_args):
    if test_args.device == 'cpu':
        return 'cpu'
    if test_args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available.')
        return 'cuda'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _build_policy(test_args, device):
    policy = Actor(
        in_dim=3,
        hidden_dim=test_args.hidden_dim,
        embedding_l=test_args.embedding_layer,
        policy_l=test_args.policy_layer,
        embedding_type=test_args.embedding_type,
        heads=test_args.heads,
        dropout=test_args.drop_out,
    ).to(device)
    return policy


def _chunk_slices(size, chunk_size):
    for start in range(0, size, chunk_size):
        yield slice(start, min(start + chunk_size, size))


def _metric_from_env(env, result_type):
    if result_type == 'incumbent':
        return env.incumbent_objs.detach().cpu().numpy().reshape(-1)
    return env.current_objs.detach().cpu().numpy().reshape(-1)


def _print_step_result(step, result, elapsed, reference=None):
    message = 'Step {:4d} | mean {:.2f} | time {:.6f}s/inst'.format(
        step,
        float(result.mean()),
        elapsed,
    )
    if reference is not None:
        gap = ((result - reference) / reference).mean()
        message += ' | gap {:.6f}'.format(float(gap))
    print(message)


def rollout_policy(policy, env, instances, device, init_type, cap_horizon, milestones, result_type, show=False):
    batch_data = BatchGraph()
    # Reset: init_type as spt
    # init_type = 'spt'
    states, feasible_actions, _ = env.reset(instances=instances, init_type=init_type, device=device, plot=show)
    initial_result = _metric_from_env(env, result_type)

    step_results = {}
    start_time = time.time()
    while env.itr < cap_horizon:
        batch_data.wrapper(*states)
        with torch.no_grad():
            actions, _ = policy(batch_data, feasible_actions)
        states, _, feasible_actions, _ = env.step(actions, device, plot=show)
        if env.itr in milestones:
            step_results[env.itr] = (
                _metric_from_env(env, result_type),
                (time.time() - start_time) / instances.shape[0],
            )

    final_result = env.current_objs.detach().cpu().numpy().reshape(-1)
    incumbent_result = env.incumbent_objs.detach().cpu().numpy().reshape(-1)
    batch_data.clean()
    return {
        'initial': initial_result,
        'final': final_result,
        'incumbent': incumbent_result,
        'milestones': step_results,
    }


def main():
    test_args = parse_args()
    _bind_main_process_away_from_cpu0()

    random.seed(test_args.seed)
    np.random.seed(test_args.seed)
    torch.manual_seed(test_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(test_args.seed)

    device = _select_device(test_args)
    model_path = Path(test_args.model_path) if test_args.model_path is not None else _default_model_path(test_args)
    data_path = _resolve_data_path(test_args)
    reference_path = _resolve_reference_path(test_args, data_path)
    milestones = _parse_milestones(test_args)
    cap_horizon = test_args.cap_horizon if test_args.cap_horizon is not None else test_args.transit

    if not model_path.is_file():
        raise FileNotFoundError('Model file not found: {}'.format(model_path))

    instances = np.load(data_path)
    if instances.ndim != 4 or instances.shape[1] != 2:
        raise ValueError('Dataset should have shape [batch, 2, n_job, n_mch], got {}'.format(instances.shape))
    if instances.shape[2] != test_args.j or instances.shape[3] != test_args.m:
        raise ValueError(
            'Dataset size {}x{} does not match args {}x{}.'.format(
                instances.shape[2], instances.shape[3], test_args.j, test_args.m
            )
        )

    reference = np.load(reference_path).reshape(-1) if reference_path is not None else None
    if reference is not None and reference.shape[0] != instances.shape[0]:
        raise ValueError('Reference result count does not match dataset size.')

    env = JsspWindow(
        n_job=test_args.j,
        n_mch=test_args.m,
        low=test_args.l,
        high=test_args.h,
        cp_solver_time=test_args.cp_solver_time,
        cp_solver_cpu=test_args.cp_solver_cpu,
        cpu_budget=test_args.cp_solver_budget,
        window_size=test_args.window_size,
        log_path=None,
        reward_type=test_args.reward_type,
        zero_improvement_penalty=test_args.zero_improvement_penalty,
    )
    policy = _build_policy(test_args, device)
    policy.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    policy.eval()

    if test_args.chunk_size > 0:
        chunk_size = min(test_args.chunk_size, instances.shape[0])
    else:
        chunk_size = instances.shape[0]

    print('Device:', device)
    print('Model :', model_path)
    print('Data  :', data_path)
    print('Batch :', instances.shape[0], 'instances')
    print('Init  :', test_args.init_type)
    # print('Init  :', 'spt')
    print('Horizon:', cap_horizon, '| Result:', test_args.result_type)

    initial_chunks = []
    final_chunks = []
    incumbent_chunks = []
    milestone_results = {step: [] for step in milestones}
    milestone_times = {step: [] for step in milestones}

    for chunk_slice in _chunk_slices(instances.shape[0], chunk_size):
        instance_chunk = instances[chunk_slice]
        rollout = rollout_policy(
            policy=policy,
            env=env,
            instances=instance_chunk,
            device=device,
            init_type=test_args.init_type,
            cap_horizon=cap_horizon,
            milestones=milestones,
            result_type=test_args.result_type,
            show=test_args.show,
        )
        initial_chunks.append(rollout['initial'])
        final_chunks.append(rollout['final'])
        incumbent_chunks.append(rollout['incumbent'])
        for step in milestones:
            if step in rollout['milestones']:
                step_result, step_time = rollout['milestones'][step]
                milestone_results[step].append(step_result)
                milestone_times[step].append(step_time)

    initial_result = np.concatenate(initial_chunks, axis=0)
    final_result = np.concatenate(final_chunks, axis=0)
    incumbent_result = np.concatenate(incumbent_chunks, axis=0)

    print('Initial mean makespan   : {:.2f}'.format(float(initial_result.mean())))
    print('Final mean makespan     : {:.2f}'.format(float(final_result.mean())))
    print('Incumbent mean makespan : {:.2f}'.format(float(incumbent_result.mean())))
    if reference is not None:
        print('Final gap               : {:.6f}'.format(float(((final_result - reference) / reference).mean())))
        print('Incumbent gap           : {:.6f}'.format(float(((incumbent_result - reference) / reference).mean())))

    for step in milestones:
        if not milestone_results[step]:
            continue
        step_result = np.concatenate(milestone_results[step], axis=0)
        step_time = float(np.mean(milestone_times[step]))
        _print_step_result(step, step_result, step_time, reference=reference)


if __name__ == '__main__':
    main()
