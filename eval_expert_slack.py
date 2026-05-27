"""
Run the 'max-window-slack' expert on the validation set (same JsspWindow env)
and compare to random. If the expert clearly beats random, it is worth cloning.

For each step the expert picks the candidate anchor whose 150-op window has the
most total machine slack -- i.e. the most reschedulable time within the window
boundaries -- since that is where CP has the most room to compress.
"""
import os, sys, time, random, argparse
import numpy as np
import torch
sys.path.insert(0, '.')

from L2S.env.environment import JsspWindow
from L2S.env.window_utils import (
    _window_plan_nodes,
    _window_machine_availability,
)


def _bind():
    if not hasattr(os, 'sched_setaffinity'):
        return
    cur = set(os.sched_getaffinity(0))
    target = {c for c in cur if c != 0}
    os.sched_setaffinity(0, target or cur)


def expert_slack_action(env, b, action_candidates, sorted_nodes, index_by_node):
    """Pick the candidate whose window has the largest sum of machine slack."""
    instance = env.instances[b]
    start_times = env.current_start_times[b]
    duration = instance[0]
    mch = instance[1]            # 1-based
    n_job, n_mch = env.n_job, env.n_mch

    best_action = action_candidates[0]
    best_slack = -1.0
    for a in action_candidates:
        nodes = _window_plan_nodes(start_times, n_job, n_mch, env.window_size, int(a),
                                   sorted_nodes=sorted_nodes, index_by_node=index_by_node)
        avail = _window_machine_availability(nodes, start_times, duration, mch, n_job, n_mch)
        if not avail:
            continue
        span = sum(latest - earliest for earliest, latest in avail.values())
        op_time = sum(int(duration[(node - 1) // n_mch, (node - 1) % n_mch]) for node in nodes)
        slack = span - op_time
        if slack > best_slack:
            best_slack = slack
            best_action = int(a)
    return int(best_action)


def per_batch_sorted_nodes(env, b):
    n_job, n_mch = env.n_job, env.n_mch
    start_times = env.current_start_times[b]
    nodes = list(range(1, n_job * n_mch + 1))
    nodes.sort(key=lambda node: (
        int(start_times[(node - 1) // n_mch, (node - 1) % n_mch]),
        int(node),
    ))
    idx_by = {n: i for i, n in enumerate(nodes)}
    return nodes, idx_by


def rollout_expert(env, instances, transit, device, init_type):
    states, bws, fas, _ = env.reset(instances, init_type=init_type, device=device)
    while env.itr < transit:
        actions = []
        for b, cand in enumerate(fas):
            if len(cand) == 0:
                actions.append(0); continue
            sn, ix = per_batch_sorted_nodes(env, b)
            actions.append(expert_slack_action(env, b, cand, sn, ix))
        states, bws, _, fas, _ = env.step(actions, device)
    return env.current_objs.squeeze(-1).cpu().numpy(), env.best_objs.squeeze(-1).cpu().numpy()


def rollout_random(env, instances, transit, device, init_type, rng):
    states, bws, fas, _ = env.reset(instances, init_type=init_type, device=device)
    while env.itr < transit:
        actions = [int(c[rng.randrange(len(c))]) if len(c) > 0 else 0 for c in fas]
        states, bws, _, fas, _ = env.step(actions, device)
    return env.current_objs.squeeze(-1).cpu().numpy(), env.best_objs.squeeze(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_instances', type=int, default=100)
    ap.add_argument('--transit', type=int, default=500)
    ap.add_argument('--cpu_budget', type=int, default=16)
    ap.add_argument('--cp_solver_time', type=float, default=1.0)
    args = ap.parse_args()

    _bind()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val = np.load('./L2S/validation_data/validation_instance_20x15[1,99].npy')[:args.num_instances]
    print(f'instances={len(val)}, transit={args.transit}')

    env = JsspWindow(n_job=20, n_mch=15, low=1, high=99,
                     cp_solver_time=args.cp_solver_time, cp_solver_cpu=1,
                     cpu_budget=args.cpu_budget, window_size=150)

    t0 = time.time()
    fe, be = rollout_expert(env, val, args.transit, device, 'spt-pdr')
    print(f'EXPERT(slack): {time.time()-t0:.1f}s  final mean={fe.mean():.2f} med={np.median(fe):.2f}  best mean={be.mean():.2f} med={np.median(be):.2f}')

    env2 = JsspWindow(n_job=20, n_mch=15, low=1, high=99,
                      cp_solver_time=args.cp_solver_time, cp_solver_cpu=1,
                      cpu_budget=args.cpu_budget, window_size=150)
    rng = random.Random(1)
    t0 = time.time()
    fr, br = rollout_random(env2, val, args.transit, device, 'spt-pdr', rng)
    print(f'RANDOM       : {time.time()-t0:.1f}s  final mean={fr.mean():.2f} med={np.median(fr):.2f}  best mean={br.mean():.2f} med={np.median(br):.2f}')

    wins = int((be < br).sum()); ties = int((be == br).sum()); losses = int((be > br).sum())
    print(f'\nexpert vs random per-instance: wins {wins} / ties {ties} / losses {losses}')
    print(f'mean delta (expert - random): {(be - br).mean():+.2f}  median: {np.median(be - br):+.2f}')


if __name__ == '__main__':
    main()
