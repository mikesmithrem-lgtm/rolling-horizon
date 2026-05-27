"""
Evaluate a trained Actor against uniform-random action selection on the
20x15 validation set, using identical JsspWindow LNS for both so the
comparison isolates *which feasible action to pick* (i.e. policy quality).

Run:
    python evaluate_vs_random.py --checkpoint <path-to-.pth>
"""
import argparse
import os
import sys
import time
import random
import numpy as np
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from L2S.env.environment import JsspWindow
from L2S.model.actor import Actor


def _bind_main_process_away_from_cpu0():
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    current = set(os.sched_getaffinity(0))
    allowed = {cpu for cpu in current if cpu != 0}
    target = allowed if allowed else current
    os.sched_setaffinity(0, target)
    return set(os.sched_getaffinity(0))


def _pack(states):
    x, e_pc, e_mc, batch = states
    return SimpleNamespace(x=x, edge_index_pc=e_pc, edge_index_mc=e_mc, batch=batch)


def rollout_policy(env, instances, policy, transit, device, init_type):
    """Roll trained policy for `transit` LNS steps; return final + best-so-far makespans."""
    states, bws, fas, _ = env.reset(instances, init_type=init_type, device=device)
    policy.eval()
    with torch.no_grad():
        while env.itr < transit:
            actions, _ = policy(_pack(states), bws, fas)
            states, bws, _, fas, _ = env.step(actions, device)
    final = env.current_objs.squeeze(-1).cpu().numpy()
    best = env.best_objs.squeeze(-1).cpu().numpy()
    return final, best


def rollout_random(env, instances, transit, device, init_type, rng):
    """Uniform-random choice over feasible actions (same env)."""
    states, bws, fas, _ = env.reset(instances, init_type=init_type, device=device)
    while env.itr < transit:
        actions = []
        for cand in fas:
            if len(cand) == 0:
                actions.append(0)
            else:
                actions.append(int(cand[rng.randrange(len(cand))]))
        states, bws, _, fas, _ = env.step(actions, device)
    final = env.current_objs.squeeze(-1).cpu().numpy()
    best = env.best_objs.squeeze(-1).cpu().numpy()
    return final, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--j", type=int, default=20)
    parser.add_argument("--m", type=int, default=15)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--h", type=int, default=99)
    parser.add_argument("--init_type", type=str, default="spt-pdr")
    parser.add_argument("--transit", type=int, default=500)
    parser.add_argument("--window_size", type=int, default=150)
    parser.add_argument("--cp_solver_time", type=float, default=1.0)
    parser.add_argument("--cp_solver_cpu", type=int, default=1)
    parser.add_argument("--cpu_budget", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embedding_layer", type=int, default=4)
    parser.add_argument("--policy_layer", type=int, default=4)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--drop_out", type=float, default=0.1)
    parser.add_argument("--num_instances", type=int, default=100,
                        help="Use first N validation instances; -1 = all")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    _bind_main_process_away_from_cpu0()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    val_path = f"./L2S/validation_data/validation_instance_{args.j}x{args.m}[{args.l},{args.h}].npy"
    val = np.load(val_path)
    if args.num_instances > 0:
        val = val[: args.num_instances]
    print(f"Loaded {len(val)} validation instances from {val_path}")

    env_pol = JsspWindow(n_job=args.j, n_mch=args.m, low=args.l, high=args.h,
                         cp_solver_time=args.cp_solver_time, cp_solver_cpu=args.cp_solver_cpu,
                         cpu_budget=args.cpu_budget, window_size=args.window_size)
    env_rnd = JsspWindow(n_job=args.j, n_mch=args.m, low=args.l, high=args.h,
                         cp_solver_time=args.cp_solver_time, cp_solver_cpu=args.cp_solver_cpu,
                         cpu_budget=args.cpu_budget, window_size=args.window_size)

    policy = Actor(in_dim=3, hidden_dim=args.hidden_dim,
                   window_op_in_dim=10, window_mch_in_dim=8,
                   embedding_l=args.embedding_layer, policy_l=args.policy_layer,
                   heads=args.heads, dropout=args.drop_out).to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    # The env operates on batches; for 100 validation instances we just feed them all
    # since the env handles batch parallelism for CP solving internally.
    print(f"\nRunning trained policy for {args.transit} LNS steps...")
    t0 = time.time()
    pol_final, pol_best = rollout_policy(env_pol, val, policy, args.transit, device, args.init_type)
    print(f"  policy rollout: {time.time()-t0:.1f}s")

    print(f"\nRunning random baseline for {args.transit} LNS steps...")
    rng = random.Random(args.seed)
    t0 = time.time()
    rnd_final, rnd_best = rollout_random(env_rnd, val, args.transit, device, args.init_type, rng)
    print(f"  random rollout: {time.time()-t0:.1f}s")

    print("\n" + "=" * 60)
    print(f"Evaluation on {len(val)} instances, {args.transit} LNS steps each")
    print("=" * 60)
    print(f"{'metric':<24}{'policy':>12}{'random':>12}{'delta':>12}")
    for name, p, r in [
        ("final makespan (mean)", pol_final.mean(), rnd_final.mean()),
        ("best-so-far (mean)",    pol_best.mean(),  rnd_best.mean()),
        ("final makespan (med)",  np.median(pol_final), np.median(rnd_final)),
        ("best-so-far (med)",     np.median(pol_best),  np.median(rnd_best)),
    ]:
        print(f"{name:<24}{p:>12.2f}{r:>12.2f}{p - r:>+12.2f}")

    wins = int((pol_best < rnd_best).sum())
    ties = int((pol_best == rnd_best).sum())
    losses = int((pol_best > rnd_best).sum())
    print(f"\nper-instance best-so-far: policy wins {wins} / ties {ties} / losses {losses}")

    rel = (rnd_best - pol_best) / rnd_best * 100
    print(f"mean relative improvement (vs random best-so-far): {rel.mean():+.2f}%  median {np.median(rel):+.2f}%")


if __name__ == "__main__":
    main()
