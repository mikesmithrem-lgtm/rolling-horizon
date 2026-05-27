"""
Same as evaluate_vs_random.py but uses argmax over candidate scores
instead of sampling, to isolate whether the policy has actually learned
useful discrimination.
"""
import argparse, os, sys, time, random
import numpy as np
import torch
from types import SimpleNamespace
from torch_geometric.utils import add_self_loops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from L2S.env.environment import JsspWindow
from L2S.model.actor import Actor


def _bind():
    if not hasattr(os, 'sched_setaffinity'):
        return
    cur = set(os.sched_getaffinity(0))
    target = {c for c in cur if c != 0}
    os.sched_setaffinity(0, target or cur)


def _pack(states):
    x, e_pc, e_mc, batch = states
    return SimpleNamespace(x=x, edge_index_pc=e_pc, edge_index_mc=e_mc, batch=batch)


def actor_argmax(actor, batch_states, batch_window_states, feasible_actions):
    """Replicate Actor.forward but pick argmax over candidate scores per batch."""
    bs = batch_states
    node_embed_gin, graph_embed_gin = actor.global_embedding_gin(
        bs.x, add_self_loops(torch.cat([bs.edge_index_pc, bs.edge_index_mc], dim=-1))[0], bs.batch)
    node_embed_dghan, graph_embed_dghan = actor.global_embedding_dghan(
        bs.x, add_self_loops(bs.edge_index_pc)[0],
        add_self_loops(bs.edge_index_mc)[0], len(feasible_actions))
    node_embed = torch.cat([node_embed_gin, node_embed_dghan], dim=-1)
    graph_embed = torch.cat([graph_embed_gin, graph_embed_dghan], dim=-1)
    B = graph_embed.shape[0]
    N = node_embed.shape[0] // B
    g_exp = graph_embed.repeat_interleave(N, dim=0)
    g_op = torch.cat([node_embed, g_exp], dim=-1).reshape(B, N, -1)

    owner = []
    action_per_w = []
    cands = [[] for _ in range(B)]
    op_feat, mch_feat, op_glob, op_mid, op_wid, mch_wid, ep, em = [], [], [], [], [], [], [], []
    anc = []
    op_off = mch_off = 0
    device = bs.x.device

    for b, ws_list in enumerate(batch_window_states):
        for ws in ws_list:
            K = ws['op_features'].size(0); M = ws['mch_features'].size(0)
            w_id = len(owner)
            op_feat.append(ws['op_features']); mch_feat.append(ws['mch_features'])
            op_glob.append(g_op[b].index_select(0, ws['op_ids']))
            op_mid.append(ws['op_machine_id'] + mch_off)
            op_wid.append(torch.full((K,), w_id, dtype=torch.long, device=device))
            mch_wid.append(torch.full((M,), w_id, dtype=torch.long, device=device))
            anc.append(op_off + int(ws['anchor_local_idx']))
            ep.append(ws['edge_index_pc'] + op_off)
            em.append(ws['edge_index_mc'] + op_off)
            owner.append(b); action_per_w.append(int(ws['action'])); cands[b].append(w_id)
            op_off += K; mch_off += M

    sampled = [0] * B
    if not owner:
        return sampled

    h = actor.window_encoder.forward_batched(
        op_global=torch.cat(op_glob, dim=0),
        op_features=torch.cat(op_feat, dim=0),
        mch_features=torch.cat(mch_feat, dim=0),
        op_machine_id=torch.cat(op_mid, dim=0),
        op_window_id=torch.cat(op_wid, dim=0),
        mch_window_id=torch.cat(mch_wid, dim=0),
        anchor_global_idx=torch.tensor(anc, dtype=torch.long, device=device),
        edge_index_pc=torch.cat(ep, dim=1) if ep else torch.zeros((2,0), dtype=torch.long, device=device),
        edge_index_mc=torch.cat(em, dim=1) if em else torch.zeros((2,0), dtype=torch.long, device=device),
        num_windows=len(owner),
    )
    for layer in actor.policy:
        h = layer(h)
    scores = actor.action_head(h).squeeze(-1)

    for b in range(B):
        if not cands[b]:
            continue
        idx_tensor = torch.tensor(cands[b], dtype=torch.long, device=device)
        s = scores[idx_tensor]
        best = int(torch.argmax(s).item())
        sampled[b] = action_per_w[cands[b][best]]
    return sampled


def rollout(env, instances, transit, device, init_type, mode, actor=None, rng=None):
    states, bws, fas, _ = env.reset(instances, init_type=init_type, device=device)
    while env.itr < transit:
        if mode == 'argmax':
            with torch.no_grad():
                actions = actor_argmax(actor, _pack(states), bws, fas)
        elif mode == 'random':
            actions = [int(c[rng.randrange(len(c))]) if len(c) > 0 else 0 for c in fas]
        states, bws, _, fas, _ = env.step(actions, device)
    return env.current_objs.squeeze(-1).cpu().numpy(), env.best_objs.squeeze(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--num_instances', type=int, default=100)
    ap.add_argument('--transit', type=int, default=500)
    ap.add_argument('--cp_solver_time', type=float, default=1.0)
    ap.add_argument('--cpu_budget', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--mode', choices=['argmax', 'random', 'both'], default='both')
    args = ap.parse_args()

    _bind()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    val = np.load('./L2S/validation_data/validation_instance_20x15[1,99].npy')[:args.num_instances]
    print(f'instances={len(val)}, transit={args.transit}')

    actor = Actor(in_dim=3, hidden_dim=128, window_op_in_dim=10, window_mch_in_dim=8,
                  embedding_l=4, policy_l=4, heads=1, dropout=0.1).to(device)
    actor.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    actor.eval()

    env = JsspWindow(n_job=20, n_mch=15, low=1, high=99,
                     cp_solver_time=args.cp_solver_time, cp_solver_cpu=1,
                     cpu_budget=args.cpu_budget, window_size=150)

    results = {}
    if args.mode in ('argmax', 'both'):
        t0 = time.time()
        f, b = rollout(env, val, args.transit, device, 'spt-pdr', 'argmax', actor=actor)
        print(f'argmax: {time.time()-t0:.1f}s  final mean={f.mean():.2f} med={np.median(f):.2f}  best mean={b.mean():.2f} med={np.median(b):.2f}')
        results['argmax'] = b
    if args.mode in ('random', 'both'):
        rng = random.Random(args.seed)
        t0 = time.time()
        f, b = rollout(env, val, args.transit, device, 'spt-pdr', 'random', rng=rng)
        print(f'random: {time.time()-t0:.1f}s  final mean={f.mean():.2f} med={np.median(f):.2f}  best mean={b.mean():.2f} med={np.median(b):.2f}')
        results['random'] = b

    if 'argmax' in results and 'random' in results:
        p, r = results['argmax'], results['random']
        wins = int((p < r).sum()); ties = int((p == r).sum()); losses = int((p > r).sum())
        print(f'\nper-instance best: argmax wins {wins} / ties {ties} / losses {losses}')
        print(f'mean delta (argmax - random): {(p - r).mean():+.2f}  median: {np.median(p - r):+.2f}')


if __name__ == '__main__':
    main()
