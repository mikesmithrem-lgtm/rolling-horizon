"""Quick random-LNS baseline on validation subset for a reference number."""
import os, sys, time, random
sys.path.insert(0, '.')
import numpy as np
import torch
from L2S.env.environment import JsspWindow

def main():
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    transit = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    cp_budget = int(sys.argv[3]) if len(sys.argv) > 3 else 8   # leave cpus for training

    if not hasattr(os, 'sched_setaffinity'):
        pass
    else:
        cur = set(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {c for c in cur if c != 0})

    val = np.load('./L2S/validation_data/validation_instance_20x15[1,99].npy')[:n_inst]
    print(f'instances={len(val)}, transit={transit}, cp_budget={cp_budget}')

    env = JsspWindow(n_job=20, n_mch=15, low=1, high=99,
                     cp_solver_time=1, cp_solver_cpu=1,
                     cpu_budget=cp_budget, window_size=150)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = random.Random(1)
    states, bws, fa, _ = env.reset(val, init_type='spt-pdr', device=dev)
    init = env.initial_objs.squeeze(-1).cpu().numpy()
    print(f'init makespan: mean={init.mean():.2f}, median={np.median(init):.2f}')

    t0 = time.time()
    while env.itr < transit:
        actions = [int(c[rng.randrange(len(c))]) if len(c)>0 else 0 for c in fa]
        states, bws, _, fa, _ = env.step(actions, dev)
        if env.itr % 50 == 0:
            cur = env.current_objs.squeeze(-1).cpu().numpy()
            best = env.best_objs.squeeze(-1).cpu().numpy()
            print(f'  step {env.itr:3d}  cur_mean={cur.mean():.2f}  best_mean={best.mean():.2f}  t={time.time()-t0:.1f}s')
    final = env.current_objs.squeeze(-1).cpu().numpy()
    best = env.best_objs.squeeze(-1).cpu().numpy()
    print(f'RANDOM-LNS done in {time.time()-t0:.1f}s')
    print(f'  final mean={final.mean():.2f}  median={np.median(final):.2f}')
    print(f'  best  mean={best.mean():.2f}  median={np.median(best):.2f}')

if __name__ == '__main__':
    main()
