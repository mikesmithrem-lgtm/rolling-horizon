import numpy as np
import os
import torch
import time
import random

from L2S.env.environment import JsspN5, BatchGraph
from L2S.model.actor import Actor
from L2S.inout import load_data

if __name__ == '__main__':
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)

    show = False
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using cuda to test...')

    # benchmark config
    p_l = 1
    p_h = 99
    init_type = ['fdd-divide-mwkr']  # ['fdd-divide-mwkr', 'spt']

    # model config
    model_l = 1
    model_h = 99
    model_init_type = 'fdd-divide-mwkr'
    reward_type = 'yaoxin'  # 'yaoxin', 'consecutive'
    zero_improvement_penalty = -3.0
    gamma = 1

    hidden_dim = 128
    embedding_layer = 4
    policy_layer = 4
    embedding_type = 'gin+dghan'  # 'gin', 'dghan', 'gin+dghan'
    heads = 1
    drop_out = 0.

    lr = 5e-5  # 5e-5, 4e-5
    steps_learn = 10
    training_episode_length = 500
    batch_size = 64
    episodes = 128000  # 128000, 256000
    step_validation = 10

    model_type = 'incumbent'  # 'incumbent', 'last-step'
    dghan_param_for_saved_model = '{}_{}'.format(heads, drop_out)

    # MDP config
    cap_horizon = 500
    performance_milestones = [500, 1000, 2000, 5000]  # [500, 1000, 2000, 5000]
    result_type = 'incumbent'  # 'current', 'incumbent'
    fea_norm_const = 1000

    # Problem config
    model_j, model_m = 20, 15
    p_j, p_m = 100, 20
    # read saved gap_against or use ortools to solve it.

    reward_tag = '{}_zip{:g}'.format(reward_type, zero_improvement_penalty)
    env = JsspN5(
        n_job=p_j,
        n_mch=p_m,
        low=p_l,
        high=p_h,
        reward_type='yaoxin',
        zero_improvement_penalty=zero_improvement_penalty,
        fea_norm_const=fea_norm_const,
    )
    torch.manual_seed(seed)
    policy = Actor(in_dim=3,
                   hidden_dim=hidden_dim,
                   embedding_l=embedding_layer,
                   policy_l=policy_layer,
                   embedding_type=embedding_type,
                   heads=heads,
                   dropout=drop_out).to(dev)
    saved_model_path = './L2S/saved_model/' \
                       '{}_{}x{}[{},{}]_{}_{}_{}_' \
                       '{}_{}_{}_{}_{}_' \
                       '{}_{}_{}_{}_{}_{}' \
                       '.pth' \
        .format(model_type, model_j, model_m, model_l, model_h, model_init_type, reward_tag, gamma,
                hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model,
                lr, steps_learn, training_episode_length, batch_size, episodes, step_validation)
    print('loading model from:', saved_model_path)
    policy.load_state_dict(torch.load(saved_model_path, map_location=torch.device(dev)))

    path = f"./benchmark/TA{p_j}x{p_m}"
    inst = []
    gap_against = []
    for file in os.listdir(path):
        if file.startswith('.') or file.startswith('cached'):
            continue
        # Solve the instance
        instance = load_data(os.path.join(path, file), device="cpu")
        costs, machines = np.array(instance['costs']), np.array(instance['machines'])
        gt_ms = instance['makespan']
        machines = machines + 1
        machines = np.array(machines, dtype=np.int64)
        shap = np.stack([costs, machines], axis=0)
        inst.append(shap)
        gap_against.append(gt_ms)

    inst = np.array(inst, dtype=np.int64)
    gap_against = np.array(gap_against)

    print('Start to test initial solution: {}...'.format(init_type[0]))
    print('Starting rollout DRL policy...')
    if p_j >= 100 and inst.shape[0] >= 20:
        chunk_size = 20
        print('Problem of size {}x{} containing {} instances is too large to form a batch. '
              'Splitting into chunks and test seperately. '
              'Chunk size is {}.'.format(p_j, p_m, inst.shape[0], chunk_size))
    else:
        chunk_size = inst.shape[0]
    n_chunks = inst.shape[0] // chunk_size
    results_each_init, inference_time_each_init = [], []
    for i in range(n_chunks):
        # t3 = time.time()
        chunk_result, chunk_time = [], []
        inst_chunk = inst[i * chunk_size:(i + 1) * chunk_size]
        batch_data = BatchGraph()
        states, feasible_actions, _ = env.reset(instances=inst_chunk, init_type=init_type[0], device=dev, plot=show)
        # t4 = time.time()
        drl_start = time.time()
        while env.itr < cap_horizon:
            # t1 = time.time()
            batch_data.wrapper(*states)
            actions, _ = policy(batch_data, feasible_actions)
            states, _, feasible_actions, _ = env.step(actions, dev, plot=show)
            # t2 = time.time()
            for log_horizon in performance_milestones:
                if env.itr == log_horizon:
                    if result_type == 'incumbent':
                        DRL_result = env.incumbent_objs.cpu().squeeze().numpy()
                    else:
                        DRL_result = env.current_objs.cpu().squeeze().numpy()
                    chunk_result.append(DRL_result)
                    chunk_time.append(time.time() - drl_start)
                    if n_chunks == 1:
                        print('For testing steps: {}    '.format(env.itr),
                              'DRL Gap: {:.6f}    '.format(((DRL_result - gap_against) / gap_against).mean()),
                              'Result Mean: {}    '.format((DRL_result.mean())),
                              'DRL results takes: {:.6f} per instance.'.format(
                                  (time.time() - drl_start) / chunk_size))
        results_each_init.append(np.stack(chunk_result))
        inference_time_each_init.append(np.array(chunk_time))
    results_each_init = np.concatenate(results_each_init, axis=-1)
    inference_time_each_init = ((np.stack(inference_time_each_init).sum(axis=0)) / n_chunks) / chunk_size
    if n_chunks > 1:
        for i, step in enumerate(performance_milestones):
            print('For testing steps: {}    '.format(step),
                  'DRL Gap: {:.6f}    '.format(((results_each_init[i] - gap_against) / gap_against).mean()),
                  'DRL results takes: {:.6f} per instance.'.format(inference_time_each_init[i]))



