import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_mean, scatter_max


class DGHANlayer(torch.nn.Module):
    def __init__(self, in_chnl, out_chnl, dropout, concat, heads=2):
        super(DGHANlayer, self).__init__()
        self.dropout = dropout
        self.opsgrp_conv = GATConv(in_chnl, out_chnl, heads=heads, dropout=dropout, concat=concat)
        self.mchgrp_conv = GATConv(in_chnl, out_chnl, heads=heads, dropout=dropout, concat=concat)

    def forward(self, node_h, edge_index_pc, edge_index_mc):
        node_h_pc = F.elu(self.opsgrp_conv(F.dropout(node_h, p=self.dropout, training=self.training), edge_index_pc))
        node_h_mc = F.elu(self.mchgrp_conv(F.dropout(node_h, p=self.dropout, training=self.training), edge_index_mc))
        node_h = torch.mean(torch.stack([node_h_pc, node_h_mc]), dim=0, keepdim=False)
        return node_h


class DGHAN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, layer_dghan=4, heads=2):
        super(DGHAN, self).__init__()
        self.layer_dghan = layer_dghan
        self.hidden_dim = hidden_dim

        self.DGHAN_layers = torch.nn.ModuleList()

        if layer_dghan == 1:
            self.DGHAN_layers.append(DGHANlayer(in_dim, hidden_dim, dropout, concat=False, heads=heads))
        else:
            self.DGHAN_layers.append(DGHANlayer(in_dim, hidden_dim, dropout, concat=True, heads=heads))
            for _ in range(layer_dghan - 2):
                self.DGHAN_layers.append(DGHANlayer(heads * hidden_dim, hidden_dim, dropout, concat=True, heads=heads))
            self.DGHAN_layers.append(DGHANlayer(heads * hidden_dim, hidden_dim, dropout, concat=False, heads=1))

    def forward(self, x, edge_index_pc, edge_index_mc, batch_size):
        h_node = self.DGHAN_layers[0](x,edge_index_pc,edge_index_mc)
        for layer in range(1, self.layer_dghan):
            h_node = self.DGHAN_layers[layer](h_node,edge_index_pc,edge_index_mc)

        return h_node, torch.mean(h_node.reshape(batch_size, -1, self.hidden_dim), dim=1)


class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_gin=4):
        super(GIN, self).__init__()
        self.layer_gin = layer_gin

        self.GIN_layers = torch.nn.ModuleList()

        self.GIN_layers.append(
            GINConv(
                Sequential(
                    Linear(in_dim, hidden_dim),
                    torch.nn.BatchNorm1d(hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim)
                ),
                eps=0,
                train_eps=False,
                aggr='mean',
                flow="source_to_target"
            )
        )

        for _ in range(layer_gin - 1):
            self.GIN_layers.append(
                GINConv(
                    Sequential(
                        Linear(hidden_dim, hidden_dim),
                        torch.nn.BatchNorm1d(hidden_dim),
                        ReLU(),
                        Linear(hidden_dim, hidden_dim)
                    ),
                    eps=0,
                    train_eps=False,
                    aggr='mean',
                    flow="source_to_target"
                )
            )

    def forward(self, x, edge_index, batch):
        hidden_rep = []
        node_pool_over_layer = 0

        h = self.GIN_layers[0](x,edge_index)
        node_pool_over_layer += h
        hidden_rep.append(h)

        for layer in range(1, self.layer_gin):
            h = self.GIN_layers[layer](h,edge_index)
            node_pool_over_layer += h
            hidden_rep.append(h)

        gPool_over_layer = 0
        for layer_h in hidden_rep:
            g_pool = global_mean_pool(layer_h, batch)
            gPool_over_layer += g_pool

        return node_pool_over_layer, gPool_over_layer


class LocalEdgeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.pc_conv = GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim)
            ),
            aggr='mean'
        )
        self.mc_conv = GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim)
            ),
            aggr='mean'
        )
        self.fuse = Sequential(
            Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            Linear(hidden_dim, hidden_dim)
        )

    def forward(self, op_h, edge_index_pc, edge_index_mc):
        if edge_index_pc is not None and edge_index_pc.size(1) > 0:
            h_pc = self.pc_conv(op_h, edge_index_pc)
        else:
            h_pc = torch.zeros_like(op_h)

        if edge_index_mc is not None and edge_index_mc.size(1) > 0:
            h_mc = self.mc_conv(op_h, edge_index_mc)
        else:
            h_mc = torch.zeros_like(op_h)

        op_h = self.fuse(torch.cat([op_h, h_pc, h_mc], dim=-1))
        return op_h
    

class WindowEncoder(nn.Module):
    def __init__(self, 
                 global_op_dim, 
                 window_op_in_dim, 
                 window_mch_in_dim, 
                 hidden_dim, 
                 local_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.op_embedding = Sequential(
            Linear(window_op_in_dim, hidden_dim),
            nn.Tanh(),
            Linear(hidden_dim, hidden_dim)
        )

        self.mch_embedding = Sequential(
            Linear(window_mch_in_dim, hidden_dim),
            nn.Tanh(),
            Linear(hidden_dim, hidden_dim)
        )

        self.op_init = Sequential(
            Linear(global_op_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            Linear(hidden_dim, hidden_dim)
        )

        self.mch_update = Sequential(
            Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            Linear(hidden_dim, hidden_dim)
        )

        self.op_mch_fuse = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            Linear(hidden_dim, hidden_dim)
        )

        self.local_encoders = nn.ModuleList([
            LocalEdgeEncoder(hidden_dim) for _ in range(local_layers)
        ])

        self.readout = Sequential(
            Linear(hidden_dim * 5, hidden_dim),
            nn.Tanh(),
            Linear(hidden_dim, hidden_dim)
        )

    def _pool_mean(self, x, group_idx, num_groups):
        out = torch.zeros(num_groups, x.size(-1), device=x.device, dtype=x.dtype)
        cnt = torch.zeros(num_groups, 1, device=x.device, dtype=x.dtype)

        out = out.index_add(0, group_idx, x)
        ones = torch.ones(group_idx.size(0), 1, device=x.device, dtype=x.dtype)
        cnt = cnt.index_add(0, group_idx, ones)

        return out / cnt.clamp(min=1.0)

    def _pool_max(self, x, group_idx, num_groups):
        outs = []
        for g in range(num_groups):
            mask = (group_idx == g)
            if mask.any():
                outs.append(x[mask].max(dim=0).values)
            else:
                outs.append(torch.zeros(x.size(-1), device=x.device, dtype=x.dtype))
        return torch.stack(outs, dim=0)

    def forward(self, global_op_embed, window_state):
        """
        global_op_embed: [N, Dg]，当前单个实例的全局op表示
        """
        op_ids = window_state["op_ids"]                   # [K]
        op_features = window_state["op_features"]         # [K, Fo]
        mch_features = window_state["mch_features"]       # [M, Fm]
        op_machine_id = window_state["op_machine_id"]     # [K]
        edge_index_pc = window_state["edge_index_pc"]     # [2, E1]
        edge_index_mc = window_state["edge_index_mc"]     # [2, E2]
        anchor_local_idx = window_state["anchor_local_idx"]

        op_global = global_op_embed[op_ids]               # [K, Dg]
        op_local = self.op_embedding(op_features)         # [K, H]
        mch_local = self.mch_embedding(mch_features)      # [M, H]

        op_machine_ctx = mch_local[op_machine_id]         # [K, H]

        op_h = self.op_init(torch.cat([op_global, op_local, op_machine_ctx], dim=-1))

        num_mch = mch_local.size(0)
        mch_mean = self._pool_mean(op_h, op_machine_id, num_mch)
        mch_max = self._pool_max(op_h, op_machine_id, num_mch)

        mch_h = self.mch_update(torch.cat([mch_local, mch_mean, mch_max], dim=-1))

        op_h = self.op_mch_fuse(torch.cat([op_h, mch_h[op_machine_id]], dim=-1))

        for encoder in self.local_encoders:
            op_h = encoder(op_h, edge_index_pc, edge_index_mc)

        anchor_h = op_h[anchor_local_idx]
        op_mean = op_h.mean(dim=0)
        op_max = op_h.max(dim=0).values
        mch_mean = mch_h.mean(dim=0)
        mch_max = mch_h.max(dim=0).values

        action_h = self.readout(torch.cat([anchor_h, op_mean, op_max, mch_mean, mch_max], dim=-1))
        return action_h

    def forward_batched(self,
                        op_global,
                        op_features,
                        mch_features,
                        op_machine_id,
                        op_window_id,
                        mch_window_id,
                        anchor_global_idx,
                        edge_index_pc,
                        edge_index_mc,
                        num_windows):
        """
        Vectorized encoding for all candidate windows of a whole batch at once.

        Shapes (sumK = total ops across all windows in this call,
                sumM = total machine slots, W = num_windows):
            op_global         : [sumK, Dg]
            op_features       : [sumK, Fo]
            mch_features      : [sumM, Fm]
            op_machine_id     : [sumK]      indices into [0, sumM) (already offset)
            op_window_id      : [sumK]      indices into [0, W)
            mch_window_id     : [sumM]      indices into [0, W)
            anchor_global_idx : [W]         indices into [0, sumK)
            edge_index_pc/mc  : [2, E]      local op ids already offset
        """
        op_local = self.op_embedding(op_features)
        mch_local = self.mch_embedding(mch_features)

        op_machine_ctx = mch_local[op_machine_id]
        op_h = self.op_init(torch.cat([op_global, op_local, op_machine_ctx], dim=-1))

        num_mch = mch_local.size(0)
        mch_mean = scatter_mean(op_h, op_machine_id, dim=0, dim_size=num_mch)
        mch_max, _ = scatter_max(op_h, op_machine_id, dim=0, dim_size=num_mch)

        mch_h = self.mch_update(torch.cat([mch_local, mch_mean, mch_max], dim=-1))
        op_h = self.op_mch_fuse(torch.cat([op_h, mch_h[op_machine_id]], dim=-1))

        for encoder in self.local_encoders:
            op_h = encoder(op_h, edge_index_pc, edge_index_mc)

        anchor_h = op_h[anchor_global_idx]                                   # [W, H]
        op_mean_w = scatter_mean(op_h, op_window_id, dim=0, dim_size=num_windows)
        op_max_w, _ = scatter_max(op_h, op_window_id, dim=0, dim_size=num_windows)
        mch_mean_w = scatter_mean(mch_h, mch_window_id, dim=0, dim_size=num_windows)
        mch_max_w, _ = scatter_max(mch_h, mch_window_id, dim=0, dim_size=num_windows)

        return self.readout(torch.cat(
            [anchor_h, op_mean_w, op_max_w, mch_mean_w, mch_max_w], dim=-1
        ))
    

class Actor(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 window_op_in_dim,
                 window_mch_in_dim,
                 embedding_l=4,
                 policy_l=3,
                 heads=4,
                 dropout=0.6):
        super(Actor, self).__init__()
        self.embedding_l = embedding_l
        self.policy_l = policy_l
        self.hidden_dim = hidden_dim

        # global encoder
        self.global_embedding_gin = GIN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            layer_gin=embedding_l
        )
        self.global_embedding_dghan = DGHAN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            layer_dghan=embedding_l,
            heads=heads
        )

        # gin+dghan -> 2H
        # concat graph embedding again -> global op embed = 4H
        self.global_op_dim = hidden_dim * 4

        # local window encoder
        self.window_encoder = WindowEncoder(
            global_op_dim=self.global_op_dim,
            window_op_in_dim=window_op_in_dim,
            window_mch_in_dim=window_mch_in_dim,
            hidden_dim=hidden_dim,
            local_layers=2
        )

        # candidate policy head
        self.policy = nn.ModuleList()
        if policy_l == 1:
            self.policy.append(
                Sequential(
                    Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    Linear(hidden_dim, hidden_dim)
                )
            )
        else:
            for i in range(policy_l):
                self.policy.append(
                    Sequential(
                        Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                        Linear(hidden_dim, hidden_dim)
                    )
                )

        self.action_head = Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            Linear(hidden_dim, 1)
        )

    def forward(self, batch_states, batch_window_states, feasible_actions):
        # ===== 1) global embedding =====
        node_embed_gin, graph_embed_gin = self.global_embedding_gin(
            batch_states.x,
            add_self_loops(torch.cat([batch_states.edge_index_pc, batch_states.edge_index_mc], dim=-1))[0],
            batch_states.batch
        )

        node_embed_dghan, graph_embed_dghan = self.global_embedding_dghan(
            batch_states.x,
            add_self_loops(batch_states.edge_index_pc)[0],
            add_self_loops(batch_states.edge_index_mc)[0],
            len(feasible_actions)
        )

        node_embed = torch.cat([node_embed_gin, node_embed_dghan], dim=-1)    # [B*N, 2H]
        graph_embed = torch.cat([graph_embed_gin, graph_embed_dghan], dim=-1) # [B, 2H]

        batch_size = graph_embed.shape[0]
        n_nodes_per_state = node_embed.shape[0] // batch_size

        graph_embed_expand = graph_embed.repeat_interleave(n_nodes_per_state, dim=0)
        global_op_embed = torch.cat([node_embed, graph_embed_expand], dim=-1) # [B*N, 4H]
        global_op_embed = global_op_embed.reshape(batch_size, n_nodes_per_state, -1)

        # ===== 2) candidate scoring (vectorized across the whole batch) =====
        device = batch_states.x.device

        owner_per_window = []        # b for each window
        action_per_window = []       # anchor node id for each window
        candidates_per_batch = [[] for _ in range(batch_size)]

        op_features_list = []
        mch_features_list = []
        op_global_list = []
        op_machine_id_list = []
        op_window_id_list = []
        mch_window_id_list = []
        anchor_global_idx_list = []
        edges_pc_list = []
        edges_mc_list = []

        op_offset = 0
        mch_offset = 0

        for b, candidate_windows in enumerate(batch_window_states):
            for ws in candidate_windows:
                K = ws["op_features"].size(0)
                M = ws["mch_features"].size(0)
                w_id = len(owner_per_window)

                op_features_list.append(ws["op_features"])
                mch_features_list.append(ws["mch_features"])
                op_global_list.append(global_op_embed[b].index_select(0, ws["op_ids"]))
                op_machine_id_list.append(ws["op_machine_id"] + mch_offset)
                op_window_id_list.append(torch.full((K,), w_id, dtype=torch.long, device=device))
                mch_window_id_list.append(torch.full((M,), w_id, dtype=torch.long, device=device))
                anchor_global_idx_list.append(op_offset + int(ws["anchor_local_idx"]))
                edges_pc_list.append(ws["edge_index_pc"] + op_offset)
                edges_mc_list.append(ws["edge_index_mc"] + op_offset)

                owner_per_window.append(b)
                action_per_window.append(int(ws["action"]))
                candidates_per_batch[b].append(w_id)

                op_offset += K
                mch_offset += M

        num_windows = len(owner_per_window)

        sampled_actions = [0] * batch_size
        log_probs = [torch.zeros((), device=device) for _ in range(batch_size)]

        if num_windows == 0:
            log_prob = torch.stack(log_probs, dim=0).unsqueeze(-1)
            return sampled_actions, log_prob

        op_features_cat = torch.cat(op_features_list, dim=0)
        mch_features_cat = torch.cat(mch_features_list, dim=0)
        op_global_cat = torch.cat(op_global_list, dim=0)
        op_machine_id_cat = torch.cat(op_machine_id_list, dim=0)
        op_window_id_cat = torch.cat(op_window_id_list, dim=0)
        mch_window_id_cat = torch.cat(mch_window_id_list, dim=0)
        anchor_idx_cat = torch.tensor(anchor_global_idx_list, dtype=torch.long, device=device)
        edges_pc_cat = torch.cat(edges_pc_list, dim=1) if edges_pc_list else torch.zeros((2, 0), dtype=torch.long, device=device)
        edges_mc_cat = torch.cat(edges_mc_list, dim=1) if edges_mc_list else torch.zeros((2, 0), dtype=torch.long, device=device)

        action_h = self.window_encoder.forward_batched(
            op_global=op_global_cat,
            op_features=op_features_cat,
            mch_features=mch_features_cat,
            op_machine_id=op_machine_id_cat,
            op_window_id=op_window_id_cat,
            mch_window_id=mch_window_id_cat,
            anchor_global_idx=anchor_idx_cat,
            edge_index_pc=edges_pc_cat,
            edge_index_mc=edges_mc_cat,
            num_windows=num_windows,
        )                                                               # [W, H]

        for layer in self.policy:
            action_h = layer(action_h)
        scores = self.action_head(action_h).squeeze(-1)                 # [W]

        for b in range(batch_size):
            cand_idx = candidates_per_batch[b]
            if len(cand_idx) == 0:
                continue
            cand_scores = scores[torch.tensor(cand_idx, dtype=torch.long, device=device)]
            dist = Categorical(logits=cand_scores)
            idx = dist.sample()
            sampled_actions[b] = action_per_window[cand_idx[int(idx.item())]]
            log_probs[b] = dist.log_prob(idx)

        log_prob = torch.stack(log_probs, dim=0).unsqueeze(-1)
        return sampled_actions, log_prob

def test_pineline():
    import random
    from env.environment import JsspWindow, BatchGraph
    from env.generateJSP import uni_instance_gen
    from types import SimpleNamespace

    def _bind_main_process_away_from_cpu0():
        if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
            return None
        current = set(os.sched_getaffinity(0))
        allowed = {cpu for cpu in current if cpu != 0}
        target = allowed if allowed else current
        os.sched_setaffinity(0, target)
        return set(os.sched_getaffinity(0))

    affinity = _bind_main_process_away_from_cpu0()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_job, n_mch = 10, 10
    low, high = 1, 99
    num_instances = 2

    env = JsspWindow(
        n_job=n_job,
        n_mch=n_mch,
        low=low,
        high=high,
        cp_solver_time=1,
        cp_solver_cpu=1,
        cpu_budget=1,
        window_size=70,
    )

    instances = np.array([
        uni_instance_gen(n_job, n_mch, low, high)
        for _ in range(num_instances)
    ])

    states, batch_window_states, feasible_actions, done = env.reset(
        instances,
        init_type="spt-pdr",
        device=device,
        plot=False,
    )

    x, edge_index_pc, edge_index_mc, batch = states

    print("==== Global state ====")
    print("x.shape =", x.shape)
    print("edge_index_pc.shape =", edge_index_pc.shape)
    print("edge_index_mc.shape =", edge_index_mc.shape)
    print("batch.shape =", batch.shape)

    print("\n==== Local window states ====")
    print("batch size =", len(batch_window_states))
    print("num feasible actions =", len(feasible_actions[0]))
    print("done =", done)

    # 打印一个 candidate 看结构是否正确
    first_candidate = batch_window_states[0][0]
    print("\nOne candidate keys:", first_candidate.keys())
    for k, v in first_candidate.items():
        if torch.is_tensor(v):
            print(k, v.shape, v.dtype)
        else:
            print(k, type(v), v)

    # 先构造成 actor 现在习惯的 batch_states 形式
    batch_states = SimpleNamespace(
        x=x,
        edge_index_pc=edge_index_pc,
        edge_index_mc=edge_index_mc,
        batch=batch,
    )

    # 这里的输入维度你要按你实际 actor 定义改
    actor = Actor(
        in_dim=x.size(-1),              # 全局节点特征维度
        hidden_dim=128,
        window_op_in_dim=10,            # 你前面定的是 10
        window_mch_in_dim=8,            # 你前面定的是 8
        embedding_l=4,
        policy_l=3,
        heads=4,
        dropout=0.0,                    # 测试时先关掉高 dropout
    ).to(device)

    actor.eval()

    with torch.no_grad():
        sampled_actions, log_prob = actor(
            batch_states=batch_states,
            batch_window_states=batch_window_states,
            feasible_actions=feasible_actions,
        )

    print("\n==== Actor output ====")
    print("sampled_actions =", sampled_actions)
    print("log_prob.shape =", log_prob.shape)
    print("log_prob =", log_prob)

    # 检查 sampled action 是否合法
    for i, a in enumerate(sampled_actions):
        assert a in feasible_actions[i], f"sampled action {a} not in feasible_actions[{i}]"

    print("\nActor forward smoke test passed.")

    # 再做一步环境交互测试
    next_states, next_batch_window_states, reward, next_feasible_actions, next_done = env.step(
        sampled_actions,
        device=device,
        plot=False,
    )

    print("\n==== Env step after actor ====")
    print("reward =", reward)
    print("next done =", next_done)
    print("next feasible num =", len(next_feasible_actions[0]))

    print("\nActor + Env interaction test passed.")

    actor.train()
    torch.autograd.set_detect_anomaly(True)
    sampled_actions, log_prob = actor(
        batch_states=batch_states,
        batch_window_states=batch_window_states,
        feasible_actions=feasible_actions,
    )
    loss = -log_prob.mean()
    loss.backward()
    print("backward passed.")


if __name__ == "__main__":
    import random
    from env.environment import JsspWindow, BatchGraph
    from env.generateJSP import uni_instance_gen
    from types import SimpleNamespace
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_job, n_mch = 10, 10
    low, high = 1, 99
    num_instances = 6

    env = JsspWindow(
        n_job=n_job,
        n_mch=n_mch,
        low=low,
        high=high,
        cp_solver_time=1,
        cp_solver_cpu=1,
        cpu_budget=1,
        window_size=70,
    )

    instances = np.array([
        uni_instance_gen(n_job, n_mch, low, high)
        for _ in range(num_instances)
    ])

    states, batch_window_states, feasible_actions, done = env.reset(
        instances,
        init_type="spt-pdr",
        device=device,
        plot=False,
    )

    x, edge_index_pc, edge_index_mc, batch = states

    print("==== Global state ====")
    print("x.shape =", x.shape)
    print("edge_index_pc.shape =", edge_index_pc.shape)
    print("edge_index_mc.shape =", edge_index_mc.shape)
    print("batch.shape =", batch.shape)

    print("\n==== Batch local windows ====")
    print("batch size =", len(batch_window_states))
    print("feasible lens =", [len(a) for a in feasible_actions])
    print("done =", done.squeeze(-1).tolist())

    assert len(batch_window_states) == num_instances
    assert len(feasible_actions) == num_instances
    assert done.shape[0] == num_instances

    for b in range(num_instances):
        assert len(batch_window_states[b]) == len(feasible_actions[b])
        for ws in batch_window_states[b]:
            assert ws["op_features"].shape[0] == ws["op_ids"].shape[0]
            assert ws["op_machine_id"].shape[0] == ws["op_ids"].shape[0]
            assert ws["op_features"].shape[1] == 10
            assert ws["mch_features"].shape[1] == 8

    batch_states = SimpleNamespace(
        x=x,
        edge_index_pc=edge_index_pc,
        edge_index_mc=edge_index_mc,
        batch=batch,
    )

    actor = Actor(
        in_dim=x.size(-1),
        hidden_dim=128,
        window_op_in_dim=10,
        window_mch_in_dim=8,
        embedding_l=4,
        policy_l=3,
        heads=4,
        dropout=0.0,
    ).to(device)

    actor.train()

    sampled_actions, log_prob = actor(
        batch_states=batch_states,
        batch_window_states=batch_window_states,
        feasible_actions=feasible_actions,
    )

    print("\n==== Actor output ====")
    print("sampled_actions =", sampled_actions)
    print("log_prob.shape =", log_prob.shape)

    assert len(sampled_actions) == num_instances
    assert log_prob.shape == (num_instances, 1)

    for b in range(num_instances):
        assert sampled_actions[b] in feasible_actions[b], \
            f"sampled action {sampled_actions[b]} not in feasible_actions[{b}]"

    next_states, next_batch_window_states, reward, next_feasible_actions, next_done = env.step(
        sampled_actions,
        device=device,
        plot=False,
    )

    print("\n==== Env step ====")
    print("reward.shape =", reward.shape)
    print("next feasible lens =", [len(a) for a in next_feasible_actions])

    loss = -log_prob.mean()
    actor.zero_grad(set_to_none=True)
    loss.backward()

    grad_none = []
    for name, p in actor.named_parameters():
        if p.requires_grad and p.grad is None:
            grad_none.append(name)

    print("\n==== Backward ====")
    print("loss =", loss.item())
    print("params without grad =", grad_none)

    print("\nBatch smoke test passed.")

    for t in range(10):
        sampled_actions, log_prob = actor(
            batch_states=batch_states,
            batch_window_states=batch_window_states,
            feasible_actions=feasible_actions,
        )
        states, batch_window_states, reward, feasible_actions, done = env.step(
            sampled_actions,
            device=device,
            plot=False,
        )
        loss = -log_prob.mean()
        actor.zero_grad(set_to_none=True)
        loss.backward()


