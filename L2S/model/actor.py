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
        device = x.device
        out = torch.zeros(num_groups, x.size(-1), device=device)
        cnt = torch.zeros(num_groups, 1, device=device)
        out.index_add_(0, group_idx, x)
        ones = torch.ones(group_idx.size(0), 1, device=device)
        cnt.index_add_(0, group_idx, ones)
        return out / cnt.clamp(min=1.0)

    def _pool_max(self, x, group_idx, num_groups):
        device = x.device
        out = torch.full((num_groups, x.size(-1)), -1e9, device=device)
        for i in range(x.size(0)):
            g = group_idx[i].item()
            out[g] = torch.maximum(out[g], x[i])
        out[out < -1e8] = 0.0
        return out

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

        # ===== 2) candidate scoring =====
        sampled_actions = []
        log_probs = []

        for b in range(batch_size):
            candidate_windows = batch_window_states[b]

            if len(candidate_windows) == 0:
                sampled_actions.append(0)
                log_probs.append(torch.zeros(1, device=batch_states.x.device).squeeze(0))
                continue

            cand_scores = []
            cand_actions = []

            for window_state in candidate_windows:
                action_h = self.window_encoder(global_op_embed[b], window_state)

                for layer in self.policy:
                    action_h = layer(action_h)

                score = self.action_head(action_h).squeeze(-1)

                cand_scores.append(score)
                cand_actions.append(window_state["action"])

            cand_scores = torch.stack(cand_scores, dim=0)   # [A]
            dist = Categorical(logits=cand_scores)
            idx = dist.sample()

            sampled_actions.append(cand_actions[idx.item()])
            log_probs.append(dist.log_prob(idx))

        log_prob = torch.stack(log_probs, dim=0).unsqueeze(-1)
        return sampled_actions, log_prob

if __name__ == '__main__':
    import random
    from env.environment import JsspWindow, BatchGraph

    def _bind_main_process_away_from_cpu0():
        if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
            return None
        current = set(os.sched_getaffinity(0))
        allowed = {cpu for cpu in current if cpu != 0}
        target = allowed if allowed else current
        os.sched_setaffinity(0, target)
        return set(os.sched_getaffinity(0))

    affinity = _bind_main_process_away_from_cpu0()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_j = 20
    n_m = 15
    l = 1
    h = 99
    hid_dim = 128

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    actor = Actor(
        in_dim=3,
        hidden_dim=hid_dim,
        embedding_l=4,
        policy_l=4,
        embedding_type='gin+dghan',
        heads=1,
        dropout=0.0
    ).to(dev)

    print(actor)