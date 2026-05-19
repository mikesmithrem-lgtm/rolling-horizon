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


class Actor(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 embedding_l=4,
                 policy_l=3,
                 embedding_type='gin',
                 heads=4,
                 dropout=0.6):
        super(Actor, self).__init__()
        self.embedding_l = embedding_l
        self.policy_l = policy_l
        self.embedding_type = embedding_type
        self.hidden_dim = hidden_dim

        if self.embedding_type == 'gin':
            self.embedding = GIN(in_dim=in_dim, hidden_dim=hidden_dim, layer_gin=embedding_l)
            base_embed_dim = hidden_dim
        elif self.embedding_type == 'dghan':
            self.embedding = DGHAN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, layer_dghan=embedding_l, heads=heads)
            base_embed_dim = hidden_dim
        elif self.embedding_type == 'gin+dghan':
            self.embedding_gin = GIN(in_dim=in_dim, hidden_dim=hidden_dim, layer_gin=embedding_l)
            self.embedding_dghan = DGHAN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, layer_dghan=embedding_l, heads=heads)
            base_embed_dim = hidden_dim * 2
        else:
            raise Exception('embedding type should be either "gin", "dghan", or "gin+dghan".')

        policy_input_dim = base_embed_dim * 2

        self.policy = torch.nn.ModuleList()
        if policy_l == 1:
            self.policy.append(
                Sequential(
                    Linear(policy_input_dim, hidden_dim),
                    torch.nn.Tanh(),
                    Linear(hidden_dim, hidden_dim)
                )
            )
        else:
            for layer in range(policy_l):
                if layer == 0:
                    self.policy.append(
                        Sequential(
                            Linear(policy_input_dim, hidden_dim),
                            torch.nn.Tanh(),
                            Linear(hidden_dim, hidden_dim)
                        )
                    )
                else:
                    self.policy.append(
                        Sequential(
                            Linear(hidden_dim, hidden_dim),
                            torch.nn.Tanh(),
                            Linear(hidden_dim, hidden_dim)
                        )
                    )

        # window branch
        # action_machine_feat: [A, n_mch, 9]
        # last dim = 8 numeric feats + 1 valid mask
        self.machine_feat_encoder = Sequential(
            Linear(8, hidden_dim),
            torch.nn.Tanh(),
            Linear(hidden_dim, hidden_dim)
        )

        # anchor_embed + machine_mean + machine_max + scalar_embed
        self.action_bonus_head = Sequential(
            Linear(hidden_dim * 3, hidden_dim),
            torch.nn.Tanh(),
            Linear(hidden_dim, 1)
        )

        self.action_head = Sequential(
            Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            Linear(hidden_dim, 1)
        )

    @staticmethod
    def _masked_mean_pool(machine_embed, machine_mask):
        denom = machine_mask.sum(dim=1).clamp(min=1.0)
        return (machine_embed * machine_mask).sum(dim=1) / denom

    @staticmethod
    def _masked_max_pool(machine_embed, machine_mask):
        neg_large = torch.full_like(machine_embed, -1e9)
        masked_embed = torch.where(machine_mask.bool(), machine_embed, neg_large)
        pooled = masked_embed.max(dim=1).values
        invalid_row = (machine_mask.sum(dim=1).squeeze(-1) <= 0)
        if invalid_row.any():
            pooled[invalid_row] = 0.0
        return pooled

    def _window_bonus(self,
                      node_embed_augmented,
                      feasible_actions,
                      action_machine_feat):
        if action_machine_feat is None:
            return None

        device = node_embed_augmented.device
        batch_size, n_nodes_per_state = node_embed_augmented.shape[:2]

        action_bonus = torch.zeros(
            (batch_size, n_nodes_per_state),
            dtype=node_embed_augmented.dtype,
            device=device
        )

        for batch_idx, (state_actions, state_machine_feat) in enumerate(
                zip(feasible_actions, action_machine_feat)):

            if len(state_actions) == 0:
                continue
            if state_machine_feat is None:
                continue

            if state_machine_feat.shape[0] != len(state_actions):
                raise ValueError('action_machine_feat.shape[0] must equal number of feasible actions.')

            action_index = torch.as_tensor(state_actions, dtype=torch.long, device=device)

            machine_raw = state_machine_feat[..., :-1]
            machine_mask = state_machine_feat[..., -1:].to(dtype=node_embed_augmented.dtype)

            machine_embed = self.machine_feat_encoder(
                machine_raw.reshape(-1, machine_raw.size(-1))
            ).reshape(
                state_machine_feat.size(0),
                state_machine_feat.size(1),
                self.hidden_dim
            )

            anchor_embed = node_embed_augmented[batch_idx, action_index, :]

            action_repr = torch.cat(
                [anchor_embed, machine_embed.reshape(-1, self.hidden_dim)],
                dim=-1
            )

            action_bonus[batch_idx, action_index] = self.action_bonus_head(action_repr).squeeze(-1)

        return action_bonus

    def forward(self, batch_states, feasible_actions):
        if self.embedding_type == 'gin':
            node_embed, graph_embed = self.embedding(
                batch_states.x,
                add_self_loops(torch.cat([batch_states.edge_index_pc, batch_states.edge_index_mc], dim=-1))[0],
                batch_states.batch
            )
        elif self.embedding_type == 'dghan':
            node_embed, graph_embed = self.embedding(
                batch_states.x,
                add_self_loops(batch_states.edge_index_pc)[0],
                add_self_loops(batch_states.edge_index_mc)[0],
                len(feasible_actions)
            )
        elif self.embedding_type == 'gin+dghan':
            node_embed_gin, graph_embed_gin = self.embedding_gin(
                batch_states.x,
                add_self_loops(torch.cat([batch_states.edge_index_pc, batch_states.edge_index_mc], dim=-1))[0],
                batch_states.batch
            )
            node_embed_dghan, graph_embed_dghan = self.embedding_dghan(
                batch_states.x,
                add_self_loops(batch_states.edge_index_pc)[0],
                add_self_loops(batch_states.edge_index_mc)[0],
                len(feasible_actions)
            )
            node_embed = torch.cat([node_embed_gin, node_embed_dghan], dim=-1)
            graph_embed = torch.cat([graph_embed_gin, graph_embed_dghan], dim=-1)
        else:
            raise Exception('embedding type should be either "gin", "dghan", or "gin+dghan".')

        device = node_embed.device
        batch_size = graph_embed.shape[0]
        n_nodes_per_state = node_embed.shape[0] // batch_size

        node_embed_augmented = torch.cat(
            [
                node_embed,
                graph_embed.repeat_interleave(repeats=n_nodes_per_state, dim=0)
            ],
            dim=-1
        ).reshape(batch_size, n_nodes_per_state, -1)

        for layer in range(self.policy_l):
            node_embed_augmented = self.policy[layer](node_embed_augmented)

        action_score = self.action_head(node_embed_augmented).squeeze(-1)

        action_window_bonus = self._window_bonus(
            node_embed_augmented=node_embed_augmented,
            feasible_actions=feasible_actions,
            action_machine_feat=getattr(batch_states, 'feasible_action_machine_feat', None),
        )

        if action_window_bonus is not None:
            action_score = action_score + action_window_bonus

        mask = torch.ones((batch_size, n_nodes_per_state), dtype=torch.bool, device=device)
        for batch_idx, state_actions in enumerate(feasible_actions):
            if len(state_actions) == 0:
                continue
            action_index = torch.as_tensor(state_actions, dtype=torch.long, device=device)
            mask[batch_idx, action_index] = False

        action_score.masked_fill_(mask, -np.inf)

        dist = Categorical(logits=action_score)
        actions_id = dist.sample()
        sampled_actions = actions_id.tolist()
        log_prob = dist.log_prob(actions_id).unsqueeze(-1)

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