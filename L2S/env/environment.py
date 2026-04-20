import collections
import os
import sys
import multiprocessing as mp
import json
import logging

from torch_geometric.utils import add_self_loops, sort_edge_index

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import networkx as nx
from ortools.sat.python import cp_model
from generateJSP import uni_instance_gen
from env.permissible_LS import permissibleLeftShift
from env.message_passing_evl import Evaluator, CPM_batch_G
import matplotlib.pyplot as plt
import time
import random
from model.actor import Actor


class BatchGraph:
    def __init__(self):
        self.x = None
        self.edge_index_pc = None
        self.edge_index_mc = None
        self.batch = None

    def wrapper(self, x, edge_index_pc, edge_index_mc, batch):
        self.x = x
        self.edge_index_pc = edge_index_pc
        self.edge_index_mc = edge_index_mc
        self.batch = batch

    def clean(self):
        self.x = None
        self.edge_index_pc = None
        self.edge_index_mc = None
        self.batch = None


def _window_node_to_job_op(node, n_mch):
    """Convert graph node id to (job, operation) indices."""
    op_id = node - 1
    return op_id // n_mch, op_id % n_mch


def _window_machine_to_index(machine):
    """Convert 1-based machine ids used in L2S instances to 0-based indices for CP."""
    return int(machine) - 1


def _window_plan_ops(op_start_times, n_job, n_mch, window_size, anchor_node):
    """Build a rolling window around the selected anchor operation."""
    anchor = _window_node_to_job_op(anchor_node, n_mch)
    sorted_ops = sorted(
        [(job, op) for job in range(n_job) for op in range(n_mch)],
        key=lambda item: (op_start_times[item[0]][item[1]], item[0], item[1]),
    )
    window_idx = sorted_ops.index(anchor)
    n_oprs = n_job * n_mch
    if window_idx > n_oprs - window_size:
        window_idx = n_oprs - window_size
    return sorted_ops[window_idx:window_idx + window_size]


def _window_machine_availability(ops_in_window, op_start_times, duration, mch, n_job, n_mch):
    """Compute the time interval each machine can use inside the selected window."""
    machine_avail = {}
    makespan = max(op_start_times[job][n_mch - 1] + duration[job][n_mch - 1] for job in range(n_job))

    for machine in {int(mch[job][op]) for job, op in ops_in_window}:
        ops_on_machine = [
            (job, op) for job in range(n_job) for op in range(n_mch)
            if int(mch[job][op]) == machine
        ]
        ops_on_machine.sort(key=lambda item: (op_start_times[item[0]][item[1]], item[0], item[1]))
        idxs = [idx for idx, op in enumerate(ops_on_machine) if op in ops_in_window]
        if not idxs:
            continue

        first_idx = idxs[0]
        last_idx = idxs[-1]
        if first_idx > 0:
            prev_job, prev_op = ops_on_machine[first_idx - 1]
            earliest = int(op_start_times[prev_job][prev_op] + duration[prev_job][prev_op])
        else:
            earliest = 0

        if last_idx < len(ops_on_machine) - 1:
            next_job, next_op = ops_on_machine[last_idx + 1]
            latest = int(op_start_times[next_job][next_op])
        else:
            latest = int(makespan)

        machine_avail[_window_machine_to_index(machine)] = (earliest, latest)

    return machine_avail


def _window_solve_subproblem(instance, n_job, n_mch, ops_in_window, op_start_times, cp_solver_time, cp_solver_cpu):
    """Solve one window with CP-SAT while keeping external machine/job boundaries fixed."""
    duration = instance[0]
    mch = instance[1] - 1
    machine_avail = _window_machine_availability(ops_in_window, op_start_times, duration, instance[1], n_job, n_mch)
    horizon = max(op_start_times[job][n_mch - 1] + duration[job][n_mch - 1] for job in range(n_job))
    model = cp_model.CpModel()
    op_vars = {}

    for job, op in ops_in_window:
        machine = int(mch[job][op])
        lb, latest_bound = machine_avail[machine]
        ub = latest_bound - int(duration[job][op])
        if op > 0 and (job, op - 1) not in ops_in_window:
            prev_end = int(op_start_times[job][op - 1] + duration[job][op - 1])
            lb = max(lb, prev_end)
        if op < n_mch - 1 and (job, op + 1) not in ops_in_window:
            after_start = int(op_start_times[job][op + 1])
            ub = min(ub, after_start)
        if lb > ub:
            return None
        op_vars[(job, op)] = model.NewIntVar(lb, ub, f"start_{job}_{op}")

    for job, op in ops_in_window:
        if op > 0 and (job, op - 1) in ops_in_window:
            model.Add(op_vars[(job, op)] >= op_vars[(job, op - 1)] + int(duration[job][op - 1]))

    machine_to_ops = {}
    interval_vars = {}
    for job, op in ops_in_window:
        machine = int(mch[job][op])
        machine_to_ops.setdefault(machine, []).append((job, op))
        interval_vars[(job, op)] = model.NewIntervalVar(
            op_vars[(job, op)],
            int(duration[job][op]),
            op_vars[(job, op)] + int(duration[job][op]),
            f"interval_{job}_{op}",
        )

    for ops in machine_to_ops.values():
        model.AddNoOverlap([interval_vars[item] for item in ops])

    makespan = model.NewIntVar(0, int(horizon), "makespan")
    for job, op in ops_in_window:
        model.Add(makespan >= op_vars[(job, op)] + int(duration[job][op]))
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(cp_solver_time)
    solver.parameters.num_search_workers = int(cp_solver_cpu)
    solver.parameters.random_seed = 0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return {(job, op): solver.Value(op_vars[(job, op)]) for job, op in ops_in_window}


def _window_build_orders(instance, n_job, n_mch, start_times, ref_start_times=None):
    """Convert matrix start times into per-machine operation orders."""
    if ref_start_times is None:
        ref_start_times = start_times
    orders = [[] for _ in range(n_mch)]
    for machine_idx in range(n_mch):
        machine = machine_idx + 1
        ops = [
            (job, op) for job in range(n_job) for op in range(n_mch)
            if int(instance[1][job][op]) == machine
        ]
        ops.sort(key=lambda item: (
            start_times[item[0]][item[1]],
            ref_start_times[item[0]][item[1]],
            item[0],
            item[1],
        ))
        orders[machine_idx] = ops
    return orders


def _window_schedule_from_orders(instance, n_job, n_mch, orders):
    """Rebuild a globally feasible schedule from machine orders."""
    duration = instance[0]
    op_start_times = np.zeros((n_job, n_mch), dtype=np.int32)
    machine_ready = [0] * n_mch
    job_ready = [0] * n_job
    order_idx = [0] * n_mch
    scheduled_ops = set()

    while len(scheduled_ops) < n_job * n_mch:
        progress = False
        for machine_idx in range(n_mch):
            if order_idx[machine_idx] >= len(orders[machine_idx]):
                continue
            job, op = orders[machine_idx][order_idx[machine_idx]]
            if (job, op) in scheduled_ops:
                order_idx[machine_idx] += 1
                continue
            if op > 0 and (job, op - 1) not in scheduled_ops:
                continue

            start_time = max(job_ready[job], machine_ready[machine_idx])
            op_start_times[job][op] = start_time
            end_time = start_time + int(duration[job][op])
            job_ready[job] = end_time
            machine_ready[machine_idx] = end_time
            scheduled_ops.add((job, op))
            order_idx[machine_idx] += 1
            progress = True

        if not progress:
            raise ValueError("Failed to rebuild a feasible schedule from machine orders.")

    return op_start_times


def _window_build_adj_mat_mc(n_job, n_mch, orders):
    """Build the augmented machine-clique adjacency matrix kept by the environment."""
    n_oprs = n_job * n_mch
    adj_mat_mc = np.zeros(shape=[n_oprs, n_oprs], dtype=int)
    for machine_ops in orders:
        for prev, curr in zip(machine_ops[:-1], machine_ops[1:]):
            prev_op_id = prev[0] * n_mch + prev[1]
            curr_op_id = curr[0] * n_mch + curr[1]
            adj_mat_mc[curr_op_id, prev_op_id] = 1
    adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant', constant_values=0)
    return np.transpose(adj_mat_mc)


def _window_solve_single(args):
    """Worker entry for one batch item; returns only serializable matrix data."""
    idx, instance, base_start_times, action, n_job, n_mch, window_size, cp_solver_time, cp_solver_cpu = args
    if action == [0, 0]:
        return {"index": idx, "applied": False}

    anchor_node = int(action[0])
    if anchor_node <= 0 or anchor_node > n_job * n_mch:
        return {"index": idx, "applied": False}

    window_ops = _window_plan_ops(base_start_times, n_job, n_mch, window_size, anchor_node)
    result = _window_solve_subproblem(
        instance=instance,
        n_job=n_job,
        n_mch=n_mch,
        ops_in_window=window_ops,
        op_start_times=base_start_times,
        cp_solver_time=cp_solver_time,
        cp_solver_cpu=cp_solver_cpu,
    )
    if result is None:
        return {"index": idx, "applied": False}

    candidate_start = np.array(base_start_times, copy=True)
    for (job, op), start_time in result.items():
        candidate_start[job][op] = int(start_time)

    orders = _window_build_orders(instance, n_job, n_mch, candidate_start, base_start_times)
    new_start_times = _window_schedule_from_orders(instance, n_job, n_mch, orders)
    adj_mat_mc = _window_build_adj_mat_mc(n_job, n_mch, orders)
    return {
        "index": idx,
        "applied": True,
        "start_times": new_start_times,
        "orders": orders,
        "adj_mat_mc": adj_mat_mc,
    }


class JsspN5:
    def __init__(self, n_job, n_mch, low, high, reward_type='yaoxin', fea_norm_const=1000, evaluator_type='message-passing'):

        self.n_job = n_job
        self.n_mch = n_mch
        self.n_oprs = self.n_job * self.n_mch
        self.low = low
        self.high = high
        self.itr = 0
        self.instances = None
        self.sub_graphs_mc = None
        self.current_graphs = None
        self.current_objs = None
        self.tabu_size = 1
        self.tabu_lists = None
        self.incumbent_objs = None
        self.reward_type = reward_type
        self.fea_norm_const = fea_norm_const
        self.evaluator_type = evaluator_type
        self.eva = Evaluator() if evaluator_type == 'message-passing' else CPM_batch_G
        self.adj_mat_pc = self._adj_mat_pc()


    def _adj_mat_pc(self):
        # Create adjacent matrix for precedence constraints
        adj_mat_pc = np.eye(self.n_oprs, k=-1, dtype=int)
        # 每个job的第一个operation没有neighborhood
        adj_mat_pc[np.arange(start=0, stop=self.n_oprs, step=1).reshape(self.n_job, -1)[:, 0]] = 0
        # pad dummy S and T nodes
        adj_mat_pc = np.pad(adj_mat_pc, 1, 'constant', constant_values=0)
        # connect S with 1st operation of each job
        adj_mat_pc[[i for i in range(1, self.n_job * self.n_mch + 2 - 1, self.n_mch)], 0] = 1
        # connect last operation of each job to T
        adj_mat_pc[-1, [i for i in range(self.n_mch, self.n_job * self.n_mch + 2 - 1, self.n_mch)]] = 1
        # convert input adj from column pointing to row, to, row pointing to column
        adj_mat_pc = np.transpose(adj_mat_pc)
        return adj_mat_pc


    def _gen_moves(self, solution, mch_mat, tabu_list=None):
        """
        solution: networkx DAG conjunctive graph
        mch_mat: the same mch from our NeurIPS 2020 paper of solution
        """
        critical_path = nx.dag_longest_path(solution)[1:-1]
        critical_blocks_opr = np.array(critical_path)
        critical_blocks = mch_mat.take(critical_blocks_opr - 1)  # -1: ops id starting from 0
        pairs = self._get_pairs(critical_blocks, critical_blocks_opr, tabu_list)
        return pairs

    @staticmethod
    def _get_pairs(cb, cb_op, tabu_list=None):
        # 找到关键路径即可
        pairs = []
        rg = cb[:-1].shape[0]  # sliding window of 2
        for i in range(rg):
            if cb[i] == cb[i + 1]:  # find potential pair
                if i == 0:
                    if cb[i + 1] != cb[i + 2]:
                        if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                            pairs.append([cb_op[i], cb_op[i + 1]])
                elif cb[i] != cb[i - 1]:
                    if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                        pairs.append([cb_op[i], cb_op[i + 1]])
                elif i + 1 == rg:
                    if cb[i + 1] != cb[i]:
                        if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                            pairs.append([cb_op[i], cb_op[i + 1]])
                elif cb[i + 1] != cb[i + 2]:
                    if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                        pairs.append([cb_op[i], cb_op[i + 1]])
                else:
                    pass
        return pairs

    @staticmethod
    def _get_pairs_has_tabu(cb, cb_op):
        pairs = []
        rg = cb[:-1].shape[0]  # sliding window of 2
        for i in range(rg):
            if cb[i] == cb[i + 1]:  # find potential pair
                if i == 0:
                    if cb[i + 1] != cb[i + 2]:
                        pairs.append([cb_op[i], cb_op[i + 1]])
                elif cb[i] != cb[i - 1]:
                    pairs.append([cb_op[i], cb_op[i + 1]])
                elif i + 1 == rg:
                    if cb[i + 1] != cb[i]:
                        pairs.append([cb_op[i], cb_op[i + 1]])
                elif cb[i + 1] != cb[i + 2]:
                    pairs.append([cb_op[i], cb_op[i + 1]])
                else:
                    pass
        return pairs

    def show_state(self, G):
        x_axis = np.pad(np.tile(np.arange(1, self.n_mch + 1, 1), self.n_job), (1, 1), 'constant', constant_values=[0, self.n_mch + 1])
        y_axis = np.pad(np.arange(self.n_job, 0, -1).repeat(self.n_mch), (1, 1), 'constant', constant_values=np.median(np.arange(self.n_job, 0, -1)))
        pos = dict((n, (x, y)) for n, x, y in zip(G.nodes(), x_axis, y_axis))
        plt.figure(figsize=(15, 10))
        plt.tight_layout()
        nx.draw_networkx_edge_labels(G, pos=pos)  # show edge weight
        nx.draw(
            G, pos=pos, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1'
            # <-- tune curvature and style ref:https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.ConnectionStyle.html
        )
        plt.show()

    def _p_list_solver(self, args, plot=False):
        instances, priority_lists, device = args[0], args[1], args[2]

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        current_graphs = []
        sub_graphs_mc = []
        for i, (instance, priority_list) in enumerate(zip(instances, priority_lists)):
            dur_mat, mch_mat = instance[0], instance[1]
            n_jobs = mch_mat.shape[0]
            n_machines = mch_mat.shape[1]
            n_operations = n_jobs * n_machines

            # prepare NIPS adj

            # Init operations mat, shapes [job, machine]
            ops_mat = np.arange(0, n_operations).reshape(mch_mat.shape).tolist()
            # Init list_for_latest_task_onMachine, [machine]
            list_for_latest_task_onMachine = [None] * n_machines
            # Init adjacent matrix for machine clique
            adj_mat_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)
            # Construct NIPS adjacent matrix only for machine cliques
            for job_id in priority_list:
                op_id = ops_mat[job_id][0]
                m_id_for_action = mch_mat[op_id // n_machines, op_id % n_machines] - 1
                if list_for_latest_task_onMachine[m_id_for_action] is not None:
                    adj_mat_mc[op_id, list_for_latest_task_onMachine[m_id_for_action]] = 1
                list_for_latest_task_onMachine[m_id_for_action] = op_id
                ops_mat[job_id].pop(0)
            adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant', constant_values=0)  # add S and T to machine clique adj
            adj_mat_mc = np.transpose(adj_mat_mc)  # convert input adj from column pointing to row, to, row pointing to column

            adj_all = self.adj_mat_pc + adj_mat_mc
            dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(self.n_oprs + 2, axis=1)
            edge_weight = np.multiply(adj_all, dur_mat)
            G = nx.from_numpy_array(edge_weight, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            G.add_weighted_edges_from([(0, i, 0) for i in range(1, self.n_oprs + 2 - 1, self.n_mch)])  # add release time, here all jobs are available at t=0. This is the only way to add release date. And if you do not add release date, startime computation will return wired value
            current_graphs.append(G)
            G_mc = nx.from_numpy_array(adj_mat_mc, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            sub_graphs_mc.append(G_mc)

            if plot:
                self.show_state(G)

            edge_indices_pc.append((torch.nonzero(torch.from_numpy(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)
            durations.append(torch.from_numpy(dur_mat[:, 0]).to(device))

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)

        durations = torch.cat(durations, dim=0).reshape(-1, 1)
        if self.evaluator_type == 'message-passing':
            est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)
        else:
            est, lst, make_span = self.eva(current_graphs, dev=device)

            # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=self.n_job * self.n_mch + 2)).to(device)

        return (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span


    def _rules_solver(self, args, plot=False):
        instances, device, rule_type = args[0], args[1], args[2]

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        current_graphs = []
        sub_graphs_mc = []
        for i, instance in enumerate(instances):
            dur_mat, dur_cp, mch_mat = instance[0], np.copy(instance[0]), instance[1]
            n_jobs, n_machines = dur_mat.shape[0], dur_mat.shape[1]
            n_operations = n_jobs * n_machines
            last_col = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, -1]
            candidate_oprs = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:,0]  # initialize action space: [n_jobs, 1], the first column
            mask = np.zeros(shape=n_jobs, dtype=bool)  # initialize the mask: [n_jobs, 1]
            adj_mat_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)  # Create adjacent matrix for machine clique

            gant_chart = -self.high * np.ones_like(dur_mat.transpose(), dtype=np.int32)
            opIDsOnMchs = -n_jobs * np.ones_like(dur_mat.transpose(), dtype=np.int32)
            finished_mark = np.zeros_like(mch_mat, dtype=np.int32)

            actions = []
            for _ in range(n_operations):

                if rule_type == 'spt':
                    candidate_masked = candidate_oprs[np.where(~mask)]
                    dur_candidate = np.take(dur_mat, candidate_masked)
                    idx = np.random.choice(np.where(dur_candidate == np.min(dur_candidate))[0])
                    action = candidate_masked[idx]
                elif rule_type == 'fdd-divide-mwkr':
                    candidate_masked = candidate_oprs[np.where(~mask)]
                    fdd = np.take(np.cumsum(dur_mat, axis=1), candidate_masked)
                    wkr = np.take(np.cumsum(np.multiply(dur_mat, 1 - finished_mark), axis=1), last_col[np.where(~mask)])
                    priority = fdd / wkr
                    idx = np.random.choice(np.where(priority == np.min(priority))[0])
                    action = candidate_masked[idx]
                else:
                    action = None
                actions.append(action)

                permissibleLeftShift(a=action, durMat=dur_mat, mchMat=mch_mat, mchsStartTimes=gant_chart,
                                     opIDsOnMchs=opIDsOnMchs)

                # update action space or mask
                if action not in last_col:
                    candidate_oprs[action // n_machines] += 1
                else:
                    mask[action // n_machines] = 1
                # update finished_mark:
                finished_mark[action // n_machines, action % n_machines] = 1

            for _ in range(opIDsOnMchs.shape[1] - 1):
                adj_mat_mc[opIDsOnMchs[:, _ + 1], opIDsOnMchs[:, _]] = 1

            # prepare augmented adj, augmented dur, and G
            adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant', constant_values=0)  # add S and T to machine clique adj
            adj_mat_mc = np.transpose(adj_mat_mc)  # convert input adj from column pointing to row, to, row pointing to column
            adj_all = self.adj_mat_pc + adj_mat_mc
            dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(self.n_oprs + 2, axis=1)
            edge_weight = np.multiply(adj_all, dur_mat)
            G = nx.from_numpy_array(edge_weight, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            G.add_weighted_edges_from([(0, i, 0) for i in range(1, self.n_oprs + 2 - 1, self.n_mch)])  # add release time, here all jobs are available at t=0. This is the only way to add release date. And if you do not add release date, startime computation will return wired value
            current_graphs.append(G)
            G_mc = nx.from_numpy_array(adj_mat_mc, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            sub_graphs_mc.append(G_mc)

            if plot:
                self.show_state(G)

            edge_indices_pc.append((torch.nonzero(torch.from_numpy(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)
            durations.append(torch.from_numpy(dur_mat[:, 0]).to(device))

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)
        durations = torch.cat(durations, dim=0).reshape(-1, 1)
        if self.evaluator_type == 'message-passing':
            est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)
        else:
            est, lst, make_span = self.eva(current_graphs, dev=device)

        # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=self.n_job * self.n_mch + 2)).to(device)

        return (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span

    def dag2pyg(self, instances, nx_graphs, device):
        n_jobs, n_machines = instances[0][0].shape
        n_operations = n_jobs * n_machines

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        for i, (instance, G_mc) in enumerate(zip(instances, nx_graphs)):
            durations.append(np.pad(instance[0].reshape(-1), (1, 1), 'constant', constant_values=0))
            adj_mat_mc = nx.adjacency_matrix(G_mc, weight=None).todense()
            edge_indices_pc.append((torch.nonzero(torch.from_numpy(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)
        durations = torch.from_numpy(np.concatenate(durations)).reshape(-1, 1).to(device)
        if self.evaluator_type == 'message-passing':
            est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)
        else:
            est, lst, make_span = self.eva(self.current_graphs, dev=device)
        # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=n_jobs * n_machines + 2)).to(device)

        return x, edge_indices_pc, edge_indices_mc, batch, make_span

    def change_nxgraph_topology(self, actions, plot=False):
        n_jobs, n_machines = self.instances[0][0].shape
        n_operations = n_jobs * n_machines

        for i, (action, G, G_mc, instance) in enumerate(zip(actions, self.current_graphs, self.sub_graphs_mc, self.instances)):
            if action == [0, 0]:  # if dummy action then do not transit
                pass
            else:  # change nx graph topology
                S = [s for s in G.predecessors(action[0]) if int((s - 1) // n_machines) != int((action[0] - 1) // n_machines) and s != 0]
                T = [t for t in G.successors(action[1]) if int((t - 1) // n_machines) != int((action[1] - 1) // n_machines) and t != n_operations + 1]
                s = S[0] if len(S) != 0 else None
                t = T[0] if len(T) != 0 else None

                if s is not None:  # connect s with action[1]
                    G.remove_edge(s, action[0])
                    G.add_edge(s, action[1], weight=np.take(instance[0], s - 1))
                    G_mc.remove_edge(s, action[0])
                    G_mc.add_edge(s, action[1], weight=np.take(instance[0], s - 1))
                else:
                    pass

                if t is not None:  # connect action[0] with t
                    G.remove_edge(action[1], t)
                    G.add_edge(action[0], t, weight=np.take(instance[0], action[0] - 1))
                    G_mc.remove_edge(action[1], t)
                    G_mc.add_edge(action[0], t, weight=np.take(instance[0], action[0] - 1))
                else:
                    pass

                # reverse edge connecting selected pair
                G.remove_edge(action[0], action[1])
                G.add_edge(action[1], action[0], weight=np.take(instance[0], action[1] - 1))
                G_mc.remove_edge(action[0], action[1])
                G_mc.add_edge(action[1], action[0], weight=np.take(instance[0], action[1] - 1))

            if plot:
                self.show_state(G)

    def step(self, actions, device, plot=False):
        self.change_nxgraph_topology(actions, plot)  # change graph topology
        x, edge_indices_pc, edge_indices_mc, batch, makespan = self.dag2pyg(self.instances, self.sub_graphs_mc, device)  # generate new state data
        if self.reward_type == 'consecutive':
            reward = self.current_objs - makespan
        elif self.reward_type == 'yaoxin':
            reward = torch.where(self.incumbent_objs - makespan > 0, self.incumbent_objs - makespan, torch.tensor(0, dtype=torch.float32, device=device))
        else:
            raise ValueError('reward type must be "yaoxin" or "consecutive".')

        self.incumbent_objs = torch.where(makespan - self.incumbent_objs < 0, makespan, self.incumbent_objs)
        self.current_objs = makespan

        # update tabu list
        if self.tabu_size != 0:
            action_reversed = [a[::-1] for a in actions]
            for i, action in enumerate(action_reversed):
                if action == [0, 0]:  # if dummy action, don't update tabu list
                    pass
                else:
                    if len(self.tabu_lists[i]) == self.tabu_size:
                        self.tabu_lists[i].pop(0)
                        self.tabu_lists[i].append(action)
                    else:
                        self.tabu_lists[i].append(action)

        self.itr = self.itr + 1


        feasible_actions, flag = self.feasible_actions(device)  # new feasible actions w.r.t updated tabu list

        return (x, edge_indices_pc, edge_indices_mc, batch), reward, feasible_actions, ~flag

    def reset(self, instances, init_type, device, plot=False):
        '''
        Reset :通过指定的方式生成instance的初始解
        :param instances:
        :param init_type: str, [plist, 优先调度], [spt, 最短处理时间], [fdd-divide-mwkr]
        :param device: str, 设备名称
        :param plot: bool, 是否画图
        :return: 初始解
        '''
        self.instances = instances
        if init_type == 'plist':
            # fixed random plist for all instances
            plist = np.repeat(np.random.permutation(np.arange(self.n_job).repeat(self.n_mch)).reshape(1, -1), repeats=self.instances.shape[0], axis=0)
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span = self._p_list_solver(args=[self.instances, plist, device], plot=plot)
        elif init_type == 'spt':
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span = self._rules_solver(args=[self.instances, device, 'spt'], plot=plot)
        elif init_type == 'fdd-divide-mwkr':
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span = self._rules_solver(args=[self.instances, device, 'fdd-divide-mwkr'], plot=plot)
        else:
            assert False, 'Initial solution type = "plist", "spt", "fdd-divide-mwkr".'

        self.sub_graphs_mc = sub_graphs_mc
        self.current_graphs = current_graphs
        self.current_objs = make_span
        self.incumbent_objs = make_span
        self.itr = 0
        self.tabu_lists = [[] for _ in range(instances.shape[0])]
        feasible_actions, flag = self.feasible_actions(device)

        return (x, edge_indices_pc, edge_indices_mc, batch), feasible_actions, ~flag

    def feasible_actions(self, device):
        actions = []
        feasible_actions_flag = []  # False for no feasible operation pairs
        for i, (G, instance, tabu_list) in enumerate(zip(self.current_graphs, self.instances, self.tabu_lists)):
            action = self._gen_moves(solution=G, mch_mat=instance[1], tabu_list=tabu_list)
            # print(action)
            if len(action) != 0:
                actions.append(action)
                feasible_actions_flag.append(True)
            else:  # if no feasible actions available append dummy actions [0, 0]
                actions.append([[0, 0]])
                feasible_actions_flag.append(False)
        return actions, torch.tensor(feasible_actions_flag, device=device).unsqueeze(1)
    

class JsspWindow:
    def __init__(self, n_job, n_mch, low, high,
                 cp_solver_time=1,
                 cp_solver_cpu=1,
                 cpu_budget=16,
                 window_size=100,
                 log_path='jssp_window.log',
                 reward_type='yaoxin', fea_norm_const=1000, evaluator_type='message-passing'):
        self.n_job = n_job
        self.n_mch = n_mch
        self.n_oprs = self.n_job * self.n_mch
        self.low = low
        self.high = high
        self.itr = 0
        self.instances = None
        self.sub_graphs_mc = None
        self.current_graphs = None
        self.current_objs = None
        self.tabu_size = 1
        self.tabu_lists = None
        self.incumbent_objs = None
        self.reward_type = reward_type
        self.fea_norm_const = fea_norm_const
        self.evaluator_type = evaluator_type
        self.eva = Evaluator() if evaluator_type == 'message-passing' else CPM_batch_G
        self.adj_mat_pc = self._adj_mat_pc()
        self.cp_solver_time = cp_solver_time
        self.cp_solver_cpu = max(1, int(cp_solver_cpu))
        self.cpu_budget = max(1, int(cpu_budget))
        self.window_size = max(1, min(window_size, self.n_oprs))
        # Matrix view of the current solution.
        self.current_start_times = None
        self.current_orders = None
        self.current_adj_mats_mc = None
        self.current_adj_mats = None
        self.last_feasible_actions = None
        self.log_path = log_path
        self.logger = self._build_logger(log_path)

    def _build_logger(self, log_path):
        """Create one file logger per environment instance."""
        if log_path is None:
            return None
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(f'JsspWindow.{id(self)}')
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()
        handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(handler)
        return logger

    @staticmethod
    # This method is for logging 
    def _tensor_to_scalar(value):
        """Convert tensors/scalars to a plain Python float for logging."""
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0].item())
        return float(value)

    @staticmethod
    # This method is for logging
    def _normalize_action(action):
        """Convert one action to a JSON-friendly integer list."""
        return [int(action[0]), int(action[1])]

    # This method is for logging
    def _normalize_actions(self, actions):
        """Convert one sample's action space to indexed records for the logger."""
        return [
            {
                "action_idx": idx,
                "action": self._normalize_action(action),
            }
            for idx, action in enumerate(actions)
        ]
    
    # This method is for logging
    def _normalize_orders(self, orders):
        """Convert per-machine order lists into plain integers for logging."""
        normalized_orders = []
        for machine_ops in orders:
            normalized_machine_ops = []
            for op in machine_ops:
                if isinstance(op, (tuple, list, np.ndarray)):
                    normalized_machine_ops.append([int(value) for value in op])
                else:
                    normalized_machine_ops.append(int(op))
            normalized_orders.append(normalized_machine_ops)
        return normalized_orders

    # This method is for logging
    def _selected_action_index(self, feasible_actions, selected_action):
        """Find the selected action's index inside the current action set."""
        normalized_selected = self._normalize_action(selected_action)
        for idx, action in enumerate(feasible_actions):
            if self._normalize_action(action) == normalized_selected:
                return idx
        return -1

    def _write_log(self, payload):
        """Write one structured log line when file logging is enabled."""
        if self.logger is None:
            return
        self.logger.info(json.dumps(payload, ensure_ascii=False))

    def _log_reset_state(self, init_type):
        """Log each sample's initial order and makespan right after reset."""
        for sample_idx, (orders, makespan) in enumerate(zip(self.current_orders, self.current_objs)):
            self._write_log({
                "event": "reset",
                "init_type": init_type,
                "iteration": self.itr,
                "sample_idx": sample_idx,
                "order": self._normalize_orders(orders),
                "makespan": self._tensor_to_scalar(makespan),
            })

    def _log_step_actions(self, iteration, actions):
        """Log the action set and the chosen action for each sample before solving."""
        if self.last_feasible_actions is None:
            return
        for sample_idx, (action_space, selected_action) in enumerate(zip(self.last_feasible_actions, actions)):
            self._write_log({
                "event": "step_actions",
                "iteration": iteration,
                "sample_idx": sample_idx,
                "candidate_actions": self._normalize_actions(action_space),
                "selected_action_idx": self._selected_action_index(action_space, selected_action),
                "selected_action": self._normalize_action(selected_action),
            })

    def _log_step_state(self, iteration, makespan):
        """Log each sample's updated order and makespan after one environment step."""
        for sample_idx, (orders, sample_makespan) in enumerate(zip(self.current_orders, makespan)):
            self._write_log({
                "event": "step_result",
                "iteration": iteration,
                "sample_idx": sample_idx,
                "order": self._normalize_orders(orders),
                "makespan": self._tensor_to_scalar(sample_makespan),
            })

    # Create the conjunctions adjacent matrix
    # Return: [batch, n_oprs+2, n_oprs+2]
    def _adj_mat_pc(self):
        """Create the fixed precedence adjacency shared by all solutions."""
        adj_mat_pc = np.eye(self.n_oprs, k=-1, dtype=int)
        adj_mat_pc[np.arange(start=0, stop=self.n_oprs, step=1).reshape(self.n_job, -1)[:, 0]] = 0
        adj_mat_pc = np.pad(adj_mat_pc, 1, 'constant', constant_values=0)
        adj_mat_pc[[i for i in range(1, self.n_job * self.n_mch + 2 - 1, self.n_mch)], 0] = 1
        adj_mat_pc[-1, [i for i in range(self.n_mch, self.n_job * self.n_mch + 2 - 1, self.n_mch)]] = 1
        return np.transpose(adj_mat_pc)

    def show_state(self, G):
        """Plot one disjunctive graph for debugging."""
        x_axis = np.pad(np.tile(np.arange(1, self.n_mch + 1, 1), self.n_job), (1, 1), 'constant', constant_values=[0, self.n_mch + 1])
        y_axis = np.pad(np.arange(self.n_job, 0, -1).repeat(self.n_mch), (1, 1), 'constant', constant_values=np.median(np.arange(self.n_job, 0, -1)))
        pos = dict((n, (x, y)) for n, x, y in zip(G.nodes(), x_axis, y_axis))
        plt.figure(figsize=(15, 10))
        plt.tight_layout()
        nx.draw_networkx_edge_labels(G, pos=pos)
        nx.draw(G, pos=pos, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1')
        plt.show()

    def _p_list_solver(self, args, plot=False):
        """Build an initial schedule from a priority list."""
        instances, priority_lists, device = args[0], args[1], args[2]

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        current_graphs = []
        sub_graphs_mc = []
        for i, (instance, priority_list) in enumerate(zip(instances, priority_lists)):
            dur_mat, mch_mat = instance[0], instance[1]
            n_jobs = mch_mat.shape[0]
            n_machines = mch_mat.shape[1]
            n_operations = n_jobs * n_machines

            ops_mat = np.arange(0, n_operations).reshape(mch_mat.shape).tolist()
            list_for_latest_task_onMachine = [None] * n_machines
            adj_mat_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)
            for job_id in priority_list:
                op_id = ops_mat[job_id][0]
                m_id_for_action = mch_mat[op_id // n_machines, op_id % n_machines] - 1
                if list_for_latest_task_onMachine[m_id_for_action] is not None:
                    adj_mat_mc[op_id, list_for_latest_task_onMachine[m_id_for_action]] = 1
                list_for_latest_task_onMachine[m_id_for_action] = op_id
                ops_mat[job_id].pop(0)
            adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant', constant_values=0)
            adj_mat_mc = np.transpose(adj_mat_mc)

            adj_all = self.adj_mat_pc + adj_mat_mc
            dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(self.n_oprs + 2, axis=1)
            edge_weight = np.multiply(adj_all, dur_mat)
            G = nx.from_numpy_array(edge_weight, parallel_edges=False, create_using=nx.DiGraph)
            G.add_weighted_edges_from([(0, i, 0) for i in range(1, self.n_oprs + 2 - 1, self.n_mch)])
            current_graphs.append(G)
            G_mc = nx.from_numpy_array(adj_mat_mc, parallel_edges=False, create_using=nx.DiGraph)
            sub_graphs_mc.append(G_mc)

            if plot:
                self.show_state(G)

            edge_indices_pc.append((torch.nonzero(torch.from_numpy(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)
            durations.append(torch.from_numpy(dur_mat[:, 0]).to(device))

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)
        durations = torch.cat(durations, dim=0).reshape(-1, 1)
        if self.evaluator_type == 'message-passing':
            est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)
        else:
            est, lst, make_span = self.eva(current_graphs, dev=device)

        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        batch = torch.from_numpy(np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=self.n_job * self.n_mch + 2)).to(device)
        return (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span

    def _rules_solver(self, args, plot=False):
        """Build an initial schedule from a dispatching rule."""
        instances, device, rule_type = args[0], args[1], args[2]

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        current_graphs = []
        sub_graphs_mc = []
        for i, instance in enumerate(instances):
            dur_mat, mch_mat = instance[0], instance[1]
            n_jobs, n_machines = dur_mat.shape[0], dur_mat.shape[1]
            n_operations = n_jobs * n_machines
            last_col = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, -1]
            candidate_oprs = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, 0]
            mask = np.zeros(shape=n_jobs, dtype=bool)
            adj_mat_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)

            gant_chart = -self.high * np.ones_like(dur_mat.transpose(), dtype=np.int32)
            opIDsOnMchs = -n_jobs * np.ones_like(dur_mat.transpose(), dtype=np.int32)
            finished_mark = np.zeros_like(mch_mat, dtype=np.int32)

            for _ in range(n_operations):
                if rule_type == 'spt':
                    candidate_masked = candidate_oprs[np.where(~mask)]
                    dur_candidate = np.take(dur_mat, candidate_masked)
                    idx = np.random.choice(np.where(dur_candidate == np.min(dur_candidate))[0])
                    action = candidate_masked[idx]
                elif rule_type == 'fdd-divide-mwkr':
                    candidate_masked = candidate_oprs[np.where(~mask)]
                    fdd = np.take(np.cumsum(dur_mat, axis=1), candidate_masked)
                    wkr = np.take(np.cumsum(np.multiply(dur_mat, 1 - finished_mark), axis=1), last_col[np.where(~mask)])
                    priority = fdd / wkr
                    idx = np.random.choice(np.where(priority == np.min(priority))[0])
                    action = candidate_masked[idx]
                else:
                    action = None

                permissibleLeftShift(a=action, durMat=dur_mat, mchMat=mch_mat, mchsStartTimes=gant_chart, opIDsOnMchs=opIDsOnMchs)
                if action not in last_col:
                    candidate_oprs[action // n_machines] += 1
                else:
                    mask[action // n_machines] = 1
                finished_mark[action // n_machines, action % n_machines] = 1

            for _ in range(opIDsOnMchs.shape[1] - 1):
                adj_mat_mc[opIDsOnMchs[:, _ + 1], opIDsOnMchs[:, _]] = 1

            adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant', constant_values=0)
            adj_mat_mc = np.transpose(adj_mat_mc)
            adj_all = self.adj_mat_pc + adj_mat_mc
            dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(self.n_oprs + 2, axis=1)
            edge_weight = np.multiply(adj_all, dur_mat)
            G = nx.from_numpy_array(edge_weight, parallel_edges=False, create_using=nx.DiGraph)
            G.add_weighted_edges_from([(0, i, 0) for i in range(1, self.n_oprs + 2 - 1, self.n_mch)])
            current_graphs.append(G)
            G_mc = nx.from_numpy_array(adj_mat_mc, parallel_edges=False, create_using=nx.DiGraph)
            sub_graphs_mc.append(G_mc)

            if plot:
                self.show_state(G)

            edge_indices_pc.append((torch.nonzero(torch.from_numpy(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)
            durations.append(torch.from_numpy(dur_mat[:, 0]).to(device))

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)
        durations = torch.cat(durations, dim=0).reshape(-1, 1)
        if self.evaluator_type == 'message-passing':
            est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)
        else:
            est, lst, make_span = self.eva(current_graphs, dev=device)

        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        batch = torch.from_numpy(np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=self.n_job * self.n_mch + 2)).to(device)
        return (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span

    def dag2pyg(self, instances, nx_graphs, device):
        """Convert the current batch of graphs into the tensors consumed by the policy."""
        n_jobs, n_machines = instances[0][0].shape
        n_operations = n_jobs * n_machines

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        for i, (instance, G_mc) in enumerate(zip(instances, nx_graphs)):
            durations.append(np.pad(instance[0].reshape(-1), (1, 1), 'constant', constant_values=0))
            adj_mat_mc = nx.adjacency_matrix(G_mc, weight=None).todense()
            edge_indices_pc.append((torch.nonzero(torch.from_numpy(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)
        durations = torch.from_numpy(np.concatenate(durations)).reshape(-1, 1).to(device)
        if self.evaluator_type == 'message-passing':
            est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)
        else:
            est, lst, make_span = self.eva(self.current_graphs, dev=device)

        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        batch = torch.from_numpy(np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=n_jobs * n_machines + 2)).to(device)
        return x, edge_indices_pc, edge_indices_mc, batch, make_span

    def _job_op_to_node(self, job, op):
        return job * self.n_mch + op + 1

    def _node_to_job_op(self, node):
        op_id = node - 1
        return op_id // self.n_mch, op_id % self.n_mch

    @staticmethod
    def _machine_to_index(machine):
        return int(machine) - 1

    def _instance_to_cp(self, instance):
        return {
            "j": self.n_job,
            "m": self.n_mch,
            "duration": instance[0],
            "mch": instance[1] - 1,
        }

    def _extract_start_times(self, graph):
        """Decode operation start times from the weighted DAG."""
        topo_order = list(nx.topological_sort(graph))
        earliest_start = dict.fromkeys(graph.nodes, -float('inf'))
        earliest_start[topo_order[0]] = 0.0
        for node in topo_order:
            for succ in graph.successors(node):
                weight = graph.edges[node, succ]["weight"]
                candidate = earliest_start[node] + weight
                if candidate > earliest_start[succ]:
                    earliest_start[succ] = candidate

        start_times = np.zeros((self.n_job, self.n_mch), dtype=np.int32)
        for node in range(1, self.n_oprs + 1):
            job, op = self._node_to_job_op(node)
            start_times[job, op] = int(round(earliest_start[node]))
        return start_times

    def _build_orders_from_start_times(self, instance, start_times, ref_start_times=None):
        """Keep per-machine operation order as a matrix-friendly Python list."""
        return _window_build_orders(instance, self.n_job, self.n_mch, start_times, ref_start_times)

    def _schedule_from_orders(self, instance, orders):
        """Materialize a feasible start-time matrix from machine orders."""
        return _window_schedule_from_orders(instance, self.n_job, self.n_mch, orders)

    def _build_adj_mat_mc_from_orders(self, orders):
        """Build and keep the machine adjacency matrix for the current solution."""
        return _window_build_adj_mat_mc(self.n_job, self.n_mch, orders)

    def _build_graphs_from_adj_mat(self, instance, adj_mat_mc):
        """Create both NetworkX graphs from the saved machine adjacency matrix."""
        dur_mat = instance[0]
        adj_all = self.adj_mat_pc + adj_mat_mc
        dur_mat_aug = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(self.n_oprs + 2, axis=1)
        edge_weight = np.multiply(adj_all, dur_mat_aug)
        graph = nx.from_numpy_array(edge_weight, parallel_edges=False, create_using=nx.DiGraph)
        graph.add_weighted_edges_from([(0, i, 0) for i in range(1, self.n_oprs + 1, self.n_mch)])
        graph_mc = nx.from_numpy_array(adj_mat_mc, parallel_edges=False, create_using=nx.DiGraph)
        return graph, graph_mc

    def _build_graphs_from_orders(self, instance, orders):
        """Convenience wrapper when the caller starts from machine orders."""
        adj_mat_mc = self._build_adj_mat_mc_from_orders(orders)
        return self._build_graphs_from_adj_mat(instance, adj_mat_mc)

    def _plan_window(self, op_start_times, anchor_node):
        """Pick the operations belonging to one rolling window."""
        return _window_plan_ops(op_start_times, self.n_job, self.n_mch, self.window_size, anchor_node)

    def _get_machine_window_availability(self, ops_in_window, op_start_times, duration, mch):
        """Expose the window boundary calculation as a class method too."""
        return _window_machine_availability(ops_in_window, op_start_times, duration, mch, self.n_job, self.n_mch)

    def _solve_window_with_machine_avail(self, instance, ops_in_window, op_start_times, machine_avail):
        """Solve one window in the current process."""
        del machine_avail
        return _window_solve_subproblem(
            instance=instance,
            n_job=self.n_job,
            n_mch=self.n_mch,
            ops_in_window=ops_in_window,
            op_start_times=op_start_times,
            cp_solver_time=self.cp_solver_time,
            cp_solver_cpu=self.cp_solver_cpu,
        )

    def _apply_window_result(self, instance, base_start_times, window_ops, result):
        """Update matrix solution first, then rebuild machine orders."""
        candidate_start = np.array(base_start_times, copy=True)
        for (job, op), start_time in result.items():
            candidate_start[job][op] = int(start_time)
        orders = self._build_orders_from_start_times(instance, candidate_start, base_start_times)
        new_start_times = self._schedule_from_orders(instance, orders)
        return new_start_times, orders

    def _refresh_matrix_state(self):
        """Synchronize matrix caches from the graph view after reset or external changes."""
        self.current_start_times = [self._extract_start_times(graph) for graph in self.current_graphs]
        self.current_orders = [
            self._build_orders_from_start_times(instance, start_times)
            for instance, start_times in zip(self.instances, self.current_start_times)
        ]
        self.current_adj_mats_mc = [
            np.asarray(nx.adjacency_matrix(graph_mc, weight=None).todense(), dtype=np.int32)
            for graph_mc in self.sub_graphs_mc
        ]
        self.current_adj_mats = [self.adj_mat_pc + adj_mat_mc for adj_mat_mc in self.current_adj_mats_mc]

    def _parallel_config(self, task_count):
        """
        Decide how many batch items to solve in parallel and how many CP threads each item can use.

        When task_count > cpu_budget, the extra items are automatically queued by Pool.map,
        which gives the required serial/parallel hybrid behavior for large batches.
        """
        if task_count <= 0:
            return 0, self.cp_solver_cpu
        worker_count = min(task_count, self.cpu_budget)
        cpu_per_task = max(1, self.cpu_budget // worker_count)
        solver_cpu = min(self.cp_solver_cpu, cpu_per_task)
        return worker_count, solver_cpu

    def _merge_parallel_results(self, solved_payloads, plot=False):
        """Merge worker results back into graph and matrix caches in sample order."""
        for payload in sorted(solved_payloads, key=lambda item: item["index"]):
            if not payload["applied"]:
                continue

            idx = payload["index"]
            adj_mat_mc = payload["adj_mat_mc"]
            graph, graph_mc = self._build_graphs_from_adj_mat(self.instances[idx], adj_mat_mc)

            self.current_start_times[idx] = payload["start_times"]
            self.current_orders[idx] = payload["orders"]
            self.current_adj_mats_mc[idx] = adj_mat_mc
            self.current_adj_mats[idx] = self.adj_mat_pc + adj_mat_mc
            self.current_graphs[idx] = graph
            self.sub_graphs_mc[idx] = graph_mc

            if plot:
                self.show_state(graph)

    def _gen_moves(self, solution, mch_mat, tabu_list=None):
        del mch_mat
        critical_path = nx.dag_longest_path(solution)[1:-1]
        return self._get_ones(critical_path, tabu_list)

    @staticmethod
    def _get_ones(cb_op, tabu_list=None):
        tabu_list = tabu_list or []
        return [[node, node] for node in cb_op if [node, node] not in tabu_list]

    def change_nxgraph_topology(self, actions, plot=False):
        """
        Solve one window for each batch item.

        The batch dimension is handled with a bounded process pool:
        - at most `cpu_budget` batch items run in parallel;
        - if batch size is larger, remaining items are solved serially in queued waves;
        - worker outputs are merged back by original sample index.
        """
        if self.current_start_times is None or self.current_orders is None or self.current_adj_mats_mc is None:
            self._refresh_matrix_state()

        tasks = [
            (
                idx,
                instance,
                np.array(self.current_start_times[idx], copy=True),
                action,
                self.n_job,
                self.n_mch,
                self.window_size,
                self.cp_solver_time,
                1,  # one sample occupies one CPU while batching across the environment
            )
            for idx, (instance, action) in enumerate(zip(self.instances, actions))
        ]

        worker_count, solver_cpu = self._parallel_config(len(tasks))
        tasks = [task[:-1] + (solver_cpu,) for task in tasks]
        if worker_count <= 1:
            solved_payloads = [_window_solve_single(task) for task in tasks]
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=worker_count) as pool:
                solved_payloads = pool.map(_window_solve_single, tasks)

        self._merge_parallel_results(solved_payloads, plot=plot)

    def step(self, actions, device, plot=False):
        """Apply a batch of actions, update graphs, and return the next state."""
        iteration = self.itr
        self._log_step_actions(iteration, actions)
        self.change_nxgraph_topology(actions, plot)
        x, edge_indices_pc, edge_indices_mc, batch, makespan = self.dag2pyg(self.instances, self.sub_graphs_mc, device)
        if self.reward_type == 'consecutive':
            reward = self.current_objs - makespan
        elif self.reward_type == 'yaoxin':
            reward = torch.where(
                self.incumbent_objs - makespan > 0,
                self.incumbent_objs - makespan,
                torch.tensor(0, dtype=torch.float32, device=device),
            )
        else:
            raise ValueError('reward type must be "yaoxin" or "consecutive".')

        self.incumbent_objs = torch.where(makespan - self.incumbent_objs < 0, makespan, self.incumbent_objs)
        self.current_objs = makespan

        if self.tabu_size != 0:
            action_reversed = [a[::-1] for a in actions]
            for i, action in enumerate(action_reversed):
                if action == [0, 0]:
                    continue
                if len(self.tabu_lists[i]) == self.tabu_size:
                    self.tabu_lists[i].pop(0)
                self.tabu_lists[i].append(action)

        self._log_step_state(iteration, makespan)
        self.itr += 1
        feasible_actions, flag = self.feasible_actions(device)
        self.last_feasible_actions = feasible_actions
        return (x, edge_indices_pc, edge_indices_mc, batch), reward, feasible_actions, ~flag

    def reset(self, instances, init_type, device, plot=False):
        """Initialize one batch of instances and construct both graph and matrix views."""
        self.instances = instances
        if init_type == 'plist':
            plist = np.repeat(
                np.random.permutation(np.arange(self.n_job).repeat(self.n_mch)).reshape(1, -1),
                repeats=self.instances.shape[0],
                axis=0,
            )
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span = self._p_list_solver(
                args=[self.instances, plist, device], plot=plot
            )
        elif init_type == 'spt':
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span = self._rules_solver(
                args=[self.instances, device, 'spt'], plot=plot
            )
        elif init_type == 'fdd-divide-mwkr':
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span = self._rules_solver(
                args=[self.instances, device, 'fdd-divide-mwkr'], plot=plot
            )
        else:
            raise AssertionError('Initial solution type = "plist", "spt", "fdd-divide-mwkr".')

        self.sub_graphs_mc = sub_graphs_mc
        self.current_graphs = current_graphs
        self.current_objs = make_span
        self.incumbent_objs = make_span
        self.itr = 0
        self.tabu_lists = [[] for _ in range(instances.shape[0])]
        self._refresh_matrix_state()
        feasible_actions, flag = self.feasible_actions(device)
        self.last_feasible_actions = feasible_actions
        self._log_reset_state(init_type)
        return (x, edge_indices_pc, edge_indices_mc, batch), feasible_actions, ~flag

    def feasible_actions(self, device):
        """Enumerate one candidate action list for each batch item."""
        actions = []
        feasible_actions_flag = []
        for G, instance, tabu_list in zip(self.current_graphs, self.instances, self.tabu_lists):
            action = self._gen_moves(solution=G, mch_mat=instance[1], tabu_list=tabu_list)
            if len(action) != 0:
                actions.append(action)
                feasible_actions_flag.append(True)
            else:
                actions.append([[0, 0]])
                feasible_actions_flag.append(False)
        return actions, torch.tensor(feasible_actions_flag, device=device).unsqueeze(1)

if __name__ == '__main__':
    from generateJSP import uni_instance_gen
    def jobshop_with_maintenance(jobs_data = None):
    # Create the model.
        model = cp_model.CpModel()

        if jobs_data is None or jobs_data.size == 0:
            jobs_data = [  # task = (machine_id, processing_time).
                [(0, 3), (1, 2), (2, 2)],  # Job0
                [(0, 2), (2, 1), (1, 4)],  # Job1
                [(1, 4), (2, 3), (0, 2)],  # Job2
            ]

        machines_count = 1 + max(task[0] for job in jobs_data for task in job)
        all_machines = range(machines_count)

        # Computes horizon dynamically as the sum of all durations.
        horizon = sum(task[1] for job in jobs_data for task in job)

        # Named tuple to store information about created variables.
        task_type = collections.namedtuple("task_type", "start end interval")
        # Named tuple to manipulate solution information.
        assigned_task_type = collections.namedtuple(
            "assigned_task_type", "start job index duration"
        )

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        for job_id, job in enumerate(jobs_data):
            for entry in enumerate(job):
                task_id, task = entry
                machine, duration = task
                suffix = f"_{job_id}_{task_id}"
                start_var = model.NewIntVar(0, horizon, "start" + suffix)
                end_var = model.NewIntVar(0, horizon, "end" + suffix)
                interval_var = model.NewIntervalVar(
                    start_var, duration, end_var, "interval" + suffix
                )
                all_tasks[job_id, task_id] = task_type(
                    start=start_var, end=end_var, interval=interval_var
                )
                machine_to_intervals[machine].append(interval_var)

        # Add maintenance interval (machine 0 is not available on time {4, 5, 6, 7}).
        # machine_to_intervals[0].append(model.NewIntervalVar(4, 4, 8, "weekend_0"))

        # Create and add disjunctive constraints.
        for machine in all_machines:
            model.AddNoOverlap(machine_to_intervals[machine])

        # Precedences inside a job.
        for job_id, job in enumerate(jobs_data):
            for task_id in range(len(job) - 1):
                model.Add(
                    all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
                )

        # Makespan objective.
        obj_var = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(
            obj_var,
            [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],
        )
        model.Minimize(obj_var)

        # Solve model.
        solver = cp_model.CpSolver()
        # Set the maximum Time
        solver.parameters.num_workers = 1
        solver.parameters.max_memory_in_mb = 16 * 1024
        solver.parameters.max_time_in_seconds = 10
        # solver.parameters.use_parallel_search = True
        # solver.parameters.log_search_progress = True
        # solution_printer = SolutionPrinter()
        # status = solver.Solve(model, solution_printer)
        status = solver.Solve(model)

        # Output solution.
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Create one list of assigned tasks per machine.
            assigned_jobs = collections.defaultdict(list)
            for job_id, job in enumerate(jobs_data):
                for task_id, task in enumerate(job):
                    machine = task[0]
                    assigned_jobs[machine].append(
                        assigned_task_type(
                            start=solver.Value(all_tasks[job_id, task_id].start),
                            job=job_id,
                            index=task_id,
                            duration=task[1],
                        )
                    )

            # Create per machine output lines.
            output = ""
            machine_orders = []
            for machine in all_machines:
                # Sort by starting time.
                assigned_jobs[machine].sort()
                sol_line_tasks = "Machine " + str(machine) + ": "
                sol_line = "           "
                machine_order = []

                for assigned_task in assigned_jobs[machine]:
                    name = f"job_{assigned_task.job}_{assigned_task.index}"
                    # add spaces to output to align columns.
                    sol_line_tasks += f"{name:>10}"
                    start = assigned_task.start
                    duration = assigned_task.duration

                    sol_tmp = f"[{start}, {start + duration}]"
                    # add spaces to output to align columns.
                    sol_line += f"{sol_tmp:>10}"

                    operation_id = assigned_task.job * machines_count + assigned_task.index
                    machine_order.append(operation_id)

                sol_line += "\n"
                sol_line_tasks += "\n"
                output += sol_line_tasks
                output += sol_line
                machine_orders.append(machine_order)

            # Finally print the solution found.
            # print(f"Optimal Schedule Length: {solver.ObjectiveValue()}")
            # print(output)
            # print(solver.ResponseStats())

            return machine_orders, solver.ObjectiveValue()

        else :
            raise TimeoutError(f"Time {solver.parameters.max_time_in_seconds} "
                            f"Not Enough For Generating Solutions")
    n_job, n_mch = 10, 10
    low, high = 1, 100
    num_instances = 1
    env = JsspWindow(n_job, n_mch, low, high, cp_solver_time=1, cp_solver_cpu=1, cpu_budget=8, window_size=70)
    instances = [uni_instance_gen(n_job, n_mch, low, high) for _ in range(num_instances)]
    for instance in instances:
        job_data = np.stack((instance[1], instance[0]), axis=-1)
        output, makespan = jobshop_with_maintenance(job_data)
        print(f"Instance makespan: {makespan}")
    instances = np.array(instances)
    states, feasible_actions, done = env.reset(instances, init_type='spt', device='cpu', plot=False)
    # print(states[0].shape, len(feasible_actions), done.shape)
    for _ in range(10):
        actions = [action[0] for action in feasible_actions]
        states, reward, feasible_actions, done = env.step(actions, device='cpu', plot=False)
        # print(states[0].shape, reward.shape, len(feasible_actions), done.shape) 
