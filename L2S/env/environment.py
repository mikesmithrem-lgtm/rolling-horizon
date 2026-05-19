import collections
import os
import sys
import multiprocessing as mp


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import networkx as nx
from ortools.sat.python import cp_model
from env.generateJSP import uni_instance_gen
from env.message_passing_evl import Evaluator, CPM_batch_G
import matplotlib.pyplot as plt
import time
import random
from env.window_utils import (
    _available_nonzero_cpus,
    _build_action_window_features,
    _init_worker_nonzero_cpu_affinity,
    _temporary_nonzero_cpu_affinity,
    _window_build_orders,
    _window_solve_single,
)
from model.actor import Actor


class BatchGraph:
    def __init__(self):
        self.x = None
        self.edge_index_pc = None
        self.edge_index_mc = None
        self.batch = None
        self.feasible_action_machine_feat = None

    def wrapper(self,
                x,
                edge_index_pc,
                edge_index_mc,
                batch,
                feasible_action_machine_feat=None):
        self.x = x
        self.edge_index_pc = edge_index_pc
        self.edge_index_mc = edge_index_mc
        self.batch = batch
        self.feasible_action_machine_feat = feasible_action_machine_feat

    def clean(self):
        self.x = None
        self.edge_index_pc = None
        self.edge_index_mc = None
        self.batch = None
        self.feasible_action_machine_feat = None


def _apply_zero_improvement_penalty(reward, zero_improvement_penalty):
    """Penalize no-improvement transitions with a fixed reward."""
    zero_reward_mask = torch.isclose(reward, torch.zeros_like(reward))
    if not torch.any(zero_reward_mask):
        return reward
    penalty = torch.full_like(reward, fill_value=zero_improvement_penalty)
    return torch.where(zero_reward_mask, penalty, reward)

class JsspWindow:
    def __init__(self, n_job, n_mch, low, high,
                 cp_solver_time=1,
                 cp_solver_cpu=1,
                 cpu_budget=16,
                 window_size=150,
                 zero_improvement_penalty=-3.0,
                 fea_norm_const=1000,
                 evaluator_type='message-passing'):
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
        self.zero_improvement_penalty = float(zero_improvement_penalty)
        self.fea_norm_const = fea_norm_const
        self.evaluator_type = evaluator_type
        self.eva = Evaluator() if evaluator_type == 'message-passing' else CPM_batch_G
        self.adj_mat_pc = self._adj_mat_pc()
        self.cp_solver_time = cp_solver_time
        self.cp_solver_cpu = max(1, int(cp_solver_cpu))
        self.cpu_budget = max(1, int(cpu_budget))
        self.available_cpus = _available_nonzero_cpus()
        self.window_size = max(1, min(window_size, self.n_oprs))
        # Matrix view of the current solution.
        self.current_start_times = None
        self.current_orders = None
        self.current_adj_mats_mc = None
        self.current_adj_mats = None
        self.last_feasible_actions = None

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

    @staticmethod
    def _spt_pdr_orders(dur_mat, mch_mat, eps=1e-3, tau=1e-4):
        """Rebuild per-machine operation orders with the same logic as pdrs.SPT."""
        n_jobs, n_machines = dur_mat.shape
        ops = np.array([(0, mch_mat[j, 0] - 1) for j in range(n_jobs)], dtype=np.int64)

        tie = 0.0
        prio = np.zeros(n_jobs, dtype=np.float64)
        for job in range(n_jobs):
            tie += tau
            prio[job] = -float(dur_mat[job, 0]) - tie

        curr_time = -1
        machine_ready = [0 for _ in range(n_machines)]
        orders_by_machine = [[] for _ in range(n_machines)]
        end_times_by_job = [[] for _ in range(n_jobs)]

        active_job = n_jobs
        while active_job > 0:
            job_order = np.argsort(prio)[::-1][:active_job]
            curr_time = min(ct for ct in machine_ready if ct > curr_time)

            for job in job_order:
                op_idx, machine = int(ops[job, 0]), int(ops[job, 1])
                min_st = max(machine_ready[machine], 0 if op_idx == 0 else end_times_by_job[job][-1])
                if min_st - eps < curr_time:
                    if op_idx < n_machines - 1:
                        ops[job, 0] = op_idx + 1
                        ops[job, 1] = int(mch_mat[job, op_idx + 1] - 1)
                        tie += tau
                        prio[job] = -float(dur_mat[job, op_idx + 1]) - tie
                    else:
                        active_job -= 1
                        prio[job] = -float('inf')

                    machine_ready[machine] = int(min_st + dur_mat[job, op_idx])
                    end_times_by_job[job].append(machine_ready[machine])
                    orders_by_machine[machine].append(job * n_machines + op_idx)

        op_ids_on_mchs = -n_jobs * np.ones((n_machines, n_jobs), dtype=np.int32)
        for machine, machine_ops in enumerate(orders_by_machine):
            op_ids_on_mchs[machine, :len(machine_ops)] = machine_ops
        return op_ids_on_mchs


    def _rules_solver(self, args, plot=False):
        """Build an initial schedule from a dispatching rule."""
        instances, device, rule_type = args[0], args[1], args[2]

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        current_graphs = []
        sub_graphs_mc = []
        for i, instance in enumerate(instances):
            # [n_j, n_m]
            dur_mat, mch_mat = instance[0], instance[1]
            n_jobs, n_machines = dur_mat.shape[0], dur_mat.shape[1]
            n_operations = n_jobs * n_machines
            # last_col: [n_j,], the last operation of each job
            last_col = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, -1]
            # candidate_oprs: [n_j,], the first operation of each job
            candidate_oprs = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, 0]
            # mask: [n_j,], true if a job finished
            mask = np.zeros(shape=n_jobs, dtype=bool)
            # adj_mat_mc: [n_opr, n_opr], the disjunctive precedence
            adj_mat_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)

            # [n_m, n_j] 
            # gant_chart: equal to op_start_times
            # opIDsOnMchs: equal to orders
            if rule_type == 'spt-pdr':
                opIDsOnMchs = self._spt_pdr_orders(dur_mat, mch_mat)
            else:
                gant_chart = -self.high * np.ones_like(dur_mat.transpose(), dtype=np.int32)
                opIDsOnMchs = -self.n_job * np.ones_like(dur_mat.transpose(), dtype=np.int32)
                finished_mark = np.zeros_like(mch_mat, dtype=np.int32)

                for _ in range(n_operations):
                    if rule_type == 'spt':
                        # candidate_masked offers idx, and dur_candidate implies the duration.
                        # shaped [n_job, ]
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
                        raise ValueError(f'Unsupported rule type: {rule_type}')

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
            job, op = divmod(node - 1, self.n_mch)
            start_times[job, op] = int(round(earliest_start[node]))
        return start_times

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

    def _build_action_window_features(self, instance, start_times, actions, device):
        return _build_action_window_features(
            instance=instance,
            start_times=start_times,
            actions=actions,
            n_job=self.n_job,
            n_mch=self.n_mch,
            window_size=self.window_size,
            fea_norm_const=self.fea_norm_const,
            device=device,
        )

    def _refresh_matrix_state(self):
        """Synchronize matrix caches from the graph view after reset or external changes."""
        self.current_start_times = [self._extract_start_times(graph) for graph in self.current_graphs]
        self.current_orders = [
            _window_build_orders(instance, self.n_job, self.n_mch, start_times)
            for instance, start_times in zip(self.instances, self.current_start_times)
        ]
        self.current_adj_mats_mc = [
            np.asarray(nx.adjacency_matrix(graph_mc, weight=None).todense(), dtype=np.int32)
            for graph_mc in self.sub_graphs_mc
        ]
        self.current_adj_mats = [self.adj_mat_pc + adj_mat_mc for adj_mat_mc in self.current_adj_mats_mc]

    def _pack_state(self, x, edge_indices_pc, edge_indices_mc, batch, action_machine_feat, action_window_scalar):
        return (
            x,
            edge_indices_pc,
            edge_indices_mc,
            batch,
            action_machine_feat,
            action_window_scalar,
        )

    def _parallel_config(self, task_count):
        """
        Decide how many batch items to solve in parallel and how many CP threads each item can use.

        When task_count > cpu_budget, the extra items are automatically queued by Pool.map,
        which gives the required serial/parallel hybrid behavior for large batches.
        """
        if task_count <= 0:
            return 0, self.cp_solver_cpu
        available_cpu_count = len(self.available_cpus) if self.available_cpus is not None else self.cpu_budget
        worker_count = min(task_count, self.cpu_budget, max(1, available_cpu_count))
        cpu_per_task = max(1, self.cpu_budget // worker_count)
        solver_cpu = min(self.cp_solver_cpu, cpu_per_task, max(1, available_cpu_count))
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

    def _gen_moves(self, solution, tabu_list=None):
        critical_path = nx.dag_longest_path(solution)[1:-1]
        return self._get_ones(critical_path, tabu_list)

    @staticmethod
    def _get_ones(cb_op, tabu_list=None):
        tabu_list = tabu_list or []
        return [node for node in cb_op if node not in tabu_list]

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
            with ctx.Pool(processes=worker_count, initializer=_init_worker_nonzero_cpu_affinity) as pool:
                solved_payloads = pool.map(_window_solve_single, tasks)
        self._merge_parallel_results(solved_payloads, plot=plot)

    def step(self, actions, device, plot=False):
        """Apply a batch of actions, update graphs, and return the next state."""
        self.change_nxgraph_topology(actions, plot)
        x, edge_indices_pc, edge_indices_mc, batch, makespan = self.dag2pyg(self.instances, self.sub_graphs_mc, device)
        reward = self.current_objs - makespan
        self.current_objs = makespan

        if self.tabu_size != 0:
            for i, action in enumerate(actions):
                if action == 0:
                    continue
                if len(self.tabu_lists[i]) == self.tabu_size:
                    self.tabu_lists[i].pop(0)
                self.tabu_lists[i].append(action)

        self.itr += 1
        feasible_actions, flag, action_machine_feat = self.feasible_actions(device)
        self.last_feasible_actions = feasible_actions

        return self._pack_state(
            x,
            edge_indices_pc,
            edge_indices_mc,
            batch,
            action_machine_feat,
        ), reward, feasible_actions, ~flag

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
        elif init_type == 'spt-pdr':
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span = self._rules_solver(
                args=[self.instances, device, 'spt-pdr'], plot=plot
            )
        elif init_type == 'fdd-divide-mwkr':
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span = self._rules_solver(
                args=[self.instances, device, 'fdd-divide-mwkr'], plot=plot
            )
        else:
            raise AssertionError('Initial solution type = "plist", "spt", "spt-pdr", "fdd-divide-mwkr".')

        self.sub_graphs_mc = sub_graphs_mc
        self.current_graphs = current_graphs
        self.current_objs = make_span
        self.itr = 0
        self.tabu_lists = [[] for _ in range(instances.shape[0])]
        self._refresh_matrix_state()

        feasible_actions, flag, action_machine_feat = self.feasible_actions(device)
        self.last_feasible_actions = feasible_actions

        return self._pack_state(
            x,
            edge_indices_pc,
            edge_indices_mc,
            batch,
            action_machine_feat,
        ), feasible_actions, ~flag

    def feasible_actions(self, device):
        """
        Enumerate one candidate action list for each batch item and
        build window features aligned with those actions.

        Returns
        -------
        actions : list[list[int]]
        feasible_actions_flag : torch.BoolTensor, shape [B, 1]
        action_machine_feat : list[torch.Tensor]
            each tensor has shape [A, n_mch, 9]
        action_window_scalar : list[torch.Tensor]
            each tensor has shape [A, 18]
        """
        actions = []
        action_machine_feat = []
        feasible_actions_flag = []

        for G, instance, tabu_list, start_times in zip(
                self.current_graphs,
                self.instances,
                self.tabu_lists,
                self.current_start_times):

            action = self._gen_moves(solution=G, tabu_list=tabu_list)

            if len(action) != 0:
                actions.append(action)
                feasible_actions_flag.append(True)
            else:
                actions.append([0])
                feasible_actions_flag.append(False)

            machine_feat = self._build_action_window_features(
                instance=instance,
                start_times=start_times,
                actions=actions[-1],
                device=device,
            )

            action_machine_feat.append(machine_feat)

        feasible_actions_flag = torch.tensor(
            feasible_actions_flag,
            dtype=torch.bool,
            device=device
        ).unsqueeze(1)

        return actions, feasible_actions_flag, action_machine_feat

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
        with _temporary_nonzero_cpu_affinity():
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

                    operation_id = assigned_task.job * machines_count + assigned_task.index + 1
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
    low, high = 1, 99
    num_instances = 4
    env = JsspWindow(n_job, n_mch, low, high, cp_solver_time=1, cp_solver_cpu=1, cpu_budget=1, window_size=60)
    instances = [uni_instance_gen(n_job, n_mch, low, high) for _ in range(num_instances)]
    for instance in instances:
        job_data = np.stack((instance[1], instance[0]), axis=-1)
        output, makespan = jobshop_with_maintenance(job_data)
        print(f"Instance makespan: {makespan}")
    instances = np.array(instances)
    states, feasible_actions, done = env.reset(instances, init_type='spt-pdr', device='cpu', plot=False)
    for _ in range(50):
        actions = [random.choice(action) for action in feasible_actions]
        states, reward, feasible_actions, done = env.step(actions, device='cpu', plot=False)
