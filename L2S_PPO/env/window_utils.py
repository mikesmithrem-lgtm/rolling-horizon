import os
from contextlib import contextmanager

import numpy as np
import torch
from ortools.sat.python import cp_model


def _available_nonzero_cpus():
    """Return the current process affinity set without CPU0 when possible."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    current_cpus = set(os.sched_getaffinity(0))
    nonzero_cpus = {cpu for cpu in current_cpus if cpu != 0}
    return nonzero_cpus if nonzero_cpus else current_cpus


@contextmanager
def _temporary_nonzero_cpu_affinity():
    """Temporarily pin the current process away from CPU0 during CP solving."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        yield
        return

    original_cpus = set(os.sched_getaffinity(0))
    target_cpus = _available_nonzero_cpus()
    if not target_cpus or target_cpus == original_cpus:
        yield
        return

    os.sched_setaffinity(0, target_cpus)
    try:
        yield
    finally:
        os.sched_setaffinity(0, original_cpus)


def _init_worker_nonzero_cpu_affinity():
    """Bind worker processes to non-zero CPUs only."""
    if not hasattr(os, "sched_setaffinity"):
        return
    target_cpus = _available_nonzero_cpus()
    if target_cpus:
        os.sched_setaffinity(0, target_cpus)


def _node_to_job_op(node, n_mch):
    """Map 1-based operation node id to (job, op)."""
    return divmod(int(node) - 1, n_mch)


def _window_plan_nodes(op_start_times, 
                       n_job, n_mch, window_size, 
                       anchor_node, sorted_nodes=None, index_by_node=None):
    """Build a rolling window around the selected anchor node."""
    if sorted_nodes is None or index_by_node is None:
        sorted_nodes = list(range(1, n_job * n_mch + 1))
        sorted_nodes.sort(
            key=lambda node: (
                op_start_times[(node - 1) // n_mch][(node - 1) % n_mch],
                node,
            )
        )
        index_by_node = {node: idx for idx, node in enumerate(sorted_nodes)}

    window_idx = index_by_node[anchor_node]
    n_oprs = n_job * n_mch
    if window_idx > n_oprs - window_size:
        window_idx = n_oprs - window_size
    return sorted_nodes[window_idx:window_idx + window_size]


def _current_schedule_stats(instance, start_times, n_job, n_mch):
    """
    Compute schedule-level statistics from current start times.
    Returns:
        op_end_times: [n_job, n_mch]
        makespan: scalar
        machine_end: [n_mch]
        job_end: [n_job]
    """
    duration = instance[0]
    mch = instance[1] - 1

    op_end_times = np.asarray(start_times, dtype=np.int32) + np.asarray(duration, dtype=np.int32)
    makespan = int(op_end_times[:, -1].max())
    machine_end = np.zeros(n_mch, dtype=np.int32)

    for j in range(n_job):
        for o in range(n_mch):
            m = int(mch[j][o])
            machine_end[m] = max(machine_end[m], int(op_end_times[j][o]))

    job_end = op_end_times[:, -1].astype(np.int32)
    return op_end_times, makespan, machine_end, job_end


def _window_machine_availability(nodes_in_window, op_start_times, duration, mch, n_job, n_mch):
    """
    Original machine availability used by CP subproblem:
    return machine_avail[machine_idx] = (earliest, latest)
    """
    machine_avail = {}
    makespan = max(op_start_times[job][n_mch - 1] + duration[job][n_mch - 1] for job in range(n_job))
    window_node_set = set(nodes_in_window)

    for machine_idx in {
        int(mch[(node - 1) // n_mch][(node - 1) % n_mch]) - 1 for node in nodes_in_window
    }:
        machine_nodes = [
            job * n_mch + op + 1
            for job in range(n_job)
            for op in range(n_mch)
            if int(mch[job][op]) - 1 == machine_idx
        ]
        machine_nodes.sort(
            key=lambda node: (
                op_start_times[(node - 1) // n_mch][(node - 1) % n_mch],
                node,
            )
        )
        idxs = [idx for idx, node in enumerate(machine_nodes) if node in window_node_set]
        if not idxs:
            continue

        first_idx = idxs[0]
        last_idx = idxs[-1]

        if first_idx > 0:
            prev_job, prev_op = divmod(machine_nodes[first_idx - 1] - 1, n_mch)
            earliest = int(op_start_times[prev_job][prev_op] + duration[prev_job][prev_op])
        else:
            earliest = 0

        if last_idx < len(machine_nodes) - 1:
            next_job, next_op = divmod(machine_nodes[last_idx + 1] - 1, n_mch)
            latest = int(op_start_times[next_job][next_op])
        else:
            latest = int(makespan)

        machine_avail[machine_idx] = (earliest, latest)

    return machine_avail


def _window_solve_subproblem(instance, n_job, n_mch, nodes_in_window, op_start_times, cp_solver_time, cp_solver_cpu):
    """Solve one window with CP-SAT while keeping external machine/job boundaries fixed."""
    duration = instance[0]
    mch = instance[1] - 1
    machine_avail = _window_machine_availability(nodes_in_window, op_start_times, duration, instance[1], n_job, n_mch)
    horizon = max(op_start_times[job][n_mch - 1] + duration[job][n_mch - 1] for job in range(n_job))
    model = cp_model.CpModel()
    op_vars = {}
    window_node_set = set(nodes_in_window)

    for node in nodes_in_window:
        job, op = divmod(node - 1, n_mch)
        machine = int(mch[job][op])
        lb, latest_bound = machine_avail[machine]
        ub = latest_bound - int(duration[job][op])

        if op > 0 and node - 1 not in window_node_set:
            prev_end = int(op_start_times[job][op - 1] + duration[job][op - 1])
            lb = max(lb, prev_end)

        if op < n_mch - 1 and node + 1 not in window_node_set:
            after_start = int(op_start_times[job][op + 1])
            ub = min(ub, after_start - int(duration[job][op]))

        if lb > ub:
            return None

        op_vars[node] = model.NewIntVar(lb, ub, f"start_{node}")

    for node in nodes_in_window:
        job, op = divmod(node - 1, n_mch)
        if op > 0 and node - 1 in window_node_set:
            model.Add(op_vars[node] >= op_vars[node - 1] + int(duration[job][op - 1]))

    machine_to_nodes = {}
    interval_vars = {}
    for node in nodes_in_window:
        job, op = divmod(node - 1, n_mch)
        machine = int(mch[job][op])
        machine_to_nodes.setdefault(machine, []).append(node)
        interval_vars[node] = model.NewIntervalVar(
            op_vars[node],
            int(duration[job][op]),
            op_vars[node] + int(duration[job][op]),
            f"interval_{node}",
        )

    for machine_nodes in machine_to_nodes.values():
        model.AddNoOverlap([interval_vars[node] for node in machine_nodes])

    makespan = model.NewIntVar(0, int(horizon), "makespan")
    for node in nodes_in_window:
        job, op = divmod(node - 1, n_mch)
        model.Add(makespan >= op_vars[node] + int(duration[job][op]))
    model.Minimize(makespan)

    with _temporary_nonzero_cpu_affinity():
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(cp_solver_time)
        solver.parameters.num_search_workers = int(cp_solver_cpu)
        solver.parameters.random_seed = 0
        status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return {node: solver.Value(op_vars[node]) for node in nodes_in_window}


def _window_build_orders(instance, n_job, n_mch, start_times, ref_start_times=None):
    """Convert matrix start times into per-machine operation orders."""
    if ref_start_times is None:
        ref_start_times = start_times
    orders = [[] for _ in range(n_mch)]
    for machine_idx in range(n_mch):
        machine = machine_idx + 1
        machine_nodes = [
            job * n_mch + op + 1
            for job in range(n_job)
            for op in range(n_mch)
            if int(instance[1][job][op]) == machine
        ]
        machine_nodes.sort(
            key=lambda node: (
                start_times[(node - 1) // n_mch][(node - 1) % n_mch],
                ref_start_times[(node - 1) // n_mch][(node - 1) % n_mch],
                node,
            )
        )
        orders[machine_idx] = machine_nodes
    return orders


def _window_schedule_from_orders(instance, n_job, n_mch, orders):
    """Rebuild a globally feasible schedule from machine orders."""
    duration = instance[0]
    op_start_times = np.zeros((n_job, n_mch), dtype=np.int32)
    machine_ready = [0] * n_mch
    job_ready = [0] * n_job
    order_idx = [0] * n_mch
    scheduled_nodes = set()

    while len(scheduled_nodes) < n_job * n_mch:
        progress = False
        for machine_idx in range(n_mch):
            if order_idx[machine_idx] >= len(orders[machine_idx]):
                continue

            node = orders[machine_idx][order_idx[machine_idx]]
            if node in scheduled_nodes:
                order_idx[machine_idx] += 1
                continue

            job, op = divmod(node - 1, n_mch)
            if op > 0 and node - 1 not in scheduled_nodes:
                continue

            start_time = max(job_ready[job], machine_ready[machine_idx])
            op_start_times[job][op] = start_time
            end_time = start_time + int(duration[job][op])
            job_ready[job] = end_time
            machine_ready[machine_idx] = end_time
            scheduled_nodes.add(node)
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
            adj_mat_mc[curr - 1, prev - 1] = 1
    adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), "constant", constant_values=0)
    return np.transpose(adj_mat_mc)


def _compute_suffix_matrices(instance, op_end_times, job_end, machine_end, n_job, n_mch):
    """Compute schedule-based suffix statistics for each operation."""
    mch = np.asarray(instance[1], dtype=np.int32) - 1
    job_suffix = np.zeros((n_job, n_mch), dtype=np.float32)
    machine_suffix = np.zeros((n_job, n_mch), dtype=np.float32)

    for j in range(n_job):
        for o in range(n_mch):
            m = int(mch[j, o])
            job_suffix[j, o] = float(max(int(job_end[j]) - int(op_end_times[j, o]), 0))
            machine_suffix[j, o] = float(max(int(machine_end[m]) - int(op_end_times[j, o]), 0))

    return job_suffix, machine_suffix


def _build_local_pc_edges(nodes_in_window, n_mch, local_idx_by_node, device):
    """Build local precedence edges among window operations only."""
    edges = []
    window_node_set = set(nodes_in_window)

    for node in nodes_in_window:
        job, op = _node_to_job_op(node, n_mch)
        if op > 0 and (node - 1) in window_node_set:
            edges.append([local_idx_by_node[node - 1], local_idx_by_node[node]])

    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


def _build_local_mc_edges(nodes_in_window, adj_mat_mc, device):
    """
    Build local machine-order edges from the augmented global machine adjacency matrix.

    adj_mat_mc is assumed to be the augmented matrix with indices aligned to node ids:
    row = source node id, col = target node id.
    """
    if len(nodes_in_window) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    nodes = np.asarray(nodes_in_window, dtype=np.int64)
    sub_adj = np.asarray(adj_mat_mc[np.ix_(nodes, nodes)], dtype=np.int32)
    rows, cols = np.nonzero(sub_adj)

    if len(rows) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_index = np.stack([rows, cols], axis=0)
    return torch.tensor(edge_index, dtype=torch.long, device=device).contiguous()


def _extract_window_machine_info(
    nodes_in_window,
    orders,
    start_times,
    duration,
    mch,
    machine_end,
    makespan,
    n_mch,
):
    """
    Build machine-level context for the current window.

    Returns
    -------
    involved_machines : list[int]
        global machine ids in ascending order
    machine_feat : np.ndarray [Mw, 8]
    node_to_local_machine : dict[node] -> local machine index
    pos_on_machine_ratio : dict[node] -> float
    """
    window_node_set = set(nodes_in_window)
    total_window_ops = max(len(nodes_in_window), 1)

    involved_machines = sorted({
        int(mch[(node - 1) // n_mch, (node - 1) % n_mch])
        for node in nodes_in_window
    })

    global_m_to_local = {m: idx for idx, m in enumerate(involved_machines)}
    node_to_local_machine = {}
    pos_on_machine_ratio = {}
    machine_feat = np.zeros((len(involved_machines), 8), dtype=np.float32)

    for global_m in involved_machines:
        local_m = global_m_to_local[global_m]
        machine_nodes_full = orders[global_m]
        idxs = [idx for idx, node in enumerate(machine_nodes_full) if node in window_node_set]
        machine_window_nodes = [machine_nodes_full[idx] for idx in idxs]

        if len(machine_window_nodes) == 0:
            continue

        first_idx = idxs[0]
        last_idx = idxs[-1]

        if first_idx > 0:
            prev_node = machine_nodes_full[first_idx - 1]
            pj, po = _node_to_job_op(prev_node, n_mch)
            avail_start = int(start_times[pj, po] + duration[pj, po])
        else:
            avail_start = 0

        if last_idx < len(machine_nodes_full) - 1:
            next_node = machine_nodes_full[last_idx + 1]
            nj, no = _node_to_job_op(next_node, n_mch)
            avail_end = int(start_times[nj, no])
        else:
            avail_end = int(makespan)

        avail_span = max(avail_end - avail_start, 0)
        window_load = float(sum(duration[_node_to_job_op(node, n_mch)] for node in machine_window_nodes))
        idle = max(float(avail_span) - window_load, 0.0)
        load_ratio = (window_load / float(avail_span)) if avail_span > 0 else 0.0
        idle_ratio = (idle / float(avail_span)) if avail_span > 0 else 0.0
        suffix_load = max(float(machine_end[global_m] - avail_end), 0.0)
        num_window_ops_ratio = float(len(machine_window_nodes)) / float(total_window_ops)

        machine_feat[local_m, 0] = float(avail_start)
        machine_feat[local_m, 1] = float(avail_end)
        machine_feat[local_m, 2] = float(avail_span)
        machine_feat[local_m, 3] = float(window_load)
        machine_feat[local_m, 4] = float(load_ratio)
        machine_feat[local_m, 5] = float(idle_ratio)
        machine_feat[local_m, 6] = float(suffix_load)
        machine_feat[local_m, 7] = float(num_window_ops_ratio)

        denom = max(len(machine_window_nodes) - 1, 1)
        for pos, node in enumerate(machine_window_nodes):
            node_to_local_machine[node] = local_m
            pos_on_machine_ratio[node] = float(pos) / float(denom) if len(machine_window_nodes) > 1 else 0.0

    return involved_machines, machine_feat, node_to_local_machine, pos_on_machine_ratio


def _build_op_features(
    nodes_in_window,
    anchor_node,
    instance,
    start_times,
    op_end_times,
    makespan,
    job_suffix,
    machine_suffix,
    pos_on_machine_ratio,
    n_job,
    n_mch,
):
    """
    Build operation features.

    op_features shape = [K, 10]
    dims:
        0: proc_time / C
        1: start_time / C
        2: end_time / C
        3: (start_time - window_start) / window_span
        4: (end_time - window_start) / window_span
        5: job_suffix / C
        6: machine_suffix / C
        7: is_anchor
        8: pos_in_job / max(n_mch - 1, 1)
        9: pos_on_machine_in_window / max(num_ops_on_machine - 1, 1)
    """
    duration = np.asarray(instance[0], dtype=np.int32)
    time_norm = max(float(makespan), 1.0)

    starts = [int(start_times[(node - 1) // n_mch, (node - 1) % n_mch]) for node in nodes_in_window]
    ends = [int(op_end_times[(node - 1) // n_mch, (node - 1) % n_mch]) for node in nodes_in_window]
    window_start = min(starts) if len(starts) > 0 else 0
    window_end = max(ends) if len(ends) > 0 else window_start
    window_span = max(float(window_end - window_start), 1.0)

    op_feat = np.zeros((len(nodes_in_window), 10), dtype=np.float32)

    for idx, node in enumerate(nodes_in_window):
        j, o = _node_to_job_op(node, n_mch)
        st = float(start_times[j, o])
        et = float(op_end_times[j, o])
        pt = float(duration[j, o])

        op_feat[idx, 0] = pt / time_norm
        op_feat[idx, 1] = st / time_norm
        op_feat[idx, 2] = et / time_norm
        op_feat[idx, 3] = (st - float(window_start)) / window_span
        op_feat[idx, 4] = (et - float(window_start)) / window_span
        op_feat[idx, 5] = float(job_suffix[j, o]) / time_norm
        op_feat[idx, 6] = float(machine_suffix[j, o]) / time_norm
        op_feat[idx, 7] = 1.0 if int(node) == int(anchor_node) else 0.0
        op_feat[idx, 8] = float(o) / float(max(n_mch - 1, 1))
        op_feat[idx, 9] = float(pos_on_machine_ratio.get(node, 0.0))

    return op_feat


def _normalize_machine_features(machine_feat, makespan):
    """Normalize machine features to the actor-ready 8D tensor."""
    time_norm = max(float(makespan), 1.0)
    out = np.array(machine_feat, dtype=np.float32, copy=True)

    if out.size == 0:
        return out

    out[:, 0] /= time_norm
    out[:, 1] /= time_norm
    out[:, 2] /= time_norm
    out[:, 3] /= time_norm
    out[:, 6] /= time_norm
    return out


def _build_action_window_states(
    instance,
    start_times,
    orders,
    adj_mat_mc,
    actions,
    n_job,
    n_mch,
    window_size,
    fea_norm_const,
    device,
):
    """
    Build window states for all feasible actions.

    Parameters
    ----------
    instance : tuple/list
        instance[0] = duration matrix [n_job, n_mch]
        instance[1] = machine matrix [n_job, n_mch], 1-based machine ids
    start_times : np.ndarray [n_job, n_mch]
    orders : list[list[int]]
        current per-machine operation orders
    adj_mat_mc : np.ndarray [n_oprs+2, n_oprs+2]
        augmented machine adjacency matrix
    actions : list[int]
        candidate anchor nodes
    device : torch.device

    Returns
    -------
    window_states : list[dict]
        each dict contains:
            action: int
            anchor_local_idx: int
            op_ids: LongTensor[K]
            op_features: FloatTensor[K, 10]
            mch_features: FloatTensor[Mw, 8]
            op_machine_id: LongTensor[K]
            edge_index_pc: LongTensor[2, E1]
            edge_index_mc: LongTensor[2, E2]
    """
    _ = fea_norm_const  # kept for interface compatibility

    if len(actions) == 0:
        return []

    duration = np.asarray(instance[0], dtype=np.int32)
    mch = np.asarray(instance[1], dtype=np.int32) - 1
    start_times = np.asarray(start_times, dtype=np.int32)

    op_end_times, makespan, machine_end, job_end = _current_schedule_stats(
        instance, start_times, n_job, n_mch
    )
    job_suffix, machine_suffix = _compute_suffix_matrices(
        instance, op_end_times, job_end, machine_end, n_job, n_mch
    )

    sorted_nodes = list(range(1, n_job * n_mch + 1))
    sorted_nodes.sort(
        key=lambda node: (
            int(start_times[(node - 1) // n_mch, (node - 1) % n_mch]),
            int(node),
        )
    )
    index_by_node = {node: idx for idx, node in enumerate(sorted_nodes)}

    window_states = []

    for action in actions:
        anchor_node = int(action)
        if anchor_node <= 0 or anchor_node > n_job * n_mch:
            continue

        nodes_in_window = _window_plan_nodes(
            start_times,
            n_job,
            n_mch,
            window_size,
            anchor_node,
            sorted_nodes=sorted_nodes,
            index_by_node=index_by_node,
        )

        if len(nodes_in_window) == 0:
            continue

        local_idx_by_node = {node: idx for idx, node in enumerate(nodes_in_window)}

        _, machine_feat_raw, node_to_local_machine, pos_on_machine_ratio = _extract_window_machine_info(
            nodes_in_window=nodes_in_window,
            orders=orders,
            start_times=start_times,
            duration=duration,
            mch=mch,
            machine_end=machine_end,
            makespan=makespan,
            n_mch=n_mch,
        )

        machine_feat = _normalize_machine_features(machine_feat_raw, makespan)
        op_feat = _build_op_features(
            nodes_in_window=nodes_in_window,
            anchor_node=anchor_node,
            instance=instance,
            start_times=start_times,
            op_end_times=op_end_times,
            makespan=makespan,
            job_suffix=job_suffix,
            machine_suffix=machine_suffix,
            pos_on_machine_ratio=pos_on_machine_ratio,
            n_job=n_job,
            n_mch=n_mch,
        )

        op_machine_id = np.array(
            [node_to_local_machine[node] for node in nodes_in_window],
            dtype=np.int64,
        )

        edge_index_pc = _build_local_pc_edges(
            nodes_in_window=nodes_in_window,
            n_mch=n_mch,
            local_idx_by_node=local_idx_by_node,
            device=device,
        )
        edge_index_mc = _build_local_mc_edges(
            nodes_in_window=nodes_in_window,
            adj_mat_mc=adj_mat_mc,
            device=device,
        )

        window_state = {
            "action": anchor_node,
            "anchor_local_idx": int(local_idx_by_node[anchor_node]),
            "op_ids": torch.tensor(nodes_in_window, dtype=torch.long, device=device),
            "op_features": torch.tensor(op_feat, dtype=torch.float, device=device),
            "mch_features": torch.tensor(machine_feat, dtype=torch.float, device=device),
            "op_machine_id": torch.tensor(op_machine_id, dtype=torch.long, device=device),
            "edge_index_pc": edge_index_pc,
            "edge_index_mc": edge_index_mc,
        }
        assert window_state["op_ids"][window_state["anchor_local_idx"]].item() == action
        window_states.append(window_state)

    return window_states


def _window_solve_single(args):
    """Worker entry for one batch item; returns only serializable matrix data."""
    idx, instance, base_start_times, action, n_job, n_mch, window_size, cp_solver_time, cp_solver_cpu = args
    anchor_node = int(action)
    if anchor_node <= 0 or anchor_node > n_job * n_mch:
        return {"index": idx, "applied": False}

    window_nodes = _window_plan_nodes(base_start_times, n_job, n_mch, window_size, anchor_node)
    result = _window_solve_subproblem(
        instance=instance,
        n_job=n_job,
        n_mch=n_mch,
        nodes_in_window=window_nodes,
        op_start_times=base_start_times,
        cp_solver_time=cp_solver_time,
        cp_solver_cpu=cp_solver_cpu,
    )
    if result is None:
        return {"index": idx, "applied": False}

    candidate_start = np.array(base_start_times, copy=True)
    for node, start_time in result.items():
        job, op = divmod(node - 1, n_mch)
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