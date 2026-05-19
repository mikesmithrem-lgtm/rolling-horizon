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


def _window_plan_nodes(op_start_times, n_job, n_mch, window_size, anchor_node, sorted_nodes=None, index_by_node=None):
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


def _window_machine_context(nodes_in_window, op_start_times, duration, mch, n_job, n_mch, machine_end, makespan):
    """
    Rich machine-level context for action/window features.

    Returns:
        machine_ctx[m] = {
            "earliest": int,
            "latest": int,
            "span": int,
            "load": float,
            "slack": float,
            "load_ratio": float,
            "suffix": float,
            "op_count": int,
            "has_after": bool,
        }
    """
    machine_ctx = {}
    window_node_set = set(nodes_in_window)
    total_window_ops = len(nodes_in_window)

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
        machine_window_nodes = [machine_nodes[idx] for idx in idxs]

        if first_idx > 0:
            prev_job, prev_op = divmod(machine_nodes[first_idx - 1] - 1, n_mch)
            earliest = int(op_start_times[prev_job][prev_op] + duration[prev_job][prev_op])
        else:
            earliest = 0

        if last_idx < len(machine_nodes) - 1:
            next_job, next_op = divmod(machine_nodes[last_idx + 1] - 1, n_mch)
            latest = int(op_start_times[next_job][next_op])
            has_after = True
        else:
            latest = int(makespan)
            has_after = False

        span = max(int(latest - earliest), 0)
        load = float(sum(duration[(node - 1) // n_mch][(node - 1) % n_mch] for node in machine_window_nodes))
        slack = max(float(span) - load, 0.0)
        load_ratio = (load / float(span)) if span > 0 else 0.0
        suffix = max(float(machine_end[machine_idx] - latest), 0.0)

        machine_ctx[machine_idx] = {
            "earliest": earliest,
            "latest": latest,
            "span": span,
            "load": load,
            "slack": slack,
            "load_ratio": load_ratio,
            "suffix": suffix,
            "op_count": len(machine_window_nodes),
            "has_after": has_after,
            "op_ratio": float(len(machine_window_nodes)) / max(float(total_window_ops), 1.0),
        }

    return machine_ctx


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


def _machine_ctx_to_tensor(machine_ctx, n_mch, makespan, device):
    """
    Convert machine context to tensor [n_mch, 9]:
        0: avail_start / C
        1: avail_end / C
        2: avail_span / C
        3: window_load / C
        4: machine_slack / C
        5: load_ratio
        6: machine_suffix / C
        7: op_ratio
        8: valid_mask
    """
    denom = max(float(makespan), 1.0)
    machine_tensor = torch.zeros((n_mch, 9), dtype=torch.float, device=device)

    for machine_idx, ctx in machine_ctx.items():
        machine_tensor[machine_idx, 0] = float(ctx["earliest"]) / denom
        machine_tensor[machine_idx, 1] = float(ctx["latest"]) / denom
        machine_tensor[machine_idx, 2] = float(ctx["span"]) / denom
        machine_tensor[machine_idx, 3] = float(ctx["load"]) / denom
        machine_tensor[machine_idx, 4] = float(ctx["slack"]) / denom
        machine_tensor[machine_idx, 5] = float(ctx["load_ratio"])
        machine_tensor[machine_idx, 6] = float(ctx["suffix"]) / denom
        machine_tensor[machine_idx, 7] = float(ctx["op_ratio"])
        machine_tensor[machine_idx, 8] = 1.0

    return machine_tensor


def _build_action_window_states(instance, 
                                  start_times, 
                                  actions, 
                                  n_job, 
                                  n_mch, 
                                  window_size,
                                  fea_norm_const, 
                                  device):
    """
    Build rich window features for each feasible action.

    Returns:
        action_machine_feat: [A, n_mch, 9]
        action_window_scalar: [A, 18]
    """
    if len(actions) == 0:
        return (
            torch.zeros((0, n_mch, 9), dtype=torch.float, device=device),
            torch.zeros((0, 18), dtype=torch.float, device=device),
        )

    sorted_nodes = list(range(1, n_job * n_mch + 1))
    sorted_nodes.sort(
        key=lambda node: (
            start_times[(node - 1) // n_mch][(node - 1) % n_mch],
            node,
        )
    )
    index_by_node = {node: idx for idx, node in enumerate(sorted_nodes)}

    duration = instance[0]
    mch = instance[1]
    op_end_times, makespan, machine_end, job_end = _current_schedule_stats(
        instance, start_times, n_job, n_mch
    )

    action_machine_feat = []
    action_window_scalar = []

    for action in actions:
        anchor_node = int(action)

        if anchor_node <= 0 or anchor_node > n_job * n_mch:
            action_machine_feat.append(torch.zeros((n_mch, 9), dtype=torch.float, device=device))
            action_window_scalar.append(torch.zeros((18,), dtype=torch.float, device=device))
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

        machine_ctx = _window_machine_context(
            nodes_in_window=nodes_in_window,
            op_start_times=start_times,
            duration=duration,
            mch=mch,
            n_job=n_job,
            n_mch=n_mch,
            machine_end=machine_end,
            makespan=makespan,
        )

        machine_tensor = _machine_ctx_to_tensor(
            machine_ctx=machine_ctx,
            n_mch=n_mch,
            makespan=makespan,
            device=device,
        )

        action_machine_feat.append(machine_tensor)

    return torch.stack(action_machine_feat, dim=0)


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