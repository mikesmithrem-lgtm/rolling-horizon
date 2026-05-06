import argparse
import heapq
import io
import os
import random
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from time import time

import numpy as np

from pdrs import FDDDivideMWKR, PDR, solve_instance


_CP_MODEL_IMPORT_ERROR = None


def _get_cp_model():
    global _CP_MODEL_IMPORT_ERROR
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from ortools.sat.python import cp_model
    except ImportError as exc:
        _CP_MODEL_IMPORT_ERROR = exc
        return None
    return cp_model


class JSPNumpyDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = sorted(
            file_name
            for file_name in os.listdir(data_dir)
            if file_name.endswith(".jsp")
        )

    def __len__(self):
        return len(self.data_files)

    def __iter__(self):
        for idx in range(len(self.data_files)):
            yield self[idx]

    def _parse_sample(self, idx):
        file_name = self.data_files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        n_jobs, n_mchs = map(int, lines[0].split())
        times = []
        machines = []
        for row_idx in range(1, n_jobs + 1):
            row = list(map(int, lines[row_idx].split()))
            machines.append(row[0::2])
            times.append(row[1::2])

        makespan = -1
        orders = []
        if len(lines) > n_jobs + 1:
            makespan = int(lines[n_jobs + 1])
            for row_idx in range(n_jobs + 2, len(lines)):
                orders.append(list(map(int, lines[row_idx].split())))

        return n_jobs, n_mchs, times, machines, makespan, orders, file_name

    def __getitem__(self, idx):
        n_jobs, n_mchs, times, machines, makespan, orders, file_name = self._parse_sample(idx)
        return {
            "j": n_jobs,
            "m": n_mchs,
            "duration": times,
            "mch": machines,
            "makespan": makespan,
            "orders": orders,
            "names": file_name,
        }


class ObjMeter:
    def __init__(self, name="makespan"):
        self.sum = {}
        self.list = {}
        self.count = {}
        self.meter = name

    def update(self, ins, val):
        shape = f"{ins['j']}x{ins['m']}"
        if shape not in self.sum:
            self.sum[shape] = val
            self.list[shape] = [val]
            self.count[shape] = 1
        else:
            self.sum[shape] += val
            self.list[shape].append(val)
            self.count[shape] += 1

    def __str__(self):
        out = ""
        for shape in sorted(self.sum):
            val = self.sum[shape] / self.count[shape]
            out += f"\t\t\t{shape}: AVG {self.meter}={val:4.3f}\n"
        return out[:-1]

    @property
    def avg(self):
        return sum(self.sum.values()) / sum(self.count.values()) if self.count else 0


def convert_solution_to_start_times(sol, jsp_instance):
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]

    op_start_times = [[0] * m for _ in range(j)]
    machine_ready = [0] * m
    scheduled_set = set()
    schedule_index = [0] * m

    while len(scheduled_set) < j * m:
        progress = False
        for machine in range(m):
            cur_op_index = schedule_index[machine]
            if cur_op_index >= len(sol[machine]):
                continue

            op_idx = sol[machine][cur_op_index]
            job, op = op_idx // m, op_idx % m
            if op > 0 and (job, op - 1) not in scheduled_set:
                continue

            prev_end = 0
            if op > 0:
                prev_end = op_start_times[job][op - 1] + duration[job][op - 1]

            op_start_times[job][op] = max(prev_end, machine_ready[machine])
            machine_ready[machine] = op_start_times[job][op] + duration[job][op]
            scheduled_set.add((job, op))
            schedule_index[machine] += 1
            progress = True

        if not progress:
            raise RuntimeError("convert_solution_to_start_times stuck: invalid machine order")

    return op_start_times


def get_machine_orders_from_start_times(op_start_times, jsp_instance):
    j, m = jsp_instance["j"], jsp_instance["m"]
    mch = jsp_instance["mch"]
    machine_orders = [[] for _ in range(m)]
    for job in range(j):
        for op in range(m):
            machine = mch[job][op]
            machine_orders[machine].append((job, op))

    for machine in range(m):
        machine_orders[machine].sort(key=lambda item: op_start_times[item[0]][item[1]])
    return machine_orders


def schedule_from_machine_orders(jsp_instance, machine_orders):
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]

    op_start_times = [[0] * m for _ in range(j)]
    job_ready = [0] * j
    machine_ready = [0] * m
    machine_idx = [0] * m
    scheduled = set()
    total_ops = j * m

    while len(scheduled) < total_ops:
        progress = False
        for machine in range(m):
            idx = machine_idx[machine]
            if idx >= len(machine_orders[machine]):
                continue

            job, op = machine_orders[machine][idx]
            if op > 0 and (job, op - 1) not in scheduled:
                continue

            start_time = max(job_ready[job], machine_ready[machine])
            op_start_times[job][op] = start_time
            end_time = start_time + duration[job][op]
            job_ready[job] = end_time
            machine_ready[machine] = end_time
            scheduled.add((job, op))
            machine_idx[machine] += 1
            progress = True

        if not progress:
            return None, None

    return op_start_times, max(job_ready)


def _copy_int_matrix(matrix, rows, cols):
    return [[int(matrix[row][col]) for col in range(cols)] for row in range(rows)]


def _normalize_jsp_instance(jsp_instance, est):
    j, m = jsp_instance["j"], jsp_instance["m"]
    normalized = dict(jsp_instance)
    normalized["duration"] = _copy_int_matrix(jsp_instance["duration"], j, m)
    normalized["mch"] = _copy_int_matrix(jsp_instance["mch"], j, m)
    normalized_est = _copy_int_matrix(est, j, m)
    return normalized, normalized_est


def compute_makespan(op_start_times, jsp_instance):
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    return max(op_start_times[job][m - 1] + duration[job][m - 1] for job in range(j))


def choose_bottleneck_machine(op_start_times, jsp_instance):
    return rank_bottleneck_machines(op_start_times, jsp_instance)[0]


def rank_bottleneck_machines(op_start_times, jsp_instance):
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]

    latest_completion = [-1] * m
    critical_load = [0] * m
    makespan = compute_makespan(op_start_times, jsp_instance)

    for job in range(j):
        for op in range(m):
            machine = mch[job][op]
            end_time = op_start_times[job][op] + duration[job][op]
            latest_completion[machine] = max(latest_completion[machine], end_time)
            if end_time == makespan:
                critical_load[machine] += duration[job][op]

    return sorted(
        range(m),
        key=lambda machine: (latest_completion[machine], critical_load[machine], machine),
        reverse=True,
    )


def _build_fixed_dag(machine_orders, jsp_instance, excluded_machine):
    j, m = jsp_instance["j"], jsp_instance["m"]
    mch = jsp_instance["mch"]

    nodes = [(job, op) for job in range(j) for op in range(m)]
    succs = {node: [] for node in nodes}
    preds = {node: [] for node in nodes}

    for job in range(j):
        for op in range(1, m):
            prev_node = (job, op - 1)
            node = (job, op)
            succs[prev_node].append(node)
            preds[node].append(prev_node)

    for machine in range(m):
        if machine == excluded_machine:
            continue
        order = machine_orders[machine]
        for prev_node, node in zip(order, order[1:]):
            succs[prev_node].append(node)
            preds[node].append(prev_node)

    indegree = {node: len(preds[node]) for node in nodes}
    queue = deque(node for node in nodes if indegree[node] == 0)
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in succs[node]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                queue.append(succ)

    if len(topo_order) != len(nodes):
        raise RuntimeError("Fixed-machine DAG contains a cycle.")
    return topo_order, succs


def compute_heads_and_tails(jsp_instance, machine_orders, target_machine):
    duration = jsp_instance["duration"]
    topo_order, succs = _build_fixed_dag(machine_orders, jsp_instance, target_machine)
    release = {node: 0 for node in topo_order}

    for node in topo_order:
        node_duration = duration[node[0]][node[1]]
        completion = release[node] + node_duration
        for succ in succs[node]:
            if completion > release[succ]:
                release[succ] = completion

    tail = {node: 0 for node in topo_order}
    for node in reversed(topo_order):
        best = 0
        for succ in succs[node]:
            candidate = duration[succ[0]][succ[1]] + tail[succ]
            if candidate > best:
                best = candidate
        tail[node] = best

    machine_ops = list(machine_orders[target_machine])
    return (
        {op: release[op] for op in machine_ops},
        {op: tail[op] for op in machine_ops},
    )


def schrage_sequence(machine_ops, release, tail, duration):
    pending = sorted(machine_ops, key=lambda op: (release[op], -tail[op], op))
    ready = []
    seq = []
    idx = 0
    current_time = min((release[op] for op in machine_ops), default=0)

    while idx < len(pending) or ready:
        while idx < len(pending) and release[pending[idx]] <= current_time:
            op = pending[idx]
            heapq.heappush(ready, (-tail[op], release[op], duration[op[0]][op[1]], op))
            idx += 1

        if not ready:
            current_time = max(current_time, release[pending[idx]])
            continue

        _, _, proc_time, op = heapq.heappop(ready)
        seq.append(op)
        current_time += proc_time

    return seq


def build_candidate_from_sequence(jsp_instance, machine_orders, machine, sequence, method_name):
    new_orders = [order[:] for order in machine_orders]
    new_orders[machine] = list(sequence)
    cand_start, cand_makespan = schedule_from_machine_orders(jsp_instance, new_orders)
    if cand_start is None:
        return None
    return {
        "method": method_name,
        "machine": machine,
        "machine_orders": new_orders,
        "start_times": cand_start,
        "makespan": cand_makespan,
    }


def solve_single_machine_cp(machine_ops, release, tail, duration, current_makespan, time_limit):
    cp_model = _get_cp_model()
    if cp_model is None:
        return None

    if not machine_ops:
        return []

    total_proc = sum(duration[job][op] for job, op in machine_ops)
    horizon = current_makespan + total_proc
    model = cp_model.CpModel()

    start_vars = {}
    end_vars = {}
    interval_vars = {}
    for job, op in machine_ops:
        proc = duration[job][op]
        lb = release[(job, op)]
        ub = horizon - proc
        start = model.NewIntVar(lb, ub, f"st_{job}_{op}")
        end = model.NewIntVar(lb + proc, horizon, f"ed_{job}_{op}")
        interval = model.NewIntervalVar(start, proc, end, f"int_{job}_{op}")
        start_vars[(job, op)] = start
        end_vars[(job, op)] = end
        interval_vars[(job, op)] = interval

    model.AddNoOverlap([interval_vars[op] for op in machine_ops])

    objective = model.NewIntVar(0, horizon + max(tail.values(), default=0), "theta")
    for op in machine_ops:
        model.Add(objective >= end_vars[op] + tail[op])
    model.Minimize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    return sorted(
        machine_ops,
        key=lambda op: (solver.Value(start_vars[op]), -tail[op], op),
    )


def optimize_machine_by_schrage(jsp_instance, machine_orders, machine, current_makespan):
    duration = jsp_instance["duration"]
    machine_ops = machine_orders[machine]
    release, tail = compute_heads_and_tails(jsp_instance, machine_orders, machine)
    sequence = schrage_sequence(machine_ops, release, tail, duration)
    return build_candidate_from_sequence(
        jsp_instance,
        machine_orders,
        machine,
        sequence,
        "schrage-head-tail",
    )


def optimize_machine_by_cp(jsp_instance, machine_orders, machine, current_makespan, time_limit=0.2):
    duration = jsp_instance["duration"]
    machine_ops = machine_orders[machine]
    release, tail = compute_heads_and_tails(jsp_instance, machine_orders, machine)
    sequence = solve_single_machine_cp(
        machine_ops,
        release,
        tail,
        duration,
        current_makespan,
        time_limit,
    )
    if sequence is None:
        return None
    return build_candidate_from_sequence(
        jsp_instance,
        machine_orders,
        machine,
        sequence,
        "cp-single-machine",
    )


def optimize_machine_by_insertion(jsp_instance, machine_orders, machine):
    order = machine_orders[machine]
    if len(order) <= 1:
        return None

    best_candidate = None
    for from_idx in range(len(order)):
        for to_idx in range(len(order)):
            if from_idx == to_idx:
                continue
            new_sequence = order[:]
            op = new_sequence.pop(from_idx)
            new_sequence.insert(to_idx, op)
            candidate = build_candidate_from_sequence(
                jsp_instance,
                machine_orders,
                machine,
                new_sequence,
                "insertion-neighborhood",
            )
            if candidate is None:
                continue
            if best_candidate is None or candidate["makespan"] < best_candidate["makespan"]:
                best_candidate = candidate

    return best_candidate


def shifting_bottleneck_search(
    jsp_instance,
    est,
    max_iterations=50,
    cp_time_limit=0.2,
    methods=("schrage", "cp", "insertion"),
    debug=False,
):
    jsp_instance, est = _normalize_jsp_instance(jsp_instance, est)
    current_start = [row[:] for row in est]
    current_makespan = compute_makespan(current_start, jsp_instance)
    best_start = [row[:] for row in current_start]
    best_makespan = current_makespan

    for iteration in range(1, max_iterations + 1):
        machine_orders = get_machine_orders_from_start_times(current_start, jsp_instance)
        candidates = []
        machine_ranking = rank_bottleneck_machines(current_start, jsp_instance)

        for bottleneck_machine in machine_ranking:
            if "schrage" in methods:
                candidate = optimize_machine_by_schrage(
                    jsp_instance,
                    machine_orders,
                    bottleneck_machine,
                    current_makespan,
                )
                if candidate is not None:
                    candidates.append(candidate)

            if "cp" in methods:
                candidate = optimize_machine_by_cp(
                    jsp_instance,
                    machine_orders,
                    bottleneck_machine,
                    current_makespan,
                    time_limit=cp_time_limit,
                )
                if candidate is not None:
                    candidates.append(candidate)

            if "insertion" in methods:
                candidate = optimize_machine_by_insertion(
                    jsp_instance,
                    machine_orders,
                    bottleneck_machine,
                )
                if candidate is not None:
                    candidates.append(candidate)

        if not candidates:
            break

        machine_rank = {machine: idx for idx, machine in enumerate(machine_ranking)}
        next_candidate = min(
            candidates,
            key=lambda item: (item["makespan"], machine_rank[item["machine"]]),
        )
        if next_candidate["makespan"] >= current_makespan:
            break

        current_start = [row[:] for row in next_candidate["start_times"]]
        current_makespan = next_candidate["makespan"]
        if current_makespan < best_makespan:
            best_start = [row[:] for row in current_start]
            best_makespan = current_makespan

        if debug:
            print(
                f"[SB] iter={iteration}, machine={next_candidate['machine']}, "
                f"method={next_candidate['method']}, makespan={current_makespan}"
            )

    return best_start, best_makespan


def build_dataset(args):
    if args.validation_npy and os.path.exists(args.validation_npy):
        test_dataset = np.load(args.validation_npy, allow_pickle=True)
        instance = [test_dataset[i] for i in range(test_dataset.shape[0])]
        return [
            {
                "names": f"validation_instance_20x15_{idx}",
                "j": 20,
                "m": 15,
                "duration": ins[0],
                "mch": ins[1] - 1,
            }
            for idx, ins in enumerate(instance)
        ]
    return JSPNumpyDataset(data_dir=args.data_dir)


def main():
    parser = argparse.ArgumentParser(description="JSSP shifting bottleneck local search")
    parser.add_argument("--data-dir", default="./benchmark/TA")
    parser.add_argument(
        "--validation-npy",
        default="./L2S/validation_data/validation_instance_20x15[1,99].npy",
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--cp-time-limit", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    cp_available = _get_cp_model() is not None
    methods = ("schrage", "cp", "insertion") if cp_available else ("schrage", "insertion")
    if not cp_available:
        print(f"CP single-machine model disabled: {_CP_MODEL_IMPORT_ERROR}")

    dataset = build_dataset(args)
    gaps = ObjMeter()
    better_gaps = ObjMeter()
    pdr = PDR(priority=FDDDivideMWKR())
    st = time()

    if args.end is None:
        args.end = len(dataset)

    for idx, jsp_dataset in enumerate(dataset, start=1):
        if idx < args.start or idx > args.end:
            continue

        sols, ms, _ = solve_instance(jsp_dataset, pdr=pdr)
        op_start_times = convert_solution_to_start_times(sols, jsp_dataset)
        init_makespan = compute_makespan(op_start_times, _normalize_jsp_instance(jsp_dataset, op_start_times)[0])
        if ms != init_makespan:
            raise AssertionError(
                f"Initial makespan mismatch for {jsp_dataset['names']}: {ms} vs {init_makespan}"
            )

        print(f"Initial solution for {jsp_dataset['names']} has makespan {ms}")
        best_start_times, best_makespan = shifting_bottleneck_search(
            jsp_dataset,
            op_start_times,
            max_iterations=args.max_iterations,
            cp_time_limit=args.cp_time_limit,
            methods=methods,
            debug=args.debug,
        )

        gap = ms
        better_gap = best_makespan
        print(jsp_dataset["names"], f" Gap: {gap:.2f}", f" Better Gap: {better_gap:.2f}")
        gaps.update(jsp_dataset, gap)
        better_gaps.update(jsp_dataset, better_gap)

    print("Overall Gaps:", gaps.avg)
    print(gaps)
    print("Overall Better Gaps:", better_gaps.avg)
    print(better_gaps)
    print(f"Total time: {time() - st:.2f} seconds")


if __name__ == "__main__":
    main()
