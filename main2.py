import os
import numpy as np
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from dataset import JSPNumpyDataset, ObjMeter
import random
import multiprocessing
from collections import defaultdict
from time import time
SEED = 0

def priority_dispatch_rule(jsp_instance, rule="spt"):
    """
    最简单的优先级调度算法（PDR），支持SPT/MOR等规则。
    rule: "spt"（最短处理时间优先），"mor"（机器顺序），"lpt"（最长处理时间优先）
    返回: op_start_times, makespan
    """
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]

    # 每个工件的下一个可调度操作
    next_op = [0] * j
    # 每台机器的最早可用时间
    machine_ready = [0] * m
    # 每个工件的最早可用时间
    job_ready = [0] * j
    # 记录每个操作的开始时间
    op_start_times = [[0] * m for _ in range(j)]
    total_ops = j * m
    scheduled = 0

    while scheduled < total_ops:
        candidates = []
        for job in range(j):
            op = next_op[job]
            if op < m:
                machine = mch[job][op]
                dur = duration[job][op]
                ready_time = max(job_ready[job], machine_ready[machine])
                # 优先级规则
                if rule == "spt":
                    priority = dur
                elif rule == "lpt":
                    priority = -dur
                elif rule == "mor":
                    priority = machine
                else:
                    priority = dur  # 默认SPT
                candidates.append((priority, ready_time, job, op, machine, dur))
        # 选择优先级最高且最早可调度的操作
        if not candidates:
            break  # 防止 candidates 为空时报错
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, st, job, op, machine, dur = candidates[0]
        op_start_times[job][op] = st
        job_ready[job] = st + dur
        machine_ready[machine] = st + dur
        next_op[job] += 1
        scheduled += 1

    makespan = max(op_start_times[job][m-1] + duration[job][m-1] for job in range(j))
    return op_start_times, makespan

def solve_window_with_machine_avail(jsp_instance, 
                                    ops_in_window, 
                                    op_start_times, 
                                    machine_avail, 
                                    num_works=1):
    # st = time()
    model = cp_model.CpModel()
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]
    # horizon = sum(op_start_times[job][op] for job in range(j) for op in range(m))
    horizon = max(op_start_times[job][m-1] + duration[job][m-1] for job in range(j))

    op_vars = {}
    for job, op in ops_in_window:
        machine = mch[job][op]
        lb = machine_avail[machine][0]
        ub = machine_avail[machine][1] - duration[job][op]
        machine = mch[job][op]
        # 限制操作只能在该机器的可用时间段内
        # if machine in machine_avail:
        #     lb = min(lb, machine_avail[machine][0])
        #     ub = min(ub, machine_avail[machine][1] - duration[job][op])
        if op > 0 and (job, op-1) not in ops_in_window:
            # 如果前一个操作不在窗口内，确保当前操作的开始时间不早于前一个操作的结束时间
            prev_end = op_start_times[job][op-1] + duration[job][op-1]
            lb = max(lb, prev_end)
        if op < m - 1 and (job, op + 1) not in ops_in_window:
            # 如果后一个操作不在窗口内，确保当前操作的开始时间不晚于后一个操作的开始时间
            after_start = op_start_times[job][op + 1]
            ub = min(ub, after_start - duration[job][op])
        # lb = 0
        # ub = horizon
        op_vars[(job, op)] = model.NewIntVar(lb, ub, f'start_{job}_{op}')

    # 工件顺序约束
    for job, op in ops_in_window:
        if op > 0 and (job, op-1) in ops_in_window:
            model.Add(op_vars[(job, op)] >= op_vars[(job, op-1)] + duration[job][op-1])

    # 机器约束
    machine_to_ops = {}
    interval_vars = {}
    for job, op in ops_in_window:
        machine = mch[job][op]
        machine_to_ops.setdefault(machine, []).append((job, op))
        interval_vars[(job, op)] = model.NewIntervalVar(
            op_vars[(job, op)],
            duration[job][op],
            op_vars[(job, op)] + duration[job][op],
            f'interval_{job}_{op}'
        )
    for machine, ops in machine_to_ops.items():
        machine_intervals = [interval_vars[(job, op)] for (job, op) in ops]
        model.AddNoOverlap(machine_intervals)

    makespan = model.NewIntVar(0, horizon, 'makespan')
    for job, op in ops_in_window:
        model.Add(makespan >= op_vars[(job, op)] + duration[job][op])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.1
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = SEED
    status = solver.Solve(model)
    # ed = time()
    # print("This CP use {:.2f} seconds".format(ed - st))
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result = {}
        for job, op in ops_in_window:
            result[(job, op)] = solver.Value(op_vars[(job, op)])
        return result
    else:
        return None

def solve_multiple_windows_parallel(jsp_instance, op_start_times, windows_data, cpu_count=16):
    """
    并行求解多个窗口。
    windows_data: 列表，每个元素是 (ops_in_window, machine_avail)
    返回: 结果列表，对应每个窗口的求解结果
    """
    size = len(windows_data)
    cpu_per = max(1, cpu_count // size)
    args_list = [(jsp_instance, op_start_times, ops_in_window, machine_avail, i, cpu_per) for (ops_in_window, machine_avail, i) in windows_data]
    with multiprocessing.Pool(processes=size) as pool:
        results = pool.map(_solve_single_window, args_list)
    return results

def _solve_single_window(args):
    jsp_instance, op_start_times, ops_in_window, machine_avail, i, num_works = args
    result = solve_window_with_machine_avail(jsp_instance, ops_in_window, op_start_times, machine_avail, num_works=num_works)
    return (result, i)

def check_machine_feasibility(multi_machine_avail):
    """
    检查multi_machine_avail中每台机器的可用区间是否无重叠。
    multi_machine_avail: List[Dict[machine, (start, end)]]
    返回: True（无重叠）或 False（有重叠）
    """
    machine_intervals = defaultdict(list)
    for avail in multi_machine_avail:
        for machine, (start, end) in avail.items():
            machine_intervals[machine].append((start, end))
    for intervals in machine_intervals.values():
        intervals.sort()
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i-1][1]:
                return False
    return True


def check_feasibility(jsp_instance, all_seq):
    """
    检查序列 all_seq 是否满足工件先后约束（job precedence constraints）。
    all_seq: 列表 of (job, op)，表示操作序列。
    """
    j, m = jsp_instance["j"], jsp_instance["m"]
    
    # 收集每个工件的操作序列
    job_ops = {job: [] for job in range(j)}
    for job, op in all_seq:
        if op in job_ops[job]:
            raise ValueError(f"工件 {job} 的操作 {op} 出现重复")
        job_ops[job].append(op)
    
    # 检查每个工件的操作是否按顺序（op 递增）
    for job in range(j):
        ops = job_ops[job]
        if sorted(ops) != list(range(m)):
            return False  # 操作不完整或不按顺序
        # 确保在序列中也是按顺序出现的（因为 sorted 后是 0,1,2,...）
        # 但由于我们按开始时间排序，可能不按 op 顺序，但约束是必须按 op 顺序调度
        # 在 JSP 中，工件顺序是硬约束：op 必须按 0,1,2,... 顺序
        # 所以，序列中每个工件的操作必须是 op 递增
        if ops != sorted(ops):
            return False
    return True

def compute_makespan_from_seq(jsp_instance, all_seq):
    """
    基于序列 all_seq 计算 makespan。
    all_seq: 列表 of (job, op)，表示操作调度顺序。
    """
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]
    
    # 初始化
    job_ready = [0] * j
    machine_ready = [0] * m
    op_start_times = [[0] * m for _ in range(j)]
    
    for job, op in all_seq:
        machine = mch[job][op]
        # 开始时间 = max(工件可用时间, 机器可用时间)
        start_time = max(job_ready[job], machine_ready[machine])
        op_start_times[job][op] = start_time
        end_time = start_time + duration[job][op]
        # 更新
        job_ready[job] = end_time
        machine_ready[machine] = end_time
    
    # makespan 是所有工件最后一个操作的结束时间
    makespan = max(job_ready[job] for job in range(j))
    return makespan, op_start_times

def get_machine_window_availability(ops_in_window, op_start_times, duration, mch):
    # 返回: {machine: (earliest_available, latest_finish)}
    machine_ops = {}
    for job, op in ops_in_window:
        machine = mch[job][op]
        machine_ops.setdefault(machine, []).append((job, op))
    machine_avail = {}
    for machine, ops in machine_ops.items():
        # 找到窗口内该机器所有操作的job-op索引集合
        ops_set = set(ops)
        # 找到所有操作的开始时间
        all_ops = []
        for j in range(len(op_start_times)):
            for o in range(len(op_start_times[j])):
                if mch[j][o] == machine:
                    all_ops.append((j, o))
        # 按开始时间排序
        all_ops_sorted = sorted(all_ops, key=lambda x: op_start_times[x[0]][x[1]])
        # 找到紧邻前一个和后一个操作
        idxs = [i for i, op in enumerate(all_ops_sorted) if op in ops_set]
        first_idx = idxs[0]
        last_idx = idxs[-1]
        # 采取前扩后压缩
        # 前一个操作的结束时间
        if first_idx > 0:
            prev_op = all_ops_sorted[first_idx - 1]
            earliest = op_start_times[prev_op[0]][prev_op[1]] + duration[prev_op[0]][prev_op[1]]
        else:
            earliest = 0
        # 当前操作最晚结束时间
        if last_idx < len(all_ops_sorted) - 1:
            # Debug
            next_op = all_ops_sorted[last_idx + 1]
            latest = op_start_times[next_op[0]][next_op[1]]
            # next_op = all_ops_sorted[last_idx]
            # latest = op_start_times[next_op[0]][next_op[1]] + jsp_dataset['duration'][next_op[0]][next_op[1]]
        else:
            # 没有后续操作，允许到makespan
            m = len(op_start_times[0])
            latest = max(op_start_times[j][m-1] + duration[j][m-1] for j in range(len(op_start_times)))
            
        # 检查区间是否合理
        for job, op in set(all_ops_sorted) - set(ops_set):
            assert mch[job][op] == machine, "机器不匹配"
            st = op_start_times[job][op]
            dur = duration[job][op]
            if not (st + dur <= earliest or st >= latest):
                print(f"机器{machine}上操作({job}, {op})与窗口内操作时间冲突: "
                        f"st={st}, dur={dur}, earliest={earliest}, latest={latest}")
                raise ValueError(f"窗口内操作与外部操作时间不合理: {job}, {op}")

        machine_avail[machine] = (earliest, latest)
    return machine_avail


def summarize_window_machine_intervals(jsp_instance, ops_in_window, start_times_by_op):
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]
    machine_stats = {}
    for job, op in ops_in_window:
        machine = mch[job][op]
        start_time = start_times_by_op[(job, op)]
        end_time = start_time + duration[job][op]
        if machine not in machine_stats:
            machine_stats[machine] = {
                "earliest_start": start_time,
                "latest_end": end_time,
                "ops": [(job, op)],
            }
        else:
            machine_stats[machine]["earliest_start"] = min(
                machine_stats[machine]["earliest_start"], start_time
            )
            machine_stats[machine]["latest_end"] = max(
                machine_stats[machine]["latest_end"], end_time
            )
            machine_stats[machine]["ops"].append((job, op))
    return machine_stats


def print_window_machine_interval_comparison(
    jsp_instance,
    ops_in_window,
    original_start_times,
    machine_avail,
    window_result,
    window_idx,
    iteration,
):
    before_stats = summarize_window_machine_intervals(
        jsp_instance,
        ops_in_window,
        {(job, op): original_start_times[job][op] for job, op in ops_in_window},
    )
    after_stats = summarize_window_machine_intervals(
        jsp_instance,
        ops_in_window,
        window_result,
    )

    print(f"Iteration {iteration}, window {window_idx}: machine interval comparison")
    for machine in sorted(machine_avail):
        avail_start, avail_end = machine_avail[machine]
        avail_span = avail_end - avail_start
        before = before_stats[machine]
        after = after_stats[machine]
        before_span = before["latest_end"] - before["earliest_start"]
        after_span = after["latest_end"] - after["earliest_start"]

        if after["earliest_start"] < avail_start or after["latest_end"] > avail_end:
            compression = "out_of_bounds"
        elif after_span < avail_span:
            compression = "compressed"
        elif after_span == avail_span:
            compression = "unchanged"
        else:
            compression = "expanded"

        print(
            f"  machine {machine}: "
            f"avail=[{avail_start}, {avail_end}] span={avail_span}; "
            f"before=[{before['earliest_start']}, {before['latest_end']}] span={before_span}; "
            f"after=[{after['earliest_start']}, {after['latest_end']}] span={after_span}; "
            f"delta_start={after['earliest_start'] - before['earliest_start']}, "
            f"delta_end={after['latest_end'] - before['latest_end']}, "
            f"vs_avail={compression}, ops={sorted(after['ops'])}"
        )

def large_neiborhood_search(
    jsp_instance: dict, 
    est: list, 
    window_size=10,
    max_iterations=100, 
    debug="all_windows", 
    cp_mode=False,
    neighborhood_mode="random",
    plot_improvements=False,
    plot_dir="gantt_improvements",
    return_history=False, 
    step_action=None):

    j, m = jsp_instance["j"], jsp_instance["m"]
    nopr = j * m
    duration = jsp_instance["duration"]

    # 展平成一维列表 [(job, op), ...]，并按初始解的开始时间排序
    all_ops = [(job, op) for job in range(j) for op in range(m)]
    op_start_flat = {(job, op): est[job][op] for job, op in all_ops}
    sorted_ops = sorted(all_ops, key=lambda x: op_start_flat[x])

    best_solution = [row[:] for row in est]
    best_makespan = max(est[job][m-1] + duration[job][m-1] for job in range(j))
    makespan_update_iterations = [0]
    makespan_update_values = [best_makespan]
    all_iterations = [0]
    all_best_makespans = [best_makespan]
    last_decrease_iteration = 0
    improvement_count = 0
    process_frame_paths = []
    instance_name = os.path.splitext(
        os.path.basename(jsp_instance.get("names", "instance"))
    )[0]
    if nopr <= window_size:
        if return_history:
            history = {
                "update_iterations": makespan_update_iterations,
                "update_makespans": makespan_update_values,
                "all_iterations": all_iterations,
                "all_best_makespans": all_best_makespans,
                "last_decrease_iteration": last_decrease_iteration,
                "best_iteration": makespan_update_iterations[-1],
            }
            return best_solution, best_makespan, history
        return best_solution, best_makespan

    mode_alias = {"geedy": "greedy"}
    valid_modes = {"random", "greedy", "best_improve", "first_improve"}
    neighborhood_mode = mode_alias.get(neighborhood_mode.lower(), neighborhood_mode.lower())
    if neighborhood_mode not in valid_modes:
        raise ValueError(
            f"Unsupported neighborhood_mode '{neighborhood_mode}'. "
            f"Expected one of {sorted(valid_modes)}"
        )

    def _window_idx_from_crit_op(crit_op, sorted_ops_ref):
        idx = sorted_ops_ref.index(crit_op)
        return min(idx, nopr - window_size)

    def _compute_window_slack(ops_in_window, start_times):
        machine_avail = get_machine_window_availability(
            ops_in_window,
            start_times,
            duration,
            jsp_instance["mch"],
        )
        total_op_time = 0
        for job, op in ops_in_window:
            total_op_time += duration[job][op]

        total_span = sum(latest - earliest for earliest, latest in machine_avail.values())
        return total_span - total_op_time

    def _optimize_single_window(base_start_times, sorted_ops_ref, window_idx):
        op_start_times_local = [row[:] for row in base_start_times]
        ops_in_window_local = list(sorted_ops_ref[window_idx:window_idx + window_size])
        machine_avail = get_machine_window_availability(
            ops_in_window_local, op_start_times_local, duration, jsp_instance["mch"]
        )
        window_result = solve_window_with_machine_avail(
            jsp_instance,
            ops_in_window_local,
            op_start_times_local,
            machine_avail,
        )
        if not window_result:
            return None, None, ops_in_window_local

        ref_start_times = [row[:] for row in op_start_times_local]
        for (job, op), st in window_result.items():
            op_start_times_local[job][op] = st

        orders = [[] for _ in range(m)]
        for machine in range(m):
            machine_ops = [
                (job, op)
                for job in range(j)
                for op in range(m)
                if jsp_instance["mch"][job][op] == machine
            ]
            machine_ops.sort(
                key=lambda item: (
                    op_start_times_local[item[0]][item[1]],
                    ref_start_times[item[0]][item[1]],
                    item[0],
                    item[1],
                )
            )
            orders[machine] = machine_ops

        rebuilt_start_times = [[0] * m for _ in range(j)]
        machine_ready = [0] * m
        job_ready = [0] * j
        order_idx = [0] * m
        scheduled_ops = set()
        while len(scheduled_ops) < nopr:
            progress = False
            for machine in range(m):
                if order_idx[machine] >= len(orders[machine]):
                    continue
                job, op = orders[machine][order_idx[machine]]
                if (job, op) in scheduled_ops:
                    order_idx[machine] += 1
                    continue
                if op > 0 and (job, op - 1) not in scheduled_ops:
                    continue

                start_time = max(job_ready[job], machine_ready[machine])
                rebuilt_start_times[job][op] = start_time
                end_time = start_time + duration[job][op]
                job_ready[job] = end_time
                machine_ready[machine] = end_time
                scheduled_ops.add((job, op))
                order_idx[machine] += 1
                progress = True
            if not progress:
                raise ValueError("调度过程中没有进展，可能是orders中的操作顺序不合理，导致死锁")

        makespan_local = max(
            rebuilt_start_times[job][m - 1] + duration[job][m - 1]
            for job in range(j)
        )
        return rebuilt_start_times, makespan_local, ops_in_window_local
    
    count = 0
    it = 0
    roll_count = 0
    find_better_in_this_iteration = True
    while it < max_iterations:
        op_start_times = [row[:] for row in best_solution]

        _est = [row[:] for row in op_start_times]
        makespan = max(
            _est[job][jsp_instance["m"] - 1] + jsp_instance["duration"][job][jsp_instance["m"] - 1]
            for job in range(jsp_instance["j"])
        )
        end_ops = []
        for job in range(jsp_instance["j"]):
            op = jsp_instance["m"] - 1
            if abs(_est[job][op] + jsp_instance["duration"][job][op] - makespan) < 1e-6:
                end_ops.append((job, op))

        def trace_critical_path(start_op):
            path = [start_op]
            job, op = start_op
            machine_ops_order = {}
            for mach in range(jsp_instance["m"]):
                ops_on_machine = []
                for j2 in range(jsp_instance["j"]):
                    for o2 in range(jsp_instance["m"]):
                        if jsp_instance["mch"][j2][o2] == mach:
                            ops_on_machine.append((j2, o2))
                ops_on_machine.sort(key=lambda x: _est[x[0]][x[1]])
                machine_ops_order[mach] = ops_on_machine

            while True:
                prev_candidates = []
                if op > 0:
                    prev_job_op = (job, op - 1)
                    if abs(
                        _est[prev_job_op[0]][prev_job_op[1]]
                        + jsp_instance["duration"][prev_job_op[0]][prev_job_op[1]]
                        - _est[job][op]
                    ) < 1e-6:
                        prev_candidates.append(prev_job_op)
                machine = jsp_instance["mch"][job][op]
                ops_on_machine = machine_ops_order[machine]
                idx = ops_on_machine.index((job, op))
                if idx > 0:
                    prev_op = ops_on_machine[idx - 1]
                    prev_j, prev_o = prev_op
                    if abs(_est[prev_j][prev_o] + jsp_instance["duration"][prev_j][prev_o] - _est[job][op]) < 1e-6:
                        prev_candidates.append(prev_op)
                if not prev_candidates:
                    break
                prev_op = random.choice(prev_candidates)
                path.append(prev_op)
                job, op = prev_op
            return list(reversed(path))

        critical_ops = [] if not end_ops else trace_critical_path(random.choice(end_ops))
        if len(critical_ops) == 0:
            raise AssertionError("没有找到关键路径上的操作，可能是计算lst时出现了问题")

        selected_window_idx = None
        selected_ops_in_window = None
        candidate_start_times = None
        candidate_makespan = None
        force_restart_accept = False

        if step_action is not None:
            assert len(step_action) == max_iterations, f"step_action长度 {len(step_action)} 不等于 max_iterations {max_iterations}"
            crit_op_id = step_action[it]
            crit_op = (crit_op_id // m, crit_op_id % m)
            selected_window_idx = _window_idx_from_crit_op(crit_op, sorted_ops)
        elif neighborhood_mode == "random":
            selected_window_idx = _window_idx_from_crit_op(random.choice(critical_ops), sorted_ops)
        elif neighborhood_mode == "greedy":
            best_slack = None
            for crit_op in critical_ops:
                cur_idx = _window_idx_from_crit_op(crit_op, sorted_ops)
                ops_in_window = sorted_ops[cur_idx:cur_idx + window_size]
                cur_slack = _compute_window_slack(ops_in_window, op_start_times)
                if best_slack is None or cur_slack > best_slack:
                    best_slack = cur_slack
                    selected_window_idx = cur_idx
        elif neighborhood_mode == "best_improve":
            best_improve_value = 0
            best_candidate = None
            all_candidates = []
            for crit_op in critical_ops:
                cur_idx = _window_idx_from_crit_op(crit_op, sorted_ops)
                cand_start_times, cand_makespan, cand_ops = _optimize_single_window(
                    op_start_times,
                    sorted_ops,
                    cur_idx,
                )
                if cand_start_times is None:
                    continue
                all_candidates.append((cur_idx, cand_start_times, cand_makespan, cand_ops))
                improve = best_makespan - cand_makespan
                if improve > best_improve_value:
                    best_improve_value = improve
                    best_candidate = (cur_idx, cand_start_times, cand_makespan, cand_ops)
            if best_candidate is not None:
                selected_window_idx, candidate_start_times, candidate_makespan, selected_ops_in_window = best_candidate
            elif all_candidates:
                min_makespan = min(item[2] for item in all_candidates)
                best_pool = [item for item in all_candidates if item[2] == min_makespan]
                selected_window_idx, candidate_start_times, candidate_makespan, selected_ops_in_window = random.choice(best_pool)
                force_restart_accept = True
        elif neighborhood_mode == "first_improve":
            all_candidates = []
            for crit_op in critical_ops:
                cur_idx = _window_idx_from_crit_op(crit_op, sorted_ops)
                cand_start_times, cand_makespan, cand_ops = _optimize_single_window(
                    op_start_times,
                    sorted_ops,
                    cur_idx,
                )
                if cand_start_times is None:
                    continue
                all_candidates.append((cur_idx, cand_start_times, cand_makespan, cand_ops))
                if cand_makespan < best_makespan:
                    selected_window_idx = cur_idx
                    candidate_start_times = cand_start_times
                    candidate_makespan = cand_makespan
                    selected_ops_in_window = cand_ops
                    break
            if selected_window_idx is None and all_candidates:
                min_makespan = min(item[2] for item in all_candidates)
                best_pool = [item for item in all_candidates if item[2] == min_makespan]
                selected_window_idx, candidate_start_times, candidate_makespan, selected_ops_in_window = random.choice(best_pool)
                force_restart_accept = True

        it += 1

        if selected_window_idx is None:
            all_iterations.append(it)
            all_best_makespans.append(best_makespan)
            continue

        if selected_ops_in_window is None:
            selected_ops_in_window = sorted_ops[selected_window_idx:selected_window_idx + window_size]
        focus_window_before = sorted(
            selected_ops_in_window,
            key=lambda item: op_start_times[item[0]][item[1]],
        )
        # if cp_mode:
        #     assert max_iterations == 1, f"CP mode must make iteration as 1"
        #     assert debug == 'single_windows', f"CP mode debug must be single window"
        #     ops_in_window = sorted_ops[:]
        #     machine_avail = get_machine_window_availability(
        #             ops_in_window, op_start_times, duration, jsp_instance["mch"]
        #         )
        #         # st = time()
        #     window_result = solve_window_with_machine_avail(jsp_instance, 
        #                                                     ops_in_window, 
        #                                                     op_start_times, 
        #                                                     machine_avail)
        #     for (job, op), st in window_result.items():
        #         op_start_times[job][op] = st
        #     makespan = max(op_start_times[job][m-1] + duration[job][m-1] for job in range(j))
        #     if makespan < best_makespan:
        #         makespan_update_iterations.append(it)
        #         makespan_update_values.append(makespan)
        #         best_makespan = makespan
        #         last_decrease_iteration = it
        #     all_iterations.append(it)
        #     all_best_makespans.append(best_makespan)
        #     if return_history:
        #         history = {
        #             "update_iterations": makespan_update_iterations,
        #             "update_makespans": makespan_update_values,
        #             "all_iterations": all_iterations,
        #             "all_best_makespans": all_best_makespans,
        #             "last_decrease_iteration": last_decrease_iteration,
        #             "best_iteration": makespan_update_iterations[-1],
        #         }
        #         return op_start_times, makespan, history
        #     return op_start_times, makespan
            
        if candidate_start_times is None:
            candidate_start_times, candidate_makespan, selected_ops_in_window = _optimize_single_window(
                op_start_times,
                sorted_ops,
                selected_window_idx,
            )
            if candidate_start_times is None:
                all_iterations.append(it)
                all_best_makespans.append(best_makespan)
                continue

        op_start_times = candidate_start_times
        makespan = candidate_makespan
        should_accept = False
        if neighborhood_mode in {"best_improve", "first_improve", "random"}:
            should_accept = makespan < best_makespan or force_restart_accept
        else:
            should_accept = True

        if should_accept:
            if makespan < best_makespan:
                count += 1
                makespan_update_iterations.append(it)
                makespan_update_values.append(makespan)
                last_decrease_iteration = it
            previous_solution = [row[:] for row in best_solution]
            previous_makespan = best_makespan
            focus_window_after = sorted(
                focus_window_before,
                key=lambda item: op_start_times[item[0]][item[1]],
            )
            changed_ops = _collect_reordered_ops(
                focus_window_before,
                focus_window_after,
            )
            best_solution = [row[:] for row in op_start_times]
            best_makespan = makespan
            if plot_improvements:
                file_stem = (
                    f"{instance_name}_improve_{improvement_count:03d}_"
                    f"it_{it:03d}_ms_{best_makespan}"
                )
                before_path = os.path.join(plot_dir, f"{file_stem}_before.png")
                after_path = os.path.join(plot_dir, f"{file_stem}_after.png")
                window_change_info = {
                    "before_order": focus_window_before,
                    "after_order": focus_window_after,
                    "changed_ops": changed_ops,
                }
                plot_gantt(
                    previous_solution,
                    jsp_instance,
                    title=(
                        f"{instance_name} | improve #{improvement_count} | "
                        f"before reorder | makespan={previous_makespan}"
                    ),
                    highlight_region=_build_window_highlight(
                        previous_solution,
                        jsp_instance,
                        focus_window_before,
                    ),
                    window_change_info=window_change_info,
                    save_path=before_path,
                )
                plot_gantt(
                    best_solution,
                    jsp_instance,
                    title=(
                        f"{instance_name} | improve #{improvement_count} | "
                        f"after reorder | makespan={best_makespan}"
                    ),
                    highlight_region=_build_window_highlight(
                        best_solution,
                        jsp_instance,
                        focus_window_after,
                    ),
                    window_change_info=window_change_info,
                    save_path=after_path,
                )
                save_gantt_transition_gif(
                    before_path,
                    after_path,
                    os.path.join(plot_dir, f"{file_stem}.gif"),
                )
                process_frame_paths.extend([before_path, after_path])
            # DEBUG: 仅在特定实例上打印改进信息
            # print(f"Iteration {it}: Found better solution with makespan {best_makespan}")
            if debug == 'all_windows':
                total_windows = (nopr - selected_window_idx) // window_size
                print(f"Iteration {it}: Found better solution with makespan {best_makespan}, total windows is {total_windows}, "
                      f"prefix windows is {selected_window_idx // window_size}, suffix windows is {(nopr - selected_window_idx) // window_size}")
            # 更新
            all_ops = [(job, op) for job in range(j) for op in range(m)]
            op_start_flat = {(job, op): op_start_times[job][op] for job, op in all_ops}
            sorted_ops = sorted(all_ops, key=lambda x: op_start_flat[x])
            find_better_in_this_iteration = True
            
            # break
        else:
            find_better_in_this_iteration = False

        all_iterations.append(it)
        all_best_makespans.append(best_makespan)

    if plot_improvements and process_frame_paths:
        save_gantt_search_process_gif(
            process_frame_paths,
            os.path.join(plot_dir, f"{instance_name}_search_process.gif"),
        )

    print(f"LNS finished after {max_iterations} iterations, improved {count} times, best makespan: {best_makespan}")
    if return_history:
        history = {
            "update_iterations": makespan_update_iterations,
            "update_makespans": makespan_update_values,
            "all_iterations": all_iterations,
            "all_best_makespans": all_best_makespans,
            "last_decrease_iteration": last_decrease_iteration,
            "best_iteration": makespan_update_iterations[-1],
        }
        return best_solution, best_makespan, history
    return best_solution, best_makespan, None


def convert_solution_to_start_times(sol, jsp_instance):
    """
    sol: 形状为 (m, n)，每行是该机器上操作的(job, op)元组顺序
    返回: op_start_times, 形状为 (j, m)
    """
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]

    op_start_times = [[0] * m for _ in range(j)]
    # 每个工件的上一个操作完成时间
    job_ready = [0] * j
    # 每台机器的最早可用时间
    machine_ready = [0] * m
    scheduled_set = set()
    schedule_index = [0] * m

    while len(scheduled_set) < j * m:
        for machine in range(m):
            cur_op_index = schedule_index[machine]
            if cur_op_index >= len(sol[machine]):
                continue  # 该机器上的操作已经全部处理完
            op_idx = sol[machine][cur_op_index]
            job, op = op_idx // m, op_idx % m

            if op > 0:
                # 前一个操作的完成时间
                if (job, op-1) not in scheduled_set:
                    continue  # 前一个操作还未调度完成，跳过
                else:
                    prev_end = op_start_times[job][op-1] + duration[job][op-1]
                    op_start_times[job][op] = max(prev_end, machine_ready[machine])
                    
                    machine_ready[machine] = op_start_times[job][op] + duration[job][op]
                    job_ready[job] = op_start_times[job][op] + duration[job][op]

                    scheduled_set.add((job, op))
                    schedule_index[machine] += 1
            else:
                # 第一个操作，直接安排在机器可用时间之后
                op_start_times[job][op] = machine_ready[machine]
                machine_ready[machine] = op_start_times[job][op] + duration[job][op]
                job_ready[job] = op_start_times[job][op] + duration[job][op]

                scheduled_set.add((job, op))
                schedule_index[machine] += 1

    return op_start_times


def _format_op_label(op):
    return f"J{op[0]}O{op[1]}"


def _collect_reordered_ops(before_order, after_order):
    changed_ops = set()
    max_len = max(len(before_order), len(after_order))
    for idx in range(max_len):
        before_op = before_order[idx] if idx < len(before_order) else None
        after_op = after_order[idx] if idx < len(after_order) else None
        if before_op != after_op:
            if before_op is not None:
                changed_ops.add(before_op)
            if after_op is not None:
                changed_ops.add(after_op)
    return changed_ops


def _build_window_highlight(op_start_times, jsp_instance, ops_in_window):
    if not ops_in_window:
        return None

    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]
    machine_orders = defaultdict(list)
    for job, op in ops_in_window:
        machine_orders[mch[job][op]].append((job, op))

    sorted_machine_orders = {
        machine: sorted(
            order,
            key=lambda item: op_start_times[item[0]][item[1]],
        )
        for machine, order in machine_orders.items()
    }
    if not sorted_machine_orders:
        return None

    return {"machine_orders": sorted_machine_orders}


def _draw_window_sequence_panel(info_ax, before_order, after_order, changed_ops):
    def draw_sequence(title, ops, top, bottom):
        info_ax.text(
            0.02,
            top,
            title,
            transform=info_ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )
        if not ops:
            info_ax.text(
                0.02,
                top - 0.06,
                "(empty)",
                transform=info_ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="gray",
            )
            return

        body_top = top - 0.06
        body_bottom = bottom + 0.02
        target_rows = 10
        tokens_per_row = max(4, int(np.ceil(len(ops) / target_rows)))
        rows = int(np.ceil(len(ops) / tokens_per_row))
        row_step = (body_top - body_bottom) / max(rows - 1, 1)
        col_step = 0.96 / max(tokens_per_row, 1)
        font_size = max(5.5, 9 - max(0, rows - 8) * 0.35)

        for idx, op in enumerate(ops):
            row = idx // tokens_per_row
            col = idx % tokens_per_row
            x = 0.02 + col * col_step
            y = body_top - row * row_step
            is_changed = op in changed_ops
            info_ax.text(
                x,
                y,
                _format_op_label(op),
                transform=info_ax.transAxes,
                ha="left",
                va="top",
                fontsize=font_size,
                fontweight="bold" if is_changed else "normal",
                color="darkred" if is_changed else "black",
                family="monospace",
            )

    info_ax.set_axis_off()
    info_ax.text(
        0.02,
        0.99,
        "Window reorder",
        transform=info_ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    info_ax.text(
        0.02,
        0.95,
        "Changed operations are highlighted in bold.",
        transform=info_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="dimgray",
    )
    draw_sequence("Before order", before_order, 0.90, 0.53)
    info_ax.plot(
        [0.02, 0.98],
        [0.50, 0.50],
        transform=info_ax.transAxes,
        color="lightgray",
        linewidth=1.0,
    )
    draw_sequence("After order", after_order, 0.47, 0.05)


def plot_gantt(
    op_start_times,
    jsp_instance,
    title="Job Shop Gantt",
    highlight_region=None,
    window_change_info=None,
    save_path=None,
):
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]
    max_completion = max(
        op_start_times[job][op] + duration[job][op]
        for job in range(j)
        for op in range(m)
    )
    label_inside_threshold = max(2.0, max_completion * 0.035)
    changed_ops = set()
    if window_change_info is not None:
        changed_ops = window_change_info.get("changed_ops")
        if changed_ops is None:
            changed_ops = _collect_reordered_ops(
                window_change_info.get("before_order", []),
                window_change_info.get("after_order", []),
            )

    fig, (ax, info_ax) = plt.subplots(
        1,
        2,
        figsize=(18, 7),
        gridspec_kw={"width_ratios": [3.7, 2.3]},
    )
    colors = plt.get_cmap("tab20").colors

    if highlight_region is not None:
        for machine, region_order in sorted(highlight_region.get("machine_orders", {}).items()):
            if not region_order:
                continue

            starts = [op_start_times[job][op] for job, op in region_order]
            finishes = [
                op_start_times[job][op] + duration[job][op]
                for job, op in region_order
            ]
            region_start = min(starts)
            region_end = max(finishes)
            ax.add_patch(
                Rectangle(
                    (region_start, machine - 0.48),
                    region_end - region_start,
                    0.96,
                    facecolor="gold",
                    edgecolor="none",
                    alpha=0.18,
                    zorder=0.5,
                )
            )

    for job in range(j):
        for op in range(m):
            start = op_start_times[job][op]
            dur = duration[job][op]
            machine = mch[job][op]
            color = colors[job % len(colors)]
            is_changed_op = (job, op) in changed_ops
            ax.barh(
                machine,
                dur,
                left=start,
                height=0.8,
                color=color,
                edgecolor="darkred" if is_changed_op else "black",
                linewidth=2.5 if is_changed_op else 1.0,
                alpha=1.0 if is_changed_op else 0.8,
                hatch="//" if is_changed_op else None,
                zorder=3 if is_changed_op else 2,
            )
            label = _format_op_label((job, op))
            if dur >= label_inside_threshold:
                ax.text(
                    start + dur / 2,
                    machine,
                    label,
                    va="center",
                    ha="center",
                    fontsize=9 if is_changed_op else 8,
                    fontweight="bold" if is_changed_op else "normal",
                    color="white",
                    zorder=4,
                )
            else:
                offset_y = 0.22 if (job + op) % 2 == 0 else -0.22
                ax.text(
                    start + dur + max_completion * 0.005,
                    machine + offset_y,
                    label,
                    va="center",
                    ha="left",
                    fontsize=8,
                    fontweight="bold" if is_changed_op else "normal",
                    color="darkred" if is_changed_op else "black",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "darkred" if is_changed_op else "none",
                        "alpha": 0.85,
                    },
                    zorder=5,
                    clip_on=False,
                )

    if window_change_info is not None:
        _draw_window_sequence_panel(
            info_ax,
            window_change_info.get("before_order", []),
            window_change_info.get("after_order", []),
            changed_ops,
        )
    else:
        info_ax.set_axis_off()

    ax.set_ylabel("Machine")
    ax.set_xlabel("Time")
    ax.set_xlim(0, max_completion + max(1.0, max_completion * 0.08))
    ax.set_yticks(range(max(max(row) for row in mch) + 1))
    ax.set_yticklabels([f"M{i}" for i in range(max(max(row) for row in mch) + 1)])
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is None:
        safe_title = "".join(
            ch if ch.isalnum() or ch in "._-()[]{}=+" else "_"
            for ch in title
        )
        save_path = f"{safe_title}.png"

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_gantt_transition_gif(before_path, after_path, gif_path, frame_duration=900):
    gif_dir = os.path.dirname(gif_path)
    if gif_dir:
        os.makedirs(gif_dir, exist_ok=True)

    with Image.open(before_path) as before_img, Image.open(after_path) as after_img:
        before_frame = before_img.convert("P", palette=Image.ADAPTIVE)
        after_frame = after_img.convert("P", palette=Image.ADAPTIVE)
        before_frame.save(
            gif_path,
            save_all=True,
            append_images=[after_frame],
            duration=[frame_duration, frame_duration],
            loop=0,
        )


def save_gantt_search_process_gif(frame_paths, gif_path, frame_duration=700):
    if not frame_paths:
        return

    gif_dir = os.path.dirname(gif_path)
    if gif_dir:
        os.makedirs(gif_dir, exist_ok=True)

    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as frame_img:
            frames.append(frame_img.convert("P", palette=Image.ADAPTIVE))

    if not frames:
        return

    first_frame, *rest_frames = frames
    first_frame.save(
        gif_path,
        save_all=True,
        append_images=rest_frames,
        duration=[frame_duration] * len(frames),
        loop=0,
    )


def _to_scalar_makespan(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return None
        return float(arr[0])
    return float(value)


def plot_makespan_update_curve(
    instance_name,
    all_iterations,
    all_best_makespans,
    best_known_makespan=None,
    last_decrease_iteration=None,
    save_path=None,
):
    if not all_iterations or not all_best_makespans:
        return None

    best_known = _to_scalar_makespan(best_known_makespan)
    x = np.asarray(all_iterations, dtype=float)
    y = np.asarray(all_best_makespans, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, y, color="tab:blue", linewidth=1.8, label="Best-so-far makespan (every iteration)")

    if best_known is not None:
        ax.axhline(
            y=best_known,
            color="tab:orange",
            linestyle="--",
            linewidth=2.0,
            label=f"Gold standard={best_known:.2f}",
        )

    if last_decrease_iteration is None:
        diffs = np.diff(y)
        decrease_idxs = np.where(diffs < 0)[0]
        if decrease_idxs.size > 0:
            last_decrease_iteration = int(x[decrease_idxs[-1] + 1])
        else:
            last_decrease_iteration = 0

    last_idx = int(np.where(x == float(last_decrease_iteration))[0][-1]) if np.any(x == float(last_decrease_iteration)) else int(np.argmin(y))
    best_iter = int(x[last_idx])
    best_value = float(y[last_idx])
    ax.scatter([best_iter], [best_value], color="crimson", s=70, zorder=4)
    ax.axvline(best_iter, color="crimson", linestyle=":", linewidth=1.5, alpha=0.9)
    ax.annotate(
        f"last decrease at iter {best_iter}\nfinal best={best_value:.2f}",
        xy=(best_iter, best_value),
        xytext=(14, -28),
        textcoords="offset points",
        fontsize=10,
        color="crimson",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
        arrowprops={"arrowstyle": "->", "color": "crimson", "lw": 1.1},
    )

    ax.set_title(f"{instance_name} makespan update curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Makespan")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()

    if save_path is None:
        safe_name = "".join(
            ch if ch.isalnum() or ch in "._-()[]{}=+" else "_"
            for ch in str(instance_name)
        )
        save_path = f"{safe_name}_makespan_curve.png"

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    import os
    allowed = set(range(1, os.cpu_count()))
    os.sched_setaffinity(0, allowed)
    print("CPU affinity:", os.sched_getaffinity(0))

    dataset = JSPNumpyDataset(data_dir="./benchmark/TA")
    gaps = ObjMeter()
    better_gaps = ObjMeter()
    random.seed(SEED)
    np.random.seed(SEED)
    st = time()

    from pdrs import solve_instance, PDR, SPT, FDDDivideMWKR

    pdr = PDR(priority=SPT())
    # Debug for validation
    # import numpy as np
    # test_dataset = np.load("./L2S/validation_data/validation_instance_20x15[1,99].npy", allow_pickle=True)
    # instance = [test_dataset[i] for i in range(test_dataset.shape[0])]
    # jsp_dataset = [{
    # "names": f"validation_instance_20x15_{idx}",
    # "j": 20,
    # "m": 15,
    # "duration": ins[0],
    # "mch": ins[1] - 1,
    # } for idx, ins in enumerate(instance)]
    # dataset = jsp_dataset
    # count = 0
    # start_var = 1
    # end_var = len(dataset)
    count = 0
    start_var = 21
    end_var = 21
    for jsp_dataset in dataset:
        count += 1
        if not (count >= start_var and count <= end_var):
            continue
        window_size = min(150, jsp_dataset["j"] * jsp_dataset["m"])
        sols, ms, times = solve_instance(jsp_dataset, pdr=pdr)
        
        op_start_times = convert_solution_to_start_times(sols, jsp_dataset)
        assert ms == max(op_start_times[job][jsp_dataset["m"]-1] + jsp_dataset["duration"][job][jsp_dataset["m"]-1] for job in range(jsp_dataset["j"])), \
            f"计算的makespan {ms} 与根据op_start_times计算的makespan不一致 {max(op_start_times[job][jsp_dataset['m']-1] + jsp_dataset['duration'][job][jsp_dataset['m']-1] for job in range(jsp_dataset['j']))}"
        

        best_known = _to_scalar_makespan(jsp_dataset.get("makespan"))
        restore_history = False
        if best_known is not None:
            print(
                f"Initial solution for {jsp_dataset['names']} has makespan {ms} "
                f"with window size {window_size}, best known makespan {best_known:.2f}"
            )
        else:
            print(
                f"Initial solution for {jsp_dataset['names']} has makespan {ms} "
                f"with window size {window_size}, no best known makespan"
            )
        # step_action = list(np.load('L2S/test_learned_step_actions.npy', allow_pickle=True) - 1)
        step_action = None
        better_solution, better_makespan, search_history = large_neiborhood_search(jsp_dataset, 
                                                                                    op_start_times, 
                                                                                    window_size=window_size, 
                                                                                    max_iterations=500, 
                                                                                    debug="single_windows",
                                                                                    cp_mode=False,
                                                                                    neighborhood_mode="random",
                                                                                    plot_improvements=False,
                                                                                    plot_dir=f"gantt_improvements_cp_ta{start_var}——test",
                                                                                    return_history=restore_history, 
                                                                                    step_action=step_action)
        if restore_history and search_history is not None:
            curve_output_dir = "gantt_improvements_ts_ms"
            curve_path = os.path.join(
                curve_output_dir,
                f"seed_{SEED}_",
                f"{jsp_dataset['names']}_makespan_curve.png",
            )
            saved_curve_path = plot_makespan_update_curve(
                jsp_dataset["names"],
                search_history["all_iterations"],
                search_history["all_best_makespans"],
                best_known_makespan=jsp_dataset.get("makespan"),
                last_decrease_iteration=search_history["last_decrease_iteration"],
                save_path=curve_path,
            )
            if saved_curve_path is not None:
                print(
                    f"Saved makespan curve for {jsp_dataset['names']} to {saved_curve_path}; "
                    f"last decrease appears at iteration {search_history['last_decrease_iteration']}"
                )
        # print("调度结果开始时间：")
        # for job, starts in enumerate(op_start_times):
        #     print(f"Job {job}: {starts}")
        # print("Makespan:", ms)
        gap = (ms - jsp_dataset["makespan"]) / jsp_dataset["makespan"] * 100
        better_gap = (better_makespan - jsp_dataset["makespan"]) / jsp_dataset["makespan"] * 100
        # gap = ms 
        # better_gap = better_makespan
        print(jsp_dataset["names"], f" Gap: {gap[0]:.2f}", f" Better Gap: {better_gap[0]:.2f}")
        # print(jsp_dataset["names"], f" Gap: {gap:.2f}", f" Better Gap: {better_gap:.2f}")
        # gaps.update(jsp_dataset, gap)
        # better_gaps.update(jsp_dataset, better_gap)
        # break
        gaps.update(jsp_dataset, gap[0])
        better_gaps.update(jsp_dataset, better_gap[0])
        # plot_gantt(op_start_times, jsp_dataset, title=f"{jsp_dataset['names']}(makespan={better_gap[0]:.2f})")
    print("Overall Gaps:", gaps.avg)
    print(gaps)
    print("Overall Better Gaps:", better_gaps.avg)
    print(better_gaps)
    end = time()
    print(f"Total time: {end - st:.2f} seconds")
