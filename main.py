import numpy as np
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
from dataset import JSPNumpyDataset, ObjMeter
import random
import multiprocessing

def compute_earliest_start_times(jsp_instance):
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    est = [[0]*m for _ in range(j)]
    for job in range(j):
        for op in range(1, m):
            est[job][op] = est[job][op-1] + duration[job][op-1]
    return est

def get_next_window(remaining_ops, est, window_size):
    # 选取最早可开始的window_size个操作
    ops_with_est = []
    for (job, op) in remaining_ops:
        ops_with_est.append(((job, op), est[job][op]))
    ops_with_est.sort(key=lambda x: x[1])
    selected = [x[0] for x in ops_with_est[:window_size]]
    return selected

def solve_window(jsp_instance, ops_in_window, op_start_times):
    model = cp_model.CpModel()
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]

    # 为窗口内的操作建立变量
    op_vars = {}
    for job, op in ops_in_window:
        op_vars[(job, op)] = model.NewIntVar(op_start_times[job][op], 10000, f'start_{job}_{op}')

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
        # 创建区间变量
        interval_vars[(job, op)] = model.NewIntervalVar(
            op_vars[(job, op)],
            duration[job][op],
            op_vars[(job, op)] + duration[job][op],
            f'interval_{job}_{op}'
        )
    for machine, ops in machine_to_ops.items():
        machine_intervals = [interval_vars[(job, op)] for (job, op) in ops]
        model.AddNoOverlap(machine_intervals)

    # 目标：最小化窗口内最大结束时间
    makespan = model.NewIntVar(0, 10000, 'makespan')
    for job, op in ops_in_window:
        model.Add(makespan >= op_vars[(job, op)] + duration[job][op])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 16  # 设置求解时间限制
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result = {}
        for job, op in ops_in_window:
            result[(job, op)] = solver.Value(op_vars[(job, op)])
        return result
    else:
        return {}

def rolling_horizon_cp(jsp_instance, window_size=10, roll_speed=5):
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]
    remaining_ops = [(job, op) for job in range(j) for op in range(m)]
    scheduled_ops = set()

    # 记录每台机器上已分配的区间
    machine_busy_intervals = {machine: 0 for machine in range(m)}
    est = [[0]*m for _ in range(j)]

    def compute_partial_est():
        # 计算考虑已调度操作和机器占用的最早开始时间
        for job in range(j):
            for op in range(m):
                if (job, op) in scheduled_ops:
                    continue
                # 工件顺序约束
                if op > 0:
                    est[job][op] = max(est[job][op], est[job][op-1] + duration[job][op-1])
                # 机器约束
                machine = mch[job][op]
                prev_end = machine_busy_intervals[machine]
                est[job][op] = max(est[job][op], prev_end)
        return est

    while remaining_ops:
        est = compute_partial_est()
        # 选取最早可开始的window_size个操作
        ops_with_est = []
        for (job, op) in remaining_ops:
            ops_with_est.append(((job, op), est[job][op]))
        ops_with_est.sort(key=lambda x: x[1])
        ops_in_window = [x[0] for x in ops_with_est[:window_size]]

        window_result = solve_window(jsp_instance, ops_in_window, est)
        # 1. 选择window_size中最早开始的roll_speed个工件（操作）
        ops_with_st = [(op, window_result[op]) for op in ops_in_window]
        ops_with_st.sort(key=lambda x: x[1])
        selected_ops = [op for op, _ in ops_with_st[:roll_speed]]

        # 2. 这些工件的执行顺序确定，更新est和machine_busy_intervals
        for op in selected_ops:
            job, op_idx = op
            st = window_result[op]
            est[job][op_idx] = st
            scheduled_ops.add(op)
            machine = mch[job][op_idx]
            machine_busy_intervals[machine] = max(machine_busy_intervals[machine], st + duration[job][op_idx])

        # 3. 剩下的window_size-roll_speed个工件和其他所有工件，结合已完成工件的est和当前machine_busy_intervals，更新其est
        # 这里不需要显式更新est，因为compute_partial_est会在下次循环时重新计算

        # 4. 滚动窗口：移除最早开始的roll_speed个工件
        for op in selected_ops:
            if op in remaining_ops:
                remaining_ops.remove(op)
        # 若窗口内操作数小于窗口大小，全部移除
        # if len(ops_in_window) < window_size:
        #     for op in ops_in_window:
        #         if op in remaining_ops:
        #             remaining_ops.remove(op)

    # 计算makespan
    makespan = max(est[job][m-1] + duration[job][m-1] for job in range(j))
    return est, makespan

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

def solve_window_with_machine_avail(jsp_instance, ops_in_window, op_start_times, machine_avail):
    model = cp_model.CpModel()
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]

    op_vars = {}
    for job, op in ops_in_window:
        lb = op_start_times[job][op]
        ub = 10000
        machine = mch[job][op]
        # 限制操作只能在该机器的可用时间段内
        if machine in machine_avail:
            lb = min(lb, machine_avail[machine][0])
            ub = min(ub, machine_avail[machine][1] - duration[job][op])
        if op > 0 and (job, op-1) not in ops_in_window:
            # 如果前一个操作不在窗口内，确保当前操作的开始时间不早于前一个操作的结束时间
            prev_end = op_start_times[job][op-1] + duration[job][op-1]
            lb = max(lb, prev_end)
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

    makespan = model.NewIntVar(0, 10000, 'makespan')
    for job, op in ops_in_window:
        model.Add(makespan >= op_vars[(job, op)] + duration[job][op])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result = {}
        for job, op in ops_in_window:
            result[(job, op)] = solver.Value(op_vars[(job, op)])
        return result
    else:
        return {}

def solve_multiple_windows_parallel(jsp_instance, op_start_times, windows_data):
    """
    并行求解多个窗口。
    windows_data: 列表，每个元素是 (ops_in_window, machine_avail)
    返回: 结果列表，对应每个窗口的求解结果
    """
    def _solve_single_window(args):
        ops_in_window, machine_avail = args
        return solve_window_with_machine_avail(jsp_instance, ops_in_window, op_start_times, machine_avail)
    
    with multiprocessing.Pool() as pool:
        results = pool.map(_solve_single_window, windows_data)
    return results

def large_neiborhood_search(
    jsp_instance: dict, 
    est: list, 
    window_size=10,
    max_iterations=100):

    j, m = jsp_instance["j"], jsp_instance["m"]
    nopr = j * m
    duration = jsp_instance["duration"]

    # 展平成一维列表 [(job, op), ...]，并按初始解的开始时间排序
    all_ops = [(job, op) for job in range(j) for op in range(m)]
    op_start_flat = {(job, op): est[job][op] for job, op in all_ops}
    sorted_ops = sorted(all_ops, key=lambda x: op_start_flat[x])

    best_solution = [row[:] for row in est]
    best_makespan = max(est[job][m-1] + duration[job][m-1] for job in range(j))
    if nopr <= window_size:
        return best_solution, best_makespan
    
    def compute_lst(jsp_instance, op_start_times):
        j, m = jsp_instance["j"], jsp_instance["m"]
        duration = jsp_instance["duration"]
        mch = jsp_instance["mch"]
        makespan = max(op_start_times[job][m-1] + duration[job][m-1] for job in range(j))
        lst = [[makespan]*m for _ in range(j)]
        for job in range(j):
            for op in reversed(range(m)):
                if op == m-1:
                    lst[job][op] = makespan - duration[job][op]
                else:
                    lst[job][op] = min(lst[job][op], lst[job][op+1] - duration[job][op])
        # 机器约束
        # 1. 收集每台机器上的所有操作及其开始时间
        machine_to_ops = {machine: [] for machine in range(m)}
        for job in range(j):
            for op in range(m):
                machine = mch[job][op]
                machine_to_ops[machine].append((job, op, op_start_times[job][op]))

        # 2. 对每台机器的操作按开始时间排序
        for machine in machine_to_ops:
            machine_to_ops[machine].sort(key=lambda x: x[2])

        # 3. 迭代更新lst，直到收敛
        changed = True
        while changed:
            changed = False
            for machine, ops in machine_to_ops.items():
                n_ops = len(ops)
                for idx, (job, op, _) in enumerate(ops):
                    # 找到该操作在机器上的下一个操作
                    if idx + 1 < n_ops:
                        next_job, next_op, _ = ops[idx + 1]
                        next_start = op_start_times[next_job][next_op]
                    else:
                        next_start = makespan
                    if lst[job][op] > next_start - duration[job][op]:
                        lst[job][op] = next_start - duration[job][op]
                        changed = True
        return lst
    count = 0
    it = 0
    roll_count = 0
    find_better_in_this_iteration = True
    max_stay = 10
    stay = 0
    while it <= max_iterations:
        if (best_makespan - jsp_instance["makespan"][0]) / jsp_instance["makespan"][0] * 100 <= 1:
            break
        op_start_times = [row[:] for row in best_solution]

        # 随机选一条关键路径
        _est = [row[:] for row in op_start_times]
        # _lst = compute_lst(jsp_instance, op_start_times)
        # 1. 找到makespan对应的终点操作（即结束时间等于makespan的操作）
        makespan = max(_est[job][jsp_instance["m"]-1] + jsp_instance["duration"][job][jsp_instance["m"]-1] for job in range(jsp_instance["j"]))
        end_ops = []
        for job in range(jsp_instance["j"]):
            op = jsp_instance["m"] - 1
            if abs(_est[job][op] + jsp_instance["duration"][job][op] - makespan) < 1e-6:
                end_ops.append((job, op))
        # 2. 从某个终点操作反向随机追溯一条关键路径
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
                # 按开始时间排序
                ops_on_machine.sort(key=lambda x: _est[x[0]][x[1]])
                machine_ops_order[mach] = ops_on_machine

            while True:
                prev_candidates = []
                # 工件顺序
                if op > 0:
                    prev_job_op = (job, op-1)
                    if abs(_est[prev_job_op[0]][prev_job_op[1]] + jsp_instance["duration"][prev_job_op[0]][prev_job_op[1]] - _est[job][op]) < 1e-6:
                        prev_candidates.append(prev_job_op)
                # 机器顺序
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
        if not end_ops:
            critical_ops = []
        else:
            start_op = random.choice(end_ops)
            critical_ops = trace_critical_path(start_op)

        # print(len(critical_ops), " critical operations in iteration ", it) 1746
        if len(critical_ops) == 0:
            window_idx = random.randint(0, nopr - window_size)
            raise AssertionError("没有找到关键路径上的操作，可能是计算lst时出现了问题")
        else:
            if find_better_in_this_iteration:
                # 找到更优解后，roll_count重置为0，it增加1，继续在关键路径上寻找更优解；
                crit_op = random.choice(critical_ops)
                window_idx = sorted_ops.index(crit_op)
                # 保证窗口不会越界
                if window_idx > nopr - window_size:
                    window_idx = nopr - window_size
                roll_count = 0
                stay = 0
                it += 1
            else:
                # 如果在当前窗口附近没有找到更优解，roll_count增加1，在后一个窗口附近寻找；如果窗口越界，roll_count重置，it增加1，继续寻找更优解。
                if roll_count == 0:
                    next_window_idx = window_idx % window_size
                else:
                    next_window_idx = window_idx + window_size
                if next_window_idx > nopr - window_size:
                    roll_count = 0
                    # print(f"Iteration {it}: No improvement found in current segement")
                    it += 1
                    # stay += 1
                    # if stay >= max_stay:
                    #     print(f"Iteration {it}: No improvement found for {max_stay} stays, next")
                    #     it += 1
                    #     stay = 0
                    crit_op = random.choice(critical_ops)
                    window_idx = sorted_ops.index(crit_op)
                    # 保证窗口不会越界
                    if window_idx > nopr - window_size:
                        window_idx = nopr - window_size
                    find_better_in_this_iteration = True 
                else:
                    window_idx = next_window_idx
                    roll_count += 1
        

        ops_in_window = sorted_ops[window_idx:window_idx + window_size]

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
                # 前一个操作的结束时间
                if first_idx > 0:
                    prev_op = all_ops_sorted[first_idx - 1]
                    earliest = op_start_times[prev_op[0]][prev_op[1]] + duration[prev_op[0]][prev_op[1]]
                else:
                    earliest = 0
                # 后一个操作的开始时间
                if last_idx < len(all_ops_sorted) - 1:
                    # Debug
                    # next_op = all_ops_sorted[last_idx + 1]
                    next_op = all_ops_sorted[last_idx]
                    latest = op_start_times[next_op[0]][next_op[1]] + jsp_dataset['duration'][next_op[0]][next_op[1]]
                else:
                    # 没有后续操作，允许到很大
                    latest = 100000
                    
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

        def check_other_ops_outside_machine_avail(ops_in_window, op_start_times, duration, mch, machine_avail):
            # 检查除了ops_in_window之外的所有操作，其在对应机器上的时间段不与machine_avail重叠
            ops_in_window_set = set(ops_in_window)
            j = len(op_start_times)
            m = len(op_start_times[0])
            for job in range(j):
                for op in range(m):
                    if (job, op) in ops_in_window_set:
                        continue
                    machine = mch[job][op]
                    if machine not in machine_avail:
                        continue
                    st = op_start_times[job][op]
                    ed = st + duration[job][op]
                    avail_st, avail_ed = machine_avail[machine]
                    # 检查是否有重叠
                    if not (ed <= avail_st or st >= avail_ed):
                        raise AssertionError(
                            f"Operation ({job},{op}) on machine {machine} "
                            f"({st},{ed}) overlaps with window availability ({avail_st},{avail_ed})"
                        )
                    
        machine_avail = get_machine_window_availability(
            ops_in_window, op_start_times, duration, jsp_instance["mch"]
        )

        # check_other_ops_outside_machine_avail(
        #     ops_in_window, op_start_times, duration, jsp_instance["mch"], machine_avail
        # )

        # 2. 修改solve_window使其支持机器可用时间段约束
        def solve_window_with_machine_avail(jsp_instance, ops_in_window, op_start_times, machine_avail):
            model = cp_model.CpModel()
            j, m = jsp_instance["j"], jsp_instance["m"]
            duration = jsp_instance["duration"]
            mch = jsp_instance["mch"]

            op_vars = {}
            for job, op in ops_in_window:
                lb = op_start_times[job][op]
                ub = 10000
                machine = mch[job][op]
                # 限制操作只能在该机器的可用时间段内
                if machine in machine_avail:
                    lb = min(lb, machine_avail[machine][0])
                    ub = min(ub, machine_avail[machine][1] - duration[job][op])
                if op > 0 and (job, op-1) not in ops_in_window:
                    # 如果前一个操作不在窗口内，确保当前操作的开始时间不早于前一个操作的结束时间
                    prev_end = op_start_times[job][op-1] + duration[job][op-1]
                    lb = max(lb, prev_end)
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

            makespan = model.NewIntVar(0, 10000, 'makespan')
            for job, op in ops_in_window:
                model.Add(makespan >= op_vars[(job, op)] + duration[job][op])
            model.Minimize(makespan)

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 0.1
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = 0
            status = solver.Solve(model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                result = {}
                for job, op in ops_in_window:
                    result[(job, op)] = solver.Value(op_vars[(job, op)])
                return result
            else:
                return {}

        window_result = solve_window_with_machine_avail(jsp_instance, 
                                                        ops_in_window, 
                                                        op_start_times, 
                                                        machine_avail)
        if not window_result:
            continue  # 优化失败，跳过

        # 更新窗口内操作的开始时间
        origin_op_start_times = [row[:] for row in op_start_times]
        for (job, op), st in window_result.items():
            op_start_times[job][op] = st

        # 计算每个机器最早可用时间
        machine_est = {}
        # 1. 先找出每台机器上，ops_in_window及其之前所有操作的最大结束时间
        for ops in sorted_ops[:window_idx + window_size]:
            job, op = ops
            machine = jsp_instance["mch"][job][op]
            end_time = op_start_times[job][op] + duration[job][op]
            if machine not in machine_est:
                machine_est[machine] = end_time
            else:
                machine_est[machine] = max(machine_est[machine], end_time)

        def check_schedule_validity(ops_in_window, 
                                    op_start_times, origin_op_start_times, machine_avail,
                                    duration, mch):
            # 检查ops_in_window及其之前所有操作的调度是否合法
            checked_ops = set(ops_in_window)
            # 加入窗口前的所有操作
            for job, op in ops_in_window:
                for prev_op in range(op):
                    checked_ops.add((job, prev_op))
            checked_ops = list(checked_ops)

            # 1. 检查工件顺序约束
            for job, op in checked_ops:
                if op > 0 and (job, op-1) in checked_ops:
                    prev_end = op_start_times[job][op-1] + duration[job][op-1]
                    assert op_start_times[job][op] >= prev_end, \
                        f"Job order violated: ({job},{op-1},{op_start_times[job][op-1]}, {duration[job][op-1]})" \
                        f"->({job},{op}, {op_start_times[job][op]}, {duration[job][op]})" \
                        f"\nOrigin is {origin_op_start_times[job][op-1]} and {origin_op_start_times[job][op]}" \
                        f"\nMachine availability for machine {mch[job][op]} is {machine_avail.get(mch[job][op], 'N/A')}"

            # 2. 检查机器约束（同一机器上操作不重叠）
            machine_to_ops = {}
            for job, op in checked_ops:
                machine = mch[job][op]
                machine_to_ops.setdefault(machine, []).append((job, op))
            for machine, ops in machine_to_ops.items():
                intervals = []
                for job, op in ops:
                    st = op_start_times[job][op]
                    ed = st + duration[job][op]
                    intervals.append((st, ed, job, op))
                intervals.sort()
                for i in range(1, len(intervals)):
                    prev_ed = intervals[i-1][1]
                    cur_st = intervals[i][0]
                    assert cur_st >= prev_ed, \
                        f"Machine {machine} overlap: ({intervals[i-1][2]},{intervals[i-1][3]}) and ({intervals[i][2]},{intervals[i][3]})"

        # 调用检验函数
        # check_schedule_validity(sorted_ops[:window_idx + window_size], 
        #                         op_start_times, origin_op_start_times, machine_avail,
        #                         duration, jsp_instance["mch"])

        # 2. 重新计算ops_after_window的操作开始时间，保证顺序不变且满足机器最早可用时间
        ops_after_window = sorted_ops[window_idx + window_size:]
        for job, op in ops_after_window:
            # 工件顺序约束
            machine = jsp_instance["mch"][job][op]
            if machine not in machine_est:
                machine_est[machine] = 0
            if op > 0:
                op_start_times[job][op] = max(op_start_times[job][op-1] + duration[job][op-1], machine_est[machine])
            else:
                op_start_times[job][op] = machine_est[machine]
            # 更新该机器的最早可用时间
            machine_est[machine] = op_start_times[job][op] + duration[job][op]

        # 重新计算makespan
        makespan = max(op_start_times[job][m-1] + duration[job][m-1] for job in range(j))
        if makespan < best_makespan:
            count += 1
            best_solution = [row[:] for row in op_start_times]
            best_makespan = makespan
            # DEBUG: 仅在特定实例上打印改进信息
            print(f"Iteration {it}: Found better solution with makespan {best_makespan}, roll_count: {roll_count}")
            # print(f"Iteration {it}: Found better solution with makespan {best_makespan}")
            # 更新
            all_ops = [(job, op) for job in range(j) for op in range(m)]
            op_start_flat = {(job, op): op_start_times[job][op] for job, op in all_ops}
            sorted_ops = sorted(all_ops, key=lambda x: op_start_flat[x])
            find_better_in_this_iteration = True
        else:
            find_better_in_this_iteration = False

    print(f"LNS finished after {max_iterations} iterations, improved {count} times, best makespan: {best_makespan}")
    return best_solution, best_makespan


def plot_gantt(op_start_times, jsp_instance, title="Job Shop Gantt"):
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]

    # 机器为行，任务操作为横条
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.get_cmap("tab20").colors

    for job in range(j):
        for op in range(m):
            start = op_start_times[job][op]
            dur = duration[job][op]
            machine = mch[job][op]
            color = colors[job % len(colors)]
            ax.barh(
                machine,
                dur,
                left=start,
                height=0.8,
                color=color,
                edgecolor="black",
                alpha=0.8,
            )
            ax.text(
                start + dur / 2,
                machine,
                f"J{job}O{op}",
                va="center",
                ha="center",
                fontsize=8,
                color="white",
            )

    ax.set_ylabel("Machine")
    ax.set_xlabel("Time")
    ax.set_yticks(range(max(max(row) for row in mch) + 1))
    ax.set_yticklabels([f"M{i}" for i in range(max(max(row) for row in mch) + 1)])
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{title}.png")

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


def main():
    dataset = JSPNumpyDataset(data_dir="./benchmark/TA")
    gaps = ObjMeter()
    better_gaps = ObjMeter()
    random.seed(0)
    from time import time
    st = time()

    from pdrs import solve_instance, PDR, SPT
    pdr = PDR(priority=SPT())
    for jsp_dataset in dataset:
        # debug
        # print("调度实例：", jsp_dataset["names"], "Jobs:", jsp_dataset["j"], "Machines:", jsp_dataset["m"])
        if jsp_dataset["names"].startswith("ta0") or jsp_dataset["names"].startswith("ta1") or jsp_dataset["names"].startswith("ta2") \
            or jsp_dataset["names"].startswith("ta3") or jsp_dataset["names"].startswith("ta4") or jsp_dataset["names"].startswith("ta5") \
            or (jsp_dataset["names"].startswith("ta6")) or jsp_dataset["names"].startswith("ta70") or jsp_dataset["names"].startswith("ta8"):
            continue
        window_size = min(160, jsp_dataset["j"] * jsp_dataset["m"])
        sols, ms, times = solve_instance(jsp_dataset, pdr=pdr)
        op_start_times = convert_solution_to_start_times(sols, jsp_dataset)
        assert ms == max(op_start_times[job][jsp_dataset["m"]-1] + jsp_dataset["duration"][job][jsp_dataset["m"]-1] for job in range(jsp_dataset["j"])), \
            f"计算的makespan {ms} 与根据op_start_times计算的makespan不一致 {max(op_start_times[job][jsp_dataset['m']-1] + jsp_dataset['duration'][job][jsp_dataset['m']-1] for job in range(jsp_dataset['j']))}"
        # op_start_times, ms = rolling_horizon_cp(jsp_dataset, window_size=window_size, roll_speed=10)

        print(f"Initial solution for {jsp_dataset['names']} has makespan {ms} with window size {window_size}, best known makespan {jsp_dataset['makespan']}")
        better_solution, better_makespan = large_neiborhood_search(jsp_dataset, 
                                                                   op_start_times, 
                                                                   window_size=window_size, 
                                                                   max_iterations=500)
        # print("调度结果开始时间：")
        # for job, starts in enumerate(op_start_times):
        #     print(f"Job {job}: {starts}")
        # print("Makespan:", ms)
        gap = (ms - jsp_dataset["makespan"]) / jsp_dataset["makespan"] * 100
        better_gap = (better_makespan - jsp_dataset["makespan"]) / jsp_dataset["makespan"] * 100
        print(jsp_dataset["names"], f" Gap: {gap[0]:.2f}", f" Better Gap: {better_gap[0]:.2f}")
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

def plot():
    jsp_dataset = {
        "j": n_jobs,
        "m": n_mchs,
        "duration": times,
        "mch": machines,
        "makespan": makespan,
        "orders": orders,
        "names": data_file
    }
    plot_gantt(op_start_times, jsp_dataset, 
               title=f"(makespan={makespan:.1f})")