import numpy as np
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
from dataset import JSPNumpyDataset, ObjMeter
import random
import multiprocessing
from collections import defaultdict
from time import time


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

def solve_window_with_machine_avail(jsp_instance, ops_in_window, op_start_times, machine_avail, num_works=1):
    # st = time()
    model = cp_model.CpModel()
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]
    horizon = sum(op_start_times[job][op] for job in range(j) for op in range(m))

    op_vars = {}
    for job, op in ops_in_window:
        machine = mch[job][op]
        # lb = machine_avail[machine][0]
        # ub = machine_avail[machine][1] - duration[job][op]
        # machine = mch[job][op]
        # # 限制操作只能在该机器的可用时间段内
        # # if machine in machine_avail:
        # #     lb = min(lb, machine_avail[machine][0])
        # #     ub = min(ub, machine_avail[machine][1] - duration[job][op])
        # if op > 0 and (job, op-1) not in ops_in_window:
        #     # 如果前一个操作不在窗口内，确保当前操作的开始时间不早于前一个操作的结束时间
        #     prev_end = op_start_times[job][op-1] + duration[job][op-1]
        #     lb = max(lb, prev_end)
        lb = 0
        ub = horizon
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
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
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

def get_machine_window_availability(nbh_in_window, op_start_times, duration, mch):
    # 返回: {machine: (earliest_available, latest_finish)}
    machine_ops = {}
    for job, op in nbh_in_window:
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
        if first_idx > 0:
            prev_op = all_ops_sorted[first_idx - 1]
            earliest = op_start_times[prev_op[0]][prev_op[1]] + duration[prev_op[0]][prev_op[1]]
        else:
            earliest = 0
        # 当前操作最晚结束时间
        if last_idx < len(all_ops_sorted) - 1:
            # Debug
            # next_op = all_ops_sorted[last_idx + 1]
            next_op = all_ops_sorted[last_idx]
            latest = op_start_times[next_op[0]][next_op[1]] + jsp_dataset['duration'][next_op[0]][next_op[1]]
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

def build_successors(op_start_times, duration, mch):
    j = len(op_start_times)
    m = len(op_start_times[0])
    succ = {}
    for job in range(j):
        for op in range(m):
            succ[(job, op)] = []
    for job in range(j):
        for op in range(m):
            start = op_start_times[job][op]
            machine = mch[job][op]
            end = start + duration[job][op]
            for job2 in range(j):
                for op2 in range(m):
                    if job2 == job and op2 == op:
                        continue
                    if mch[job2][op2] == machine:
                        start2 = op_start_times[job2][op2]
                        if start2 >= end:
                            succ[(job, op)].append((job2, op2))
                    if job2 == job and op2 == op + 1:
                        succ[(job, op)].append((job2, op2))
    return succ

def extract_critical_path(op_start_times, duration, mch):
    j = len(op_start_times)
    m = len(op_start_times[0])
    finish = np.array(op_start_times) + np.array(duration)
    machine_seq = [[] for _ in range(m)]
    for m_idx in range(m):
        rows, cols = np.where(np.array(mch) == m_idx)
        for row, col in zip(rows, cols):
            machine_seq[m_idx].append((row, col))
        machine_seq[m_idx].sort(key=lambda x: op_start_times[x[0]][x[1]])

    makespan = max(finish[job][m-1] for job in range(j))
    # 反向找到一个达成 makespan 的操作
    endpoint = random.choices([(job, m-1) for job in range(j) if finish[job][m-1] == makespan], k=1)[0]
    path = [endpoint]
    cur = endpoint
    while True:
        job, op = cur
        candidates = []
        # Job Precedence
        if op > 0:
            prev = (job, op - 1)
            candidates.append(prev)
        # Machine Precedence
        machine = mch[job][op]
        for i in range(len(machine_seq[machine])):
            if machine_seq[machine][i] == cur:
                if i > 0:
                    candidates.append(machine_seq[machine][i-1])
                break
        if not candidates:
            break
        # 选择最晚结束的前驱
        sorted_candidates = sorted(candidates, key=lambda x: finish[x[0]][x[1]], reverse=True)
        prev = sorted_candidates[0]
        if finish[prev] < op_start_times[job][op]:
            break
        path.append(prev)
        cur = prev
    path.reverse()
    return path

def split_critical_blocks(crit_path, mch):
    blocks = []
    if not crit_path:
        return blocks
    cur_block = [crit_path[0]]
    for prev, cur in zip(crit_path, crit_path[1:]):
        if mch[prev[0]][prev[1]] == mch[cur[0]][cur[1]]:
            cur_block.append(cur)
        else:
            if len(cur_block) >= 2:
                blocks.append(cur_block)
            cur_block = [cur]
    if len(cur_block) >= 2:
        blocks.append(cur_block)
    return blocks

def build_window_from_block_by_start_time(block, 
                                        op_start_times, 
                                        duration, 
                                        mch, 
                                        block_start_idx, 
                                        max_length=4, 
                                        cp_size=200):
    j = len(op_start_times)
    m = len(op_start_times[0])
    window = set(block)
    makespan = max(op_start_times[job][m-1] + duration[job][m-1] for job in range(j))
    # sort machine operations by start time
    machine_seq = [[] for _ in range(m)]
    for m_idx in range(m):
        rows, cols = np.where(np.array(mch) == m_idx)
        for row, col in zip(rows, cols):
            machine_seq[m_idx].append((row, col))
        machine_seq[m_idx].sort(key=lambda x: op_start_times[x[0]][x[1]])

    job, op = block[block_start_idx]
    job_c, op_c = job, op + 1
    job_d, op_d = block[block_start_idx + 1]
    all_ops = [(job, op) for job in range(j) for op in range(m)]
    sorted_ops = sorted(all_ops, key=lambda x: op_start_times[x[0]][x[1]])
    start_idx = -1
    end_idx = -1
    cur_idx = 0
    for job1, op1 in sorted_ops:
        if job1 == job and op1 == op:
            start_idx = cur_idx
        if op < m - 1 and job1 == job_c and op1 == op_c:
            end_idx = cur_idx
        if job1 == job_d and op1 == op_d:
            end_idx == cur_idx
        cur_idx += 1
    assert end_idx > start_idx, f"start idx must less than end idx, but got {start_idx} -> {end_idx}"
    ops_in_window = sorted_ops[start_idx:min(end_idx + 1, start_idx + cp_size)]

    length = 1
    while len(ops_in_window) < cp_size and length < max_length:
        if block_start_idx + length >= len(block) - 1:
            break
        job, op = block[block_start_idx + length]
        job_c, op_c = job, op + 1
        job_d, op_d = block[block_start_idx + length + 1]
        cur_idx = start_idx
        for job1, op1 in sorted_ops:
            if op < m - 1 and job1 == job_c and op1 == op_c:
                end_idx = cur_idx
            if job1 == job_d and op1 == op_d:
                end_idx == cur_idx
            cur_idx += 1
        assert end_idx > start_idx, f"start idx must less than end idx, but got {start_idx} -> {end_idx}"
        ops_in_window = sorted_ops[start_idx:min(end_idx + 1, start_idx + cp_size)]
        length += 1

    return ops_in_window

def build_window_from_block(block, op_start_times, duration, mch, block_start_idx, cp_size=200):
    j = len(op_start_times)
    m = len(op_start_times[0])
    window = set(block)
    makespan = max(op_start_times[job][m-1] + duration[job][m-1] for job in range(j))
    # sort machine operations by start time
    machine_seq = [[] for _ in range(m)]
    for m_idx in range(m):
        rows, cols = np.where(np.array(mch) == m_idx)
        for row, col in zip(rows, cols):
            machine_seq[m_idx].append((row, col))
        machine_seq[m_idx].sort(key=lambda x: op_start_times[x[0]][x[1]])

    # next op
    ops_machine_avail = {}
    machine_avail_set = {}
    window = set()

    job, op = block[block_start_idx]
    job_, op_ = block[block_start_idx + 1]
    window.add((job, op))
    window.add((job_, op_))
    if op < m - 1:
        op_next = (job, op + 1)
        machine_next = mch[job][op + 1]
        machine_next_idx = 0
        for idx in range(len(machine_seq[machine_next])):
            if machine_seq[machine_next][idx] == (job, op):
                break
            machine_next_idx += 1
        before_idx, after_idx = machine_next_idx - 1, machine_next_idx + 1
        if before_idx >= 0:
            op_before = machine_seq[machine_next][before_idx]
            op_before_ed = op_start_times[op_before[0]][op_before[1]] + duration[op_before[0]][op_before[1]]
        else:
            op_before_ed = 0
        if after_idx < len(machine_seq[machine_next]):
            op_after = machine_seq[machine_next][after_idx]
            op_after_st = op_start_times[op_after[0]][op_after[1]]
        else:
            op_after_st = makespan
        ops_machine_avail[op_next] = (machine_next, op_before_ed, op_after_st)
        if machine_next not in machine_avail_set.keys():
            machine_avail_set[machine_next] = []
        machine_avail_set[machine_next].append((op_next, op_before_ed, op_after_st))

    if op_ < m - 1:
        op_next = (job_, op_ + 1)
        machine_next = mch[job_][op_ + 1]
        machine_next_idx = 0
        for idx in range(len(machine_seq[machine_next])):
            if machine_seq[machine_next][idx] == (job_, op_):
                break
            machine_next_idx += 1
        before_idx, after_idx = machine_next_idx - 1, machine_next_idx + 1
        if before_idx >= 0:
            op_before = machine_seq[machine_next][before_idx]
            op_before_ed = op_start_times[op_before[0]][op_before[1]] + duration[op_before[0]][op_before[1]]
        else:
            op_before_ed = 0
        if after_idx < len(machine_seq[machine_next]):
            op_after = machine_seq[machine_next][after_idx]
            op_after_st = op_start_times[op_after[0]][op_after[1]]
        else:
            op_after_st = makespan
        ops_machine_avail[op_next] = (machine_next, op_before_ed, op_after_st)
        if machine_next not in machine_avail_set.keys():
            machine_avail_set[machine_next] = []
        machine_avail_set[machine_next].append((op_next, op_before_ed, op_after_st))

    idx = 2
    while len(window) + len(ops_machine_avail.keys()) <= cp_size:
        if block_start_idx + idx >= len(block):
            break
        job_, op_ = block[block_start_idx + idx]
        window.add((job_, op_))
        if op_ < m - 1:
            op_next = (job_, op_ + 1)
            machine_next = mch[job_][op_ + 1]
            machine_next_idx = 0
            for idx in range(len(machine_seq[machine_next])):
                if machine_seq[machine_next][idx] == (job_, op_):
                    break
                machine_next_idx += 1
            before_idx, after_idx = machine_next_idx - 1, machine_next_idx + 1
            if before_idx >= 0:
                op_before = machine_seq[machine_next][before_idx]
                op_before_ed = op_start_times[op_before[0]][op_before[1]] + duration[op_before[0]][op_before[1]]
            else:
                op_before_ed = 0
            if after_idx < len(machine_seq[machine_next]):
                op_after = machine_seq[machine_next][after_idx]
                op_after_st = op_start_times[op_after[0]][op_after[1]]
            else:
                op_after_st = makespan
            if machine_next not in machine_avail_set.keys():
                machine_avail_set[machine_next] = []
            machine_avail_set[machine_next].append((op_next, op_before_ed, op_after_st))
            ops_machine_avail[op_next] = (machine_next, op_before_ed, op_after_st)
    
    return ops_machine_avail, window

def build_machine_avail_for_window(window_ops, op_start_times, duration, mch):
    machine_avail = {}
    all_ops_on_machine = {}
    j = len(op_start_times)
    m = len(op_start_times[0])
    for job in range(j):
        for op in range(m):
            machine = mch[job][op]
            all_ops_on_machine.setdefault(machine, []).append((job, op))
    for machine in set(mch[job][op] for job, op in window_ops):
        ops = sorted(all_ops_on_machine[machine], key=lambda x: op_start_times[x[0]][x[1]])
        idxs = [i for i, op in enumerate(ops) if op in window_ops]
        first = idxs[0]
        last = idxs[-1]
        if first > 0:
            prev = ops[first - 1]
            earliest = op_start_times[prev[0]][prev[1]] + duration[prev[0]][prev[1]]
        else:
            earliest = 0
        if last < len(ops) - 1:
            nxt = ops[last + 1]
            latest = op_start_times[nxt[0]][nxt[1]]
        else:
            latest = max(
                op_start_times[job][m-1] + duration[job][m-1]
                for job in range(j)
            )
        if latest < earliest:
            raise ValueError(f"机器{machine}的可用区间不合理: earliest={earliest}, latest={latest}")
        machine_avail[machine] = (earliest, latest)
    return machine_avail

def apply_window_result(jsp_instance, op_start_times, window_ops, result):
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]

    all_ops = [(job, op) for job in range(j) for op in range(m)]
    sorted_ops = sorted(all_ops, key=lambda x: op_start_times[x[0]][x[1]])

    for (job, op), st in result.items():
        original_max_time = max(original_max_time, op_start_times[job][op] + duration[job][op])
        # print(f"For job ({job}, {op}), original start time: {op_start_times[job][op]}, new start time: {st}")
        op_start_times[job][op] = st

    # 计算每个机器最早可用时间
    machine_est = {}
    # 1. 先找出每台机器上，ops_in_window及其之前所有操作的最大结束时间
    # if debug:
    #     ops_in_window.sort(key=lambda x: window_result[(x[0], x[1])])
    #     sorted_ops[window_idx:window_idx + window_size] = ops_in_window

    for ops in sorted_ops[:window_idx + window_size]:
        job, op = ops
        machine = jsp_instance["mch"][job][op]
        end_time = op_start_times[job][op] + duration[job][op]
        if machine not in machine_est:
            machine_est[machine] = end_time
        else:
            machine_est[machine] = max(machine_est[machine], end_time)

    after_max_time = max(machine_est[i] for i in machine_est.keys())
    # assert after_max_time <= original_max_time, f"机器最早可用时间不应该变晚: {after_max_time} vs {original_max_time}"
    # print(f"After window update, max machine earliest time is {after_max_time}, original max time was {original_max_time}")
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
    
    return op_start_times, makespan

def nbh_large_neiborhood_search(jsp_instance, op_start_times, max_iterations=1000, debug=None):
    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]

    current_start = [row.copy() for row in op_start_times]
    current_makespan = max(
        current_start[job][m-1] + duration[job][m-1]
        for job in range(j)
    )

    for iteration in range(max_iterations):
        crit_path = extract_critical_path(current_start, duration, mch)
        # debug
        # for job, op in crit_path:
        #     print(f"({job}, {op}), op_start={current_start[job][op]}, duration={duration[job][op]}, machine={mch[job][op]}")
        blocks = split_critical_blocks(crit_path, mch)
        for i, block in enumerate(blocks):
            print(f"Block {i} length is {len(block)}, machine={mch[block[0][0]][block[0][1]]}, "
                  f"start time is {op_start_times[block[0][0]][block[0][1]]}, "
                  f"end time is {op_start_times[block[-1][0]][block[-1][1]] + duration[block[-1][0]][block[-1][1]]}")
        if not blocks:
            break

        windows = []
        for block in blocks:
            block_start_idx = random.choices(range(len(block) - 1), k=1)[0]
            window = build_window_from_block_by_start_time(block, 
                                                           current_start, 
                                                           duration, 
                                                           mch, 
                                                           block_start_idx)
            machine_avail = build_machine_avail_for_window(window, current_start, duration, mch)
            for machine, (earliest, latest) in machine_avail.items():
                print(f"  Machine {machine} availability: earliest={earliest}, latest={latest}")
            windows.append((window, machine_avail))
        if not windows:
            break

        window_idx = random.randrange(len(windows))
        window_ops, machine_avail = windows[window_idx]
        result = solve_window_with_machine_avail(jsp_instance, window_ops, current_start, machine_avail)
        if result is None:
            continue

        candidate_start = apply_window_result(current_start, window_ops, result)
        candidate_makespan = max(
            candidate_start[job][m-1] + duration[job][m-1]
            for job in range(j)
        )
        if candidate_makespan < current_makespan:
            current_start = candidate_start
            current_makespan = candidate_makespan
            if debug:
                print(f"[LNS] iter={iteration} better makespan={current_makespan}")
        # 否则丢弃该窗口结果，继续下一次迭代

    return current_start, current_makespan


if __name__ == "__main__":
    dataset = JSPNumpyDataset(data_dir="./benchmark/TA")
    gaps = ObjMeter()
    better_gaps = ObjMeter()
    random.seed(0)
    st = time()

    from pdrs import solve_instance, PDR, SPT

    pdr = PDR(priority=SPT())

    for jsp_dataset in dataset:
        if jsp_dataset["j"] <= 50:
            continue
        sols, ms, times = solve_instance(jsp_dataset, pdr=pdr)
        
        op_start_times = convert_solution_to_start_times(sols, jsp_dataset)
        assert ms == max(op_start_times[job][jsp_dataset["m"]-1] + jsp_dataset["duration"][job][jsp_dataset["m"]-1] for job in range(jsp_dataset["j"])), \
            f"计算的makespan {ms} 与根据op_start_times计算的makespan不一致 {max(op_start_times[job][jsp_dataset['m']-1] + jsp_dataset['duration'][job][jsp_dataset['m']-1] for job in range(jsp_dataset['j']))}"
        
        print(f"Initial solution for {jsp_dataset['names']} has makespan {ms}")

        better_solution, better_makespan = nbh_large_neiborhood_search(jsp_dataset, 
                                                                       op_start_times,
                                                                       max_iterations=5000,
                                                                       debug="suffix_windows")

        gap = ms 
        better_gap = better_makespan

        print(jsp_dataset["names"], f" Gap: {gap:.2f}", f" Better Gap: {better_gap:.2f}")
        gaps.update(jsp_dataset, gap)
        better_gaps.update(jsp_dataset, better_gap)

    print("Overall Gaps:", gaps.avg)
    print(gaps)
    print("Overall Better Gaps:", better_gaps.avg)
    print(better_gaps)
    end = time()
    print(f"Total time: {end - st:.2f} seconds")