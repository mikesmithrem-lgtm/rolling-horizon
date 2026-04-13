import os
import random
from collections import defaultdict
from time import time


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

        first_row = lines[0].split()
        n_jobs, n_mchs = int(first_row[0]), int(first_row[1])

        times = []
        machines = []
        for i in range(1, n_jobs + 1):
            row = list(map(int, lines[i].split()))
            machines.append(row[0::2])
            times.append(row[1::2])

        makespan = -1
        orders = []
        if len(lines) > n_jobs + 1:
            makespan = int(lines[n_jobs + 1])
            for i in range(n_jobs + 2, len(lines)):
                orders.append(list(map(int, lines[i].split())))

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


class Priority:
    name = "Priority"
    tau = 0.0001

    def __init__(self):
        self._tie = 0.0

    def __call__(self, job, op, costs):
        raise NotImplementedError()


class SPT(Priority):
    name = "SPT"

    def __call__(self, job, op, costs):
        self._tie += self.tau
        return -costs[job][op] - self._tie


class PDR:
    def __init__(self, priority, top=3, eps=0.001):
        self._eps = eps
        self.priority = priority
        self.name = priority.name
        self.top = top

    def __call__(self, num_j, num_m, costs, machines, randomized=False):
        ops = [[0, machines[job][0]] for job in range(num_j)]
        prio = [self.priority(job, 0, costs) for job in range(num_j)]

        curr_time = -1
        mac_time = [0] * num_m
        sol = [[] for _ in range(num_m)]
        times = [[] for _ in range(num_j)]
        active_job = num_j

        while active_job > 0:
            job_order = sorted(
                range(num_j),
                key=lambda job: prio[job],
                reverse=True,
            )
            job_order = [job for job in job_order if prio[job] != -float("inf")]

            if randomized:
                rand_order = []
                top_k = job_order[:self.top]
                remaining = job_order[self.top:]
                while top_k:
                    pick_idx = random.randint(0, len(top_k) - 1)
                    rand_order.append(top_k.pop(pick_idx))
                    if remaining:
                        top_k.append(remaining.pop(0))
                job_order = rand_order

            future_times = [ct for ct in mac_time if ct > curr_time]
            if not future_times:
                break
            curr_time = min(future_times)

            for job in job_order:
                op, machine = ops[job]
                min_st = max(mac_time[machine], 0 if op == 0 else times[job][-1])

                if min_st - self._eps < curr_time:
                    if op < num_m - 1:
                        ops[job][0] = op + 1
                        ops[job][1] = machines[job][op + 1]
                        prio[job] = self.priority(job, op + 1, costs)
                    else:
                        active_job -= 1
                        prio[job] = -float("inf")

                    mac_time[machine] = min_st + costs[job][op]
                    times[job].append(mac_time[machine])
                    sol[machine].append(job * num_m + op)

        return sol, times


def solve_instance(ins, pdr, beta=1, seed=1234):
    costs = ins["duration"]
    machines = ins["mch"]
    st = time()

    sol, times = pdr(ins["j"], ins["m"], costs, machines)
    best_sol = sol
    best_ms = max(job_times[-1] for job_times in times)

    if beta > 1:
        random.seed(seed)
        for _ in range(beta - 1):
            sol, times = pdr(ins["j"], ins["m"], costs, machines, randomized=True)
            ms = max(job_times[-1] for job_times in times)
            if ms < best_ms:
                best_sol = sol
                best_ms = ms

    et = time() - st
    return best_sol, best_ms, et


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
    from ortools.sat.python import cp_model

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
            ub = min(ub, after_start)
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

def large_neiborhood_search(
    jsp_instance: dict, 
    est: list, 
    use_multi_window=False,
    window_size=10,
    max_iterations=100, 
    debug="all_windows", 
    cp_mode=False):

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
    
    count = 0
    it = 0
    roll_count = 0
    find_better_in_this_iteration = True
    while it <= max_iterations:
        # if (best_makespan - jsp_instance["makespan"][0]) / jsp_instance["makespan"][0] * 100 <= 1:
        #     break
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
            crit_op = random.choice(critical_ops)
            window_idx = sorted_ops.index(crit_op)
            # 保证窗口不会越界
            if window_idx > nopr - window_size:
                window_idx = nopr - window_size
            it += 1
        

        ops_in_window = sorted_ops[window_idx:window_idx + window_size]
        if cp_mode:
            assert max_iterations == 1, f"CP mode must make iteration as 1"
            assert debug == 'single_windows', f"CP mode debug must be single window"
            ops_in_window = sorted_ops[:]
            machine_avail = get_machine_window_availability(
                    ops_in_window, op_start_times, duration, jsp_instance["mch"]
                )
                # st = time()
            window_result = solve_window_with_machine_avail(jsp_instance, 
                                                            ops_in_window, 
                                                            op_start_times, 
                                                            machine_avail)
            for (job, op), st in window_result.items():
                op_start_times[job][op] = st
            makespan = max(op_start_times[job][m-1] + duration[job][m-1] for job in range(j))
            return op_start_times, makespan
            

        if use_multi_window:
            if debug == 'all_windows':
                multi_ops_in_windows = []
                multi_machine_avail = []
                windows_data = []
                start_idx = window_idx % window_size
                total_windows = nopr // window_size
                if start_idx > 0:
                    first_ops_in_window = sorted_ops[0:start_idx]
                    # print("first ops in window:", len(first_ops_in_window))
                    multi_ops_in_windows.append(first_ops_in_window)
                assert total_windows > 0, f"窗口大小 {window_size} 太大，无法划分出完整的窗口，nopr: {nopr}"
                if total_windows > 0:
                    window_list = [sorted_ops[start_idx + i*window_size:start_idx + (i+1)*window_size] for i in range(0, total_windows)]
                    # print("total ops in windows_list:", sum(len(w) for w in window_list))
                    multi_ops_in_windows.extend(window_list)
                if start_idx + total_windows * window_size < nopr:
                    last_ops_in_window = sorted_ops[start_idx + total_windows * window_size:]
                    # print("last ops in window:", len(last_ops_in_window))
                    multi_ops_in_windows.append(last_ops_in_window)
                assert sum(len(w) for w in multi_ops_in_windows) == nopr, f"多窗口划分错误，操作总数不匹配: {sum(len(w) for w in multi_ops_in_windows)} vs {nopr}"
                for i in range(len(multi_ops_in_windows)):
                    ops_in_window = multi_ops_in_windows[i]
                    machine_avail = get_machine_window_availability(
                        ops_in_window, op_start_times, duration, jsp_instance["mch"]
                    )
                    multi_machine_avail.append(machine_avail)
                    windows_data.append((ops_in_window, machine_avail, i))

                if not check_machine_feasibility(multi_machine_avail):
                    raise ValueError("窗口内机器可用时间不合理，可能是计算机器可用时间时出现了问题")

                results = solve_multiple_windows_parallel(jsp_instance, op_start_times, windows_data)
                results.sort(key=lambda x: x[1])  # 按窗口索引排序，保证结果顺序与窗口顺序一致
                assert len(results) == len(multi_ops_in_windows), f"求解结果数量 {len(results)} 与窗口数量 {len(multi_ops_in_windows)} 不匹配"

                # 3. 更新解析结果到op_start_times
                orders = [[] for _ in range(m)]
                for (result, i) in results:
                    if not result:
                        raise ValueError(f"窗口 {i} 求解失败，可能是模型无解或者求解时间过短导致的。")
                        continue  # 跳过求解失败的窗口
                    ops_in_window = multi_ops_in_windows[i]
                    ops_in_window.sort(key=lambda x: result[x[0], x[1]]) 
                    for (job, op), st in result.items():
                        machine = jsp_instance["mch"][job][op]
                        orders[machine].append((job, op))
                assert all([len(o) == j for o in orders]), f"某台机器的操作数量不正确: {[len(o) for o in orders]} vs {j}"
            elif debug == 'prefix_windows':
                multi_ops_in_windows = []
                multi_machine_avail = []
                windows_data = []
                start_idx = window_idx % window_size
                total_windows = window_idx // window_size
                multi_ops_in_windows.append(sorted_ops[0:start_idx])
                if total_windows > 0:
                    window_list = [sorted_ops[start_idx + i*window_size:start_idx + (i+1)*window_size] for i in range(0, total_windows)]
                    multi_ops_in_windows.extend(window_list)
                assert sum(len(w) for w in multi_ops_in_windows) == window_idx, f"多窗口划分错误，操作总数不匹配: {sum(len(w) for w in multi_ops_in_windows)} vs {window_idx}"

                for i in range(len(multi_ops_in_windows)):
                    ops_in_window = multi_ops_in_windows[i]
                    machine_avail = get_machine_window_availability(
                        ops_in_window, op_start_times, duration, jsp_instance["mch"]
                    )
                    multi_machine_avail.append(machine_avail)
                    windows_data.append((ops_in_window, machine_avail, i))
                
                if not check_machine_feasibility(multi_machine_avail):
                    raise ValueError("窗口内机器可用时间不合理，可能是计算机器可用时间时出现了问题")
                
                # st = time()
                # temp = solve_window_with_machine_avail(jsp_instance, multi_ops_in_windows[0], op_start_times, multi_machine_avail[0])
                # ed = time()
                # print(f"Single window solving took {ed - st:.2f} seconds")

                # st = time()
                results = solve_multiple_windows_parallel(jsp_instance, op_start_times, windows_data)
                # ed = time()
                # print(f"Parallel solving for multiple windows took {ed - st:.2f} seconds")
                # import sys
                # sys.exit(0)
                results.sort(key=lambda x: x[1])  # 按窗口索引排序，保证结果顺序与窗口顺序一致

                orders = [[] for _ in range(m)]
                for i in range(len(multi_ops_in_windows)):
                    ops_in_window = multi_ops_in_windows[i]
                    machine_avail = multi_machine_avail[i]
                    result = results[i][0]
                    if not result:
                        ops_in_window.sort(key=lambda x: op_start_times[x[0]][x[1]])  # 按原始开始时间排序
                    else:
                        ops_in_window.sort(key=lambda x: result[x[0], x[1]])
                    for job, op in ops_in_window:
                        machine = jsp_instance["mch"][job][op]
                        orders[machine].append((job, op))

                for ops in sorted_ops[window_idx:]:
                    job, op = ops
                    machine = jsp_instance["mch"][job][op]
                    orders[machine].append((job, op))
                assert all([len(o) == j for o in orders]), f"某台机器的操作数量不正确: {[len(o) for o in orders]} vs {j}"
            else:
                multi_ops_in_windows = []
                multi_machine_avail = []
                windows_data = []
                start_idx = window_idx 
                total_windows = (nopr - start_idx) // window_size
                if total_windows == 0:
                    # only one window
                    multi_ops_in_windows.append(sorted_ops[start_idx:])
                else:
                    window_list = [sorted_ops[start_idx + i*window_size:start_idx + (i+1)*window_size] for i in range(0, total_windows)]
                    multi_ops_in_windows.extend(window_list)
                    if start_idx + total_windows * window_size < nopr:
                        last_ops_in_window = sorted_ops[start_idx + total_windows * window_size:]
                        multi_ops_in_windows.append(last_ops_in_window)
                assert sum(len(w) for w in multi_ops_in_windows) == nopr - start_idx, f"多窗口划分错误，操作总数不匹配: {sum(len(w) for w in multi_ops_in_windows)} vs {nopr - start_idx}"

                for i in range(len(multi_ops_in_windows)):
                    ops_in_window = multi_ops_in_windows[i]
                    machine_avail = get_machine_window_availability(
                        ops_in_window, op_start_times, duration, jsp_instance["mch"]
                    )
                    multi_machine_avail.append(machine_avail)
                    windows_data.append((ops_in_window, machine_avail, i))
                
                if not check_machine_feasibility(multi_machine_avail):
                    raise ValueError("窗口内机器可用时间不合理，可能是计算机器可用时间时出现了问题")
                
                # st = time()
                # temp = solve_window_with_machine_avail(jsp_instance, multi_ops_in_windows[0], op_start_times, multi_machine_avail[0])
                # ed = time()
                # print(f"Single window solving took {ed - st:.2f} seconds")

                # st = time()
                results = solve_multiple_windows_parallel(jsp_instance, op_start_times, windows_data)
                # ed = time()
                # print(f"Parallel solving for multiple windows took {ed - st:.2f} seconds")
                # import sys
                # sys.exit(0)
                results.sort(key=lambda x: x[1])  # 按窗口索引排序，保证结果顺序与窗口顺序一致

                orders = [[] for _ in range(m)]
                # 对于前面的操作保证顺序不变，直接加入orders
                for ops in sorted_ops[:start_idx]:
                    job, op = ops
                    machine = jsp_instance["mch"][job][op]
                    orders[machine].append((job, op))
                for i in range(len(multi_ops_in_windows)):
                    ops_in_window = multi_ops_in_windows[i]
                    machine_avail = multi_machine_avail[i]
                    result = results[i][0]
                    if not result:
                        ops_in_window.sort(key=lambda x: op_start_times[x[0]][x[1]])  # 按原始开始时间排序
                    else:
                        ops_in_window.sort(key=lambda x: result[x[0], x[1]])
                    for job, op in ops_in_window:
                        machine = jsp_instance["mch"][job][op]
                        orders[machine].append((job, op))
                assert all([len(o) == j for o in orders]), f"某台机器的操作数量不正确: {[len(o) for o in orders]} vs {j}"
            # 实现根据orders更新op_start_times的逻辑，确保工件顺序约束和机器约束都得到满足
            # 这里需要一个调度逻辑，按照orders中每台机器的操作顺序，计算每个操作的开始时间，并更新op_start_times
            machine_ready = [0] * m
            job_ready = [0] * j
            order_idx = [0] * m  # 记录每台机器当前调度到哪个操作
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
                    # 检查工件前一个操作是否完成
                    if op > 0 and (job, op-1) not in scheduled_ops:
                        continue
                    # 可以调度这个操作了
                    start_time = max(job_ready[job], machine_ready[machine])
                    op_start_times[job][op] = start_time
                    end_time = start_time + duration[job][op]
                    job_ready[job] = end_time
                    machine_ready[machine] = end_time
                    scheduled_ops.add((job, op))
                    order_idx[machine] += 1
                    progress = True
                if not progress:
                    raise ValueError("调度过程中没有进展，可能是orders中的操作顺序不合理，导致死锁")
            makespan = max(op_start_times[job][m-1] + duration[job][m-1] for job in range(j))
            # print(f"Iteration {it}: Found better solution with makespan {makespan} (improvement: {(best_makespan - makespan) / best_makespan * 100:.2f}%)")
        else:
            if debug == 'all_windows':
                multi_ops_in_windows = []
                multi_machine_avail = []
                windows_data = []
                start_idx = window_idx % window_size
                total_windows = nopr // window_size
                if start_idx > 0:
                    first_ops_in_window = sorted_ops[0:start_idx]
                    # print("first ops in window:", len(first_ops_in_window))
                    multi_ops_in_windows.append(first_ops_in_window)
                assert total_windows > 0, f"窗口大小 {window_size} 太大，无法划分出完整的窗口，nopr: {nopr}"
                if total_windows > 0:
                    window_list = [sorted_ops[start_idx + i*window_size:start_idx + (i+1)*window_size] for i in range(0, total_windows)]
                    # print("total ops in windows_list:", sum(len(w) for w in window_list))
                    multi_ops_in_windows.extend(window_list)
                if start_idx + total_windows * window_size < nopr:
                    last_ops_in_window = sorted_ops[start_idx + total_windows * window_size:]
                    # print("last ops in window:", len(last_ops_in_window))
                    multi_ops_in_windows.append(last_ops_in_window)
                assert sum(len(w) for w in multi_ops_in_windows) == nopr, f"多窗口划分错误，操作总数不匹配: {sum(len(w) for w in multi_ops_in_windows)} vs {nopr}"
                for i in range(len(multi_ops_in_windows)):
                    ops_in_window = multi_ops_in_windows[i]
                    machine_avail = get_machine_window_availability(
                        ops_in_window, op_start_times, duration, jsp_instance["mch"]
                    )
                    multi_machine_avail.append(machine_avail)
                orders = [[] for _ in range(m)]
                # print(f"Total Windows in Debug: {len(multi_ops_in_windows)}")
                for i in range(len(multi_ops_in_windows)):
                    ops_in_window = multi_ops_in_windows[i]
                    machine_avail = multi_machine_avail[i]
                    result = solve_window_with_machine_avail(jsp_instance, ops_in_window, op_start_times, machine_avail)
                    ops_in_window.sort(key=lambda x: result[x[0], x[1]])
                    for job, op in ops_in_window:
                        machine = jsp_instance["mch"][job][op]
                        orders[machine].append((job, op))
                assert all([len(o) == j for o in orders]), f"某台机器的操作数量不正确: {[len(o) for o in orders]} vs {j}"
                # 实现根据orders更新op_start_times的逻辑，确保工件顺序约束和机器约束都得到满足
                # 这里需要一个调度逻辑，按照orders中每台机器的操作顺序，计算每个操作的开始时间，并更新op_start_times
                machine_ready = [0] * m
                job_ready = [0] * j
                order_idx = [0] * m  # 记录每台机器当前调度到哪个操作
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
                        # 检查工件前一个操作是否完成
                        if op > 0 and (job, op-1) not in scheduled_ops:
                            continue
                        # 可以调度这个操作了
                        start_time = max(job_ready[job], machine_ready[machine])
                        op_start_times[job][op] = start_time
                        end_time = start_time + duration[job][op]
                        job_ready[job] = end_time
                        machine_ready[machine] = end_time
                        scheduled_ops.add((job, op))
                        order_idx[machine] += 1
                        progress = True
                    if not progress:
                        raise ValueError("调度过程中没有进展，可能是orders中的操作顺序不合理，导致死锁")
                makespan = max(op_start_times[job][m-1] + duration[job][m-1] for job in range(j))
            elif debug == 'suffix_windows':
                multi_ops_in_windows = []
                multi_machine_avail = []
                windows_data = []
                start_idx = window_idx 
                total_windows = (nopr - start_idx) // window_size
                if total_windows == 0:
                    # only one window
                    multi_ops_in_windows.append(sorted_ops[start_idx:])
                else:
                    window_list = [sorted_ops[start_idx + i*window_size:start_idx + (i+1)*window_size] for i in range(0, total_windows)]
                    multi_ops_in_windows.extend(window_list)
                    if start_idx + total_windows * window_size < nopr:
                        last_ops_in_window = sorted_ops[start_idx + total_windows * window_size:]
                        multi_ops_in_windows.append(last_ops_in_window)
                assert sum(len(w) for w in multi_ops_in_windows) == nopr - start_idx, f"多窗口划分错误，操作总数不匹配: {sum(len(w) for w in multi_ops_in_windows)} vs {nopr - start_idx}"

                for i in range(len(multi_ops_in_windows)):
                    ops_in_window = multi_ops_in_windows[i]
                    machine_avail = get_machine_window_availability(
                        ops_in_window, op_start_times, duration, jsp_instance["mch"]
                    )
                    multi_machine_avail.append(machine_avail)
                orders = [[] for _ in range(m)]
                # 对于前面的操作保证顺序不变，直接加入orders
                for ops in sorted_ops[:start_idx]:
                    job, op = ops
                    machine = jsp_instance["mch"][job][op]
                    orders[machine].append((job, op))
                for i in range(len(multi_ops_in_windows)):
                    ops_in_window = multi_ops_in_windows[i]
                    machine_avail = multi_machine_avail[i]
                    # st = time()
                    result = solve_window_with_machine_avail(jsp_instance, ops_in_window, op_start_times, machine_avail)
                    # ed = time()
                    # print(f"Window {i} solved in {ed - st:.2f} seconds")
                    if result is None:
                        ops_in_window.sort(key=lambda x: op_start_times[x[0]][x[1]])  # 按原始开始时间排序
                    else:
                        try:
                            ops_in_window.sort(key=lambda x: result[(x[0], x[1])])
                        except KeyError as e:
                            print(e)
                            print(result.keys())
                            print(result.values())
                            import sys
                            sys.exit(0)
                    for job, op in ops_in_window:
                        machine = jsp_instance["mch"][job][op]
                        orders[machine].append((job, op))
                assert all([len(o) == j for o in orders]), f"某台机器的操作数量不正确: {[len(o) for o in orders]} vs {j}"
                # import sys
                # sys.exit(0)
                # 实现根据orders更新op_start_times的逻辑，确保工件顺序约束和机器约束都得到满足
                # 这里需要一个调度逻辑，按照orders中每台机器的操作顺序，计算每个操作的开始时间，并更新op_start_times
                machine_ready = [0] * m
                job_ready = [0] * j
                order_idx = [0] * m  # 记录每台机器当前调度到哪个操作
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
                        # 检查工件前一个操作是否完成
                        if op > 0 and (job, op-1) not in scheduled_ops:
                            continue
                        # 可以调度这个操作了
                        start_time = max(job_ready[job], machine_ready[machine])
                        op_start_times[job][op] = start_time
                        end_time = start_time + duration[job][op]
                        job_ready[job] = end_time
                        machine_ready[machine] = end_time
                        scheduled_ops.add((job, op))
                        order_idx[machine] += 1
                        progress = True
                    if not progress:
                        raise ValueError("调度过程中没有进展，可能是orders中的操作顺序不合理，导致死锁")
                makespan = max(op_start_times[job][m-1] + duration[job][m-1] for job in range(j))
            elif debug == 'prefix_windows':
                multi_ops_in_windows = []
                multi_machine_avail = []
                windows_data = []
                start_idx = window_idx % window_size
                total_windows = window_idx // window_size
                multi_ops_in_windows.append(sorted_ops[0:start_idx])
                if total_windows > 0:
                    window_list = [sorted_ops[start_idx + i*window_size:start_idx + (i+1)*window_size] for i in range(0, total_windows)]
                    multi_ops_in_windows.extend(window_list)
                assert sum(len(w) for w in multi_ops_in_windows) == window_idx, f"多窗口划分错误，操作总数不匹配: {sum(len(w) for w in multi_ops_in_windows)} vs {window_idx}"

                for i in range(len(multi_ops_in_windows)):
                    ops_in_window = multi_ops_in_windows[i]
                    machine_avail = get_machine_window_availability(
                        ops_in_window, op_start_times, duration, jsp_instance["mch"]
                    )
                    multi_machine_avail.append(machine_avail)
                orders = [[] for _ in range(m)]
                # 对于前面的操作保证顺序不变，直接加入orders
                for i in range(len(multi_ops_in_windows)):
                    ops_in_window = multi_ops_in_windows[i]
                    machine_avail = multi_machine_avail[i]
                    # st = time()
                    result = solve_window_with_machine_avail(jsp_instance, ops_in_window, op_start_times, machine_avail)
                    # ed = time()
                    # print(f"Window {i} solved in {ed - st:.2f} seconds")
                    ops_in_window.sort(key=lambda x: result[x[0], x[1]])
                    for job, op in ops_in_window:
                        machine = jsp_instance["mch"][job][op]
                        orders[machine].append((job, op))
                for ops in sorted_ops[window_idx:]:
                    job, op = ops
                    machine = jsp_instance["mch"][job][op]
                    orders[machine].append((job, op))
                assert all([len(o) == j for o in orders]), f"某台机器的操作数量不正确: {[len(o) for o in orders]} vs {j}"
                # import sys
                # sys.exit(0)
                # 实现根据orders更新op_start_times的逻辑，确保工件顺序约束和机器约束都得到满足
                # 这里需要一个调度逻辑，按照orders中每台机器的操作顺序，计算每个操作的开始时间，并更新op_start_times
                machine_ready = [0] * m
                job_ready = [0] * j
                order_idx = [0] * m  # 记录每台机器当前调度到哪个操作
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
                        # 检查工件前一个操作是否完成
                        if op > 0 and (job, op-1) not in scheduled_ops:
                            continue
                        # 可以调度这个操作了
                        start_time = max(job_ready[job], machine_ready[machine])
                        op_start_times[job][op] = start_time
                        end_time = start_time + duration[job][op]
                        job_ready[job] = end_time
                        machine_ready[machine] = end_time
                        scheduled_ops.add((job, op))
                        order_idx[machine] += 1
                        progress = True
                    if not progress:
                        raise ValueError("调度过程中没有进展，可能是orders中的操作顺序不合理，导致死锁")
                makespan = max(op_start_times[job][m-1] + duration[job][m-1] for job in range(j))
            else:
                total_windows = (nopr - window_idx) // window_size
                machine_avail = get_machine_window_availability(
                    ops_in_window, op_start_times, duration, jsp_instance["mch"]
                )
                # st = time()
                window_result = solve_window_with_machine_avail(jsp_instance, 
                                                                ops_in_window, 
                                                                op_start_times, 
                                                                machine_avail)
                # ed = time()
                # print(f"Window {window_idx} solved in {ed - st:.2f} seconds")
                if not window_result:
                    continue  # 优化失败，跳过

                # 更新窗口内操作的开始时间
                original_max_time = 0
                for (job, op), st in window_result.items():
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
                # if after_max_time == original_max_time:
                #     print(f"Iteration {it} find new solution but no improvement")
                # elif after_max_time > original_max_time:
                #     print(f"Iteration {it} find worse solution")
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
            # print(f"Iteration {it}: Found better solution with makespan {makespan} (improvement: {(best_makespan - makespan) / best_makespan * 100:.2f}%)")
        #print(f"Iteration {it}: Found better solution with makespan {makespan} (improvement: {(best_makespan - makespan) / best_makespan * 100:.2f}%)")
        if makespan < best_makespan:
            count += 1
            best_solution = [row[:] for row in op_start_times]
            best_makespan = makespan
            # DEBUG: 仅在特定实例上打印改进信息
            # print(f"Iteration {it}: Found better solution with makespan {best_makespan}")
            if debug == 'all_windows':
                print(f"Iteration {it}: Found better solution with makespan {best_makespan}, total windows is {total_windows}, "
                      f"prefix windows is {window_idx // window_size}, suffix windows is {(nopr - window_idx) // window_size}")
            # 更新
            all_ops = [(job, op) for job in range(j) for op in range(m)]
            op_start_flat = {(job, op): op_start_times[job][op] for job, op in all_ops}
            sorted_ops = sorted(all_ops, key=lambda x: op_start_flat[x])
            find_better_in_this_iteration = True
            
            # break
        else:
            find_better_in_this_iteration = False

    print(f"LNS finished after {max_iterations} iterations, improved {count} times, best makespan: {best_makespan}")
    return best_solution, best_makespan


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


def _copy_int_matrix(matrix, rows, cols):
    return [[int(matrix[row][col]) for col in range(cols)] for row in range(rows)]


def _normalize_jsp_instance_for_tabu(jsp_instance, est):
    """
    将 TS 路径中会频繁访问的数据转成纯 Python int/list，
    减少在高频循环中反复访问 numpy 标量带来的原生层风险。
    """
    j, m = jsp_instance["j"], jsp_instance["m"]
    normalized = dict(jsp_instance)
    normalized["duration"] = _copy_int_matrix(jsp_instance["duration"], j, m)
    normalized["mch"] = _copy_int_matrix(jsp_instance["mch"], j, m)
    normalized_est = _copy_int_matrix(est, j, m)
    return normalized, normalized_est


def get_machine_orders_from_start_times(op_start_times, jsp_instance):
    """
    根据开始时间恢复每台机器上的操作顺序。
    返回: List[List[(job, op)]]
    """
    j, m = jsp_instance["j"], jsp_instance["m"]
    mch = jsp_instance["mch"]

    machine_orders = [[] for _ in range(m)]
    for job in range(j):
        for op in range(m):
            machine = mch[job][op]
            machine_orders[machine].append((job, op))

    for machine in range(m):
        order = machine_orders[machine]
        order.sort(key=lambda x: op_start_times[x[0]][x[1]])
        machine_orders[machine] = order
    return machine_orders


def schedule_from_machine_orders(jsp_instance, machine_orders):
    """
    给定每台机器上的操作顺序，按最早可行开始时间解码调度。
    若机器顺序与工件顺序形成环，则返回 (None, None)。
    """
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
            assert type(machine) == int, f"machine should be int, but got {type(machine)} for machine {machine}"
            assert type(machine_idx) == list, f"machine_idx should be list, but got {type(machine_idx)}"
            assert type(machine_idx[machine]) == int, f"machine_idx should be int, but got {type(machine_idx[machine])} for machine {machine}"
            idx = machine_idx[machine]
            if idx >= len(machine_orders[machine]):
                continue

            job, op = (machine_orders[machine])[idx]
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

    makespan = max(job_ready)
    return op_start_times, makespan


def extract_critical_path(op_start_times, duration, mch):
    """
    从当前调度中提取一条关键路径。
    """
    j = len(op_start_times)
    m = len(op_start_times[0])
    finish = [
        [op_start_times[job][op] + duration[job][op] for op in range(m)]
        for job in range(j)
    ]

    machine_seq = [[] for _ in range(m)]
    for job in range(j):
        for op in range(m):
            machine_seq[mch[job][op]].append((job, op))
    for machine in range(m):
        order = machine_seq[machine]
        order.sort(key=lambda x: op_start_times[x[0]][x[1]])
        machine_seq[machine] = order

    makespan = max(finish[job][m - 1] for job in range(j))
    end_ops = [(job, m - 1) for job in range(j) if finish[job][m - 1] == makespan]
    if not end_ops:
        return []

    current = random.choice(end_ops)
    path = [current]

    while True:
        job, op = current
        start_time = op_start_times[job][op]
        candidates = []

        if op > 0:
            prev_job_op = (job, op - 1)
            prev_finish = op_start_times[job][op - 1] + duration[job][op - 1]
            if prev_finish == start_time:
                candidates.append(prev_job_op)

        machine = mch[job][op]
        machine_ops = machine_seq[machine]
        #idx = machine_ops.index(current)
        idx = 0
        for job_, op_ in machine_ops:
            if (job_, op_) == current:
                break
            idx += 1

        if idx > 0:
            prev_machine_op = machine_ops[idx - 1]
            prev_finish = op_start_times[prev_machine_op[0]][prev_machine_op[1]] + duration[prev_machine_op[0]][prev_machine_op[1]]
            if prev_finish == start_time:
                candidates.append(prev_machine_op)

        if not candidates:
            break

        current = random.choice(candidates)
        path.append(current)

    path.reverse()
    return path


def split_critical_blocks(crit_path, mch):
    """
    将关键路径按机器分割为关键块。
    """
    if not crit_path:
        return []

    blocks = []
    current_block = [crit_path[0]]
    for prev, cur in zip(crit_path, crit_path[1:]):
        try:
            if mch[prev[0]][prev[1]] == mch[cur[0]][cur[1]]:
                    current_block.append(cur)
            else:
                if len(current_block) >= 2:
                    blocks.append(current_block)
                current_block = [cur]
        except Exception as e:
            print(f"mch: {mch}, prev: {prev}, cur: {cur}, mch[prev[0]][prev[1]]: {mch[prev[0]][prev[1]]}, mch[cur[0]][cur[1]]: {mch[cur[0]][cur[1]]}")
            print("Error in split_critical_blocks:", e)

    if len(current_block) >= 2:
        blocks.append(current_block)
    return blocks


def _generate_n5_neighbors(jsp_instance, op_start_times):
    """
    生成 N5 邻域:
    对每个关键块，仅交换最前面的两个或最后面的两个操作。
    """
    duration = jsp_instance["duration"]
    mch = jsp_instance["mch"]

    crit_path = extract_critical_path(op_start_times, duration, mch)
    blocks = split_critical_blocks(crit_path, mch)
    if not blocks:
        return []

    machine_orders = get_machine_orders_from_start_times(op_start_times, jsp_instance)
    op_pos = {}
    for machine in range(len(machine_orders)):
        order = machine_orders[machine]
        for idx, op in enumerate(order):
            op_pos[(op[0], op[1])] = (machine, idx)

    neighbors = []
    seen_pairs = set()
    for block in blocks:
        candidate_pairs = [(block[0], block[1])]
        if len(block) > 2:
            candidate_pairs.append((block[-2], block[-1]))

        for first, second in candidate_pairs:
            machine, idx = op_pos[first]
            if op_pos[second][0] != machine:
                continue
            if idx + 1 >= len(machine_orders[machine]):
                continue
            if machine_orders[machine][idx + 1] != second:
                continue

            move_key = (first, second)
            if move_key in seen_pairs:
                continue
            seen_pairs.add(move_key)

            new_orders = [order[:] for order in machine_orders]
            new_orders[machine][idx], new_orders[machine][idx + 1] = (
                new_orders[machine][idx + 1],
                new_orders[machine][idx],
            )

            cand_start_times, cand_makespan = schedule_from_machine_orders(
                jsp_instance, new_orders
            )
            if cand_start_times is None:
                continue

            neighbors.append({
                "machine": machine,
                "swap": (first, second),
                "move_key": move_key,
                "tabu_key": (second, first),
                "start_times": cand_start_times,
                "makespan": cand_makespan,
            })

    return neighbors


def tabu_search_n5(
    jsp_instance,
    est,
    max_iterations=200,
    tabu_tenure=None,
    max_no_improve=None,
    debug=False,
):
    """
    使用经典禁忌搜索 + N5 邻域求解 JSSP。

    参数:
        jsp_instance: JSSP 实例
        est: 初始解的开始时间矩阵，形状为 (j, m)
        max_iterations: 最大迭代次数
        tabu_tenure: 禁忌长度；若为 None，则自动设置
        max_no_improve: 连续未改进的最大次数；若为 None，则默认为 max_iterations
        debug: 是否打印搜索过程

    返回:
        best_start_times, best_makespan
    """
    jsp_instance, est = _normalize_jsp_instance_for_tabu(jsp_instance, est)

    j, m = jsp_instance["j"], jsp_instance["m"]
    duration = jsp_instance["duration"]
    total_ops = j * m

    current_start = [row[:] for row in est]
    current_makespan = max(
        current_start[job][m - 1] + duration[job][m - 1]
        for job in range(j)
    )
    best_start = [row[:] for row in current_start]
    best_makespan = current_makespan

    if tabu_tenure is None:
        tabu_tenure = max(5, min(10, total_ops // 10 + 1))
    if max_no_improve is None:
        max_no_improve = max_iterations

    tabu_until = {}
    no_improve_rounds = 0

    for iteration in range(1, max_iterations + 1):
        best_candidate = None
        best_fallback = None
        neighbors = _generate_n5_neighbors(jsp_instance, current_start)
        if iteration % 500 == 0 or debug:
            print(f"Tabu Search Iteration {iteration}, current makespan: {current_makespan}, best makespan: {best_makespan}")
        for candidate in neighbors:
            if (
                best_fallback is None
                or candidate["makespan"] < best_fallback["makespan"]
            ):
                best_fallback = candidate

            is_tabu = tabu_until.get(candidate["move_key"], 0) > iteration
            if is_tabu and candidate["makespan"] >= best_makespan:
                continue

            if (
                best_candidate is None
                or candidate["makespan"] < best_candidate["makespan"]
            ):
                best_candidate = candidate

        if not neighbors:
            break

        chosen = best_candidate if best_candidate is not None else best_fallback
        if chosen is None:
            break

        current_start = [row[:] for row in chosen["start_times"]]
        current_makespan = chosen["makespan"]
        tabu_until[chosen["tabu_key"]] = iteration + tabu_tenure

        if current_makespan < best_makespan:
            best_start = [row[:] for row in current_start]
            best_makespan = current_makespan
            no_improve_rounds = 0
            if debug:
                print(
                    f"[TS-N5] iter={iteration}, "
                    f"machine={chosen['machine']}, "
                    f"swap={chosen['swap']}, "
                    f"best_makespan={best_makespan}"
                )
        else:
            no_improve_rounds += 1

        if no_improve_rounds >= max_no_improve:
            break

    return best_start, best_makespan


if __name__ == "__main__":
    dataset = JSPNumpyDataset(data_dir="./benchmark/TA")
    gaps = ObjMeter()
    better_gaps = ObjMeter()
    random.seed(0)
    st = time()

    pdr = PDR(priority=SPT())
    start_var = 61
    end_var = 70
    count = 0
    for jsp_dataset in dataset:
        count += 1
        if not (count >= start_var and count <= end_var):
            continue
        sols, ms, _ = solve_instance(jsp_dataset, pdr=pdr)
        
        op_start_times = convert_solution_to_start_times(sols, jsp_dataset)
        assert ms == max(op_start_times[job][jsp_dataset["m"]-1] + jsp_dataset["duration"][job][jsp_dataset["m"]-1] for job in range(jsp_dataset["j"])), \
            f"计算的makespan {ms} 与根据op_start_times计算的makespan不一致 {max(op_start_times[job][jsp_dataset['m']-1] + jsp_dataset['duration'][job][jsp_dataset['m']-1] for job in range(jsp_dataset['j']))}"
        
        print(f"Initial solution for {jsp_dataset['names']} has makespan {ms}, best known makespan {jsp_dataset['makespan']}")
        best_start_times, best_makespan = tabu_search_n5(
            jsp_dataset,
            op_start_times,
            max_iterations=5000,
            tabu_tenure=20,
            max_no_improve=None,
            debug=False,
        )
        gap = (ms - jsp_dataset["makespan"]) / jsp_dataset["makespan"] * 100
        better_gap = (best_makespan - jsp_dataset["makespan"]) / jsp_dataset["makespan"] * 100
        print(jsp_dataset["names"], f" Gap: {gap:.2f}", f" Better Gap: {better_gap:.2f}")
        gaps.update(jsp_dataset, gap)
        better_gaps.update(jsp_dataset, better_gap)
        # plot_gantt(op_start_times, jsp_dataset, title=f"{jsp_dataset['names']}(makespan={better_gap:.2f})")
    print("Overall Gaps:", gaps.avg)
    print(gaps)
    print("Overall Better Gaps:", better_gaps.avg)
    print(better_gaps)
    end = time()
    print(f"Total time: {end - st:.2f} seconds")
