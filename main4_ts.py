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
        progress = False
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
                    progress = True
            else:
                # 第一个操作，直接安排在机器可用时间之后
                op_start_times[job][op] = machine_ready[machine]
                machine_ready[machine] = op_start_times[job][op] + duration[job][op]
                job_ready[job] = op_start_times[job][op] + duration[job][op]

                scheduled_set.add((job, op))
                schedule_index[machine] += 1
                progress = True
        if not progress:
            raise RuntimeError("convert_solution_to_start_times stuck: invalid machine order")

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
            raise e

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
    import os
    allowed = set(range(1, os.cpu_count()))
    os.sched_setaffinity(0, allowed)
    print("CPU affinity:", os.sched_getaffinity(0))

    dataset = JSPNumpyDataset(data_dir="./benchmark/DMU")
    gaps = ObjMeter()
    better_gaps = ObjMeter()
    random.seed(0)
    st = time()

    pdr = PDR(priority=SPT())
    start_var = 1
    end_var = 80
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
