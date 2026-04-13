
import collections
import os.path
import os
from typing import Sequence

import numpy as np
from ortools.sat.python import cp_model
import multiprocessing
import functools

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self) -> None:
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self) -> None:
        print(
            f"Solution {self.__solution_count}, time = {self.wall_time} s,"
            f" objective = {self.objective_value}"
        )
        self.__solution_count += 1


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
        print(f"Optimal Schedule Length: {solver.ObjectiveValue()}")
        print(output)
        print(solver.ResponseStats())

        return machine_orders, solver.ObjectiveValue()

    else :
        raise TimeoutError(f"Time {solver.parameters.max_time_in_seconds} "
                           f"Not Enough For Generating Solutions")


def main(dataset_type, num_instances, n_j, n_m) -> None:
    from generatorUtils import generateInstanceWithoutGt
    from inout import load_dataset
    import os

    file_input_dir = f"raw0_{n_j}x{n_m}/jsp_{n_j}x{n_m}-{dataset_type}"
    file_output_dir = f"jsp0_{n_j}x{n_m}/dataset_{n_j}x{n_m}_{dataset_type}_{num_instances}"

    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)

    # if not os.path.exists(file_input_dir):
    #     os.makedirs(file_input_dir)

    # generateInstanceWithoutGt(num_instance=num_instances, n_j=n_j, n_m=n_m, file_dir=file_input_dir)
    instances = load_dataset(file_input_dir, basic=True)
    flag = False
    compute_idx = len(os.listdir(os.path.join(file_output_dir)))
    print(f"Already computed instances num {compute_idx} for {file_input_dir}")

    for idx, ins in enumerate(instances):
        if idx < compute_idx:
            continue
        print(f'Solving {ins["name"]} ({ins["shape"]}): UB = {ins["makespan"]}')
        machine_ids, process_times = ins['machines'], ins['costs']
        assert len(machine_ids) == len(process_times)
        job_data = np.stack((machine_ids, process_times), axis=-1)
        output, makespan = jobshop_with_maintenance(job_data)

        #FIXME: output the result
        with open(os.path.join(file_output_dir, f"{int(ins['name'])}.jsp"), "w") as file:
            file.write(f"{ins['j']} {ins['m']}\n")
            for one_job_item in job_data:
                for one_operation_item in one_job_item:
                    file.write(f"{one_operation_item[0]} {one_operation_item[1]} ")
                file.write("\n")
            file.write(f"{str(int(makespan))}\n")
            for one_machine_item in output:
                file.write(f"{' '.join(map(str, one_machine_item))}\n")
            file.close()


def multi_main():
    num_parallel_tasks = 1
    num_exchange_tasks = 1

    num_instance = 10
    n_j, n_m = 150, 20
    dataset_types = [f'train_{i}' for i in range(num_exchange_tasks)]

    partial_func = functools.partial(main, num_instances=num_instance, n_j=n_j, n_m=n_m)
    with multiprocessing.Pool(num_parallel_tasks) as pool:
        pool.map(partial_func, dataset_types)


if __name__ == '__main__':
    multi_main()
