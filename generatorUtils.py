import os.path
import os
import numpy as np

def set_random_seed():
    seed = os.getpid()
    np.random.seed(seed)


def permute_rows(x):
    """
    :param x: np array 2-D
    :return:
    """
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high):
    """
    Args:
        n_j: number of jobs
        n_m: number of machines
        low: low process time
        high: high process time
    Returns:
        times: n_j * n_m
        machines: n_j * n_m
    """
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(0, n_m), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    return times, machines


def override(fn):
    """
    override decorator
    """
    return fn


def generateInstanceWithoutGt(num_instance=1, n_j=6, n_m=6, file_dir=f'DataSet'):
    # num_instance = 1
    # n_j = 6
    # n_m = 6
    # file_dir = f'DataSet{n_j}x{n_m}'
    set_random_seed()
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    for i in range(num_instance):
        times, machines = uni_instance_gen(n_j=n_j, n_m=n_m, low=1, high=99)
        filename = os.path.join(file_dir, str(i) + ".jsp")
        with open(filename, "w") as f:
            f.write(f"{n_j} {n_m}\n")

            for time_list, machine_list in zip(times, machines):
                for time, machine in zip(time_list, machine_list):
                    f.write(f"{machine} {time} ")
                f.write("\n")
            f.close()