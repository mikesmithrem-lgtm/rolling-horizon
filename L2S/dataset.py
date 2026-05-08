import torch
import os
import numpy as np
class JSPNumpyDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir):
    self.data_dir = data_dir
    self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jsp')]
    self.data_files.sort()  # 确保文件顺序一致

  def __len__(self):
    return len(self.data_files)

  def get_sample(self, idx):
      data_file = self.data_files[idx]
      data_file_name = os.path.join(self.data_dir, data_file)
      data_file_lines = open(data_file_name, 'r').read().splitlines()

      first_row = data_file_lines[0].split(" ")
      n_jobs, n_mchs = int(first_row[0]), int(first_row[1])

      times, machines = [], []

      for i in range(1, n_jobs + 1):
          mid_rows = data_file_lines[i].strip().split(" ")
          time = list(map(int, mid_rows[1::2]))
          machine = list(map(int, mid_rows[0::2]))
          times.append(time)
          machines.append(machine)

      makespan_idx = n_jobs + 1
      if makespan_idx >= len(data_file_lines):
          makespan = -1
          orders = []
          return (
              np.array(times),
              np.array(machines),
              np.array([makespan]),
              np.array(orders)
          )
      makespan = int(data_file_lines[makespan_idx])

      orders = []
      for i in range(makespan_idx + 1, len(data_file_lines)):
          order = data_file_lines[i].split(" ")
          order = list(map(int, order))
          orders.append(order)

      return (
          np.array(times),
          np.array(machines),
          np.array([makespan]),
          np.array(orders)
      )

  def __getitem__(self, idx):
      times, machines, makespan, orders = self.get_sample(idx)
      n_jobs, n_mchs = times.shape[0], times.shape[1]
      data_file = self.data_files[idx]
      return {
         "j": n_jobs,
         "m": n_mchs,
         "duration": times,
         "mch": machines,
         "makespan": makespan,
         "orders": orders,
         "names": data_file
      }


class ObjMeter:
    def __init__(self, name = 'makespan'):
        self.sum = {}
        self.list = {}
        self.count = {}
        self.meter = name

    def update(self, ins: dict, val: float):
        """
        Update with a new value for an instance.

        Args:
            ins: JSP instance.
            val: objective value (e.g. makespan) of the solution.
        Returns:
            None
        """
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
        """ Compute total average value regardless of shapes. """
        return sum(self.sum.values()) / sum(self.count.values()) if self.count \
            else 0


if __name__ == "__main__":
    dataset = JSPNumpyDataset(data_dir="./benchmark/LA")
    for i in range(len(dataset)):
        sample = dataset[i]
        print("Names:", sample["names"])
        print("Times:", sample["duration"])
        print("Machines:", sample["mch"])
        print("Makespan:", sample["makespan"])
        print("Orders:", sample["orders"])
        break