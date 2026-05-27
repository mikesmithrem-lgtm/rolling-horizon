from dataset import JSPNumpyDataset
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
	
from L2S.env.generateJSP import uni_instance_gen
from L2S.env.permissible_LS import permissibleLeftShift


def pdr_spt_init_extracted(duration, mch_zero_based, eps=1e-3, tau=1e-4):
	"""
	Extracted SPT init logic from pdrs.PDR.__call__ + SPT priority.

	Args:
		duration: shape [n_job, n_mch]
		mch_zero_based: shape [n_job, n_mch], machine ids in [0, n_mch-1]
	Returns:
		orders_by_machine: list[list[int]], operation ids per machine
		op_start_times: shape [n_job, n_mch]
		makespan: int
	"""
	n_job, n_mch = duration.shape
	ops = np.array([(0, mch_zero_based[j, 0]) for j in range(n_job)], dtype=np.int64)

	# SPT priority in pdrs: priority = -duration - tie, tie increases every call.
	tie = 0.0
	prio = np.zeros(n_job, dtype=np.float64)
	for j in range(n_job):
		tie += tau
		prio[j] = -float(duration[j, 0]) - tie

	curr_time = -1
	mac_time = [0 for _ in range(n_mch)]
	sol = [[] for _ in range(n_mch)]
	end_times_by_job = [[] for _ in range(n_job)]

	active_job = n_job
	while active_job > 0:
		job_order = np.argsort(prio)[::-1][:active_job]
		curr_time = min(ct for ct in mac_time if ct > curr_time)

		for j in job_order:
			idx, machine = int(ops[j, 0]), int(ops[j, 1])
			min_st = max(mac_time[machine], 0 if idx == 0 else end_times_by_job[j][-1])
			if min_st - eps < curr_time:
				if idx < n_mch - 1:
					ops[j, 0] = idx + 1
					ops[j, 1] = int(mch_zero_based[j, idx + 1])
					tie += tau
					prio[j] = -float(duration[j, idx + 1]) - tie
				else:
					active_job -= 1
					prio[j] = -float("inf")

				mac_time[machine] = int(min_st + duration[j, idx])
				end_times_by_job[j].append(mac_time[machine])
				sol[machine].append(j * n_mch + idx)

	op_start_times, makespan = rebuild_start_times_from_orders(sol, duration, mch_zero_based)
	return sol, op_start_times, makespan


def env_rules_spt_init_extracted(duration, mch_one_based, high=99, seed=0):
	"""
	Extracted SPT init logic from environment.JsspN5/JsspWindow._rules_solver.

	Args:
		duration: shape [n_job, n_mch]
		mch_one_based: shape [n_job, n_mch], machine ids in [1, n_mch]
		high: sentinel used in permissibleLeftShift; should match project default 99
		seed: random seed for tie-breaking in _rules_solver (np.random.choice)
	Returns:
		orders_by_machine: list[list[int]], operation ids per machine
		op_start_times: shape [n_job, n_mch]
		makespan: int
	"""
	np.random.seed(seed)

	n_job, n_mch = duration.shape
	n_operations = n_job * n_mch
	last_col = np.arange(0, n_operations, 1).reshape(n_job, -1)[:, -1]
	candidate_oprs = np.arange(0, n_operations, 1).reshape(n_job, -1)[:, 0]
	mask = np.zeros(shape=n_job, dtype=bool)

	# Keep same memory layout and sentinel semantics as _rules_solver.
	gant_chart = -high * np.ones_like(duration.transpose(), dtype=np.int32)
	op_ids_on_mchs = -n_job * np.ones_like(duration.transpose(), dtype=np.int32)
	finished_mark = np.zeros_like(mch_one_based, dtype=np.int32)

	for _ in range(n_operations):
		candidate_masked = candidate_oprs[np.where(~mask)]
		dur_candidate = np.take(duration, candidate_masked)
		idx = np.random.choice(np.where(dur_candidate == np.min(dur_candidate))[0])
		action = int(candidate_masked[idx])

		permissibleLeftShift(
			a=action,
			durMat=duration,
			mchMat=mch_one_based,
			mchsStartTimes=gant_chart,
			opIDsOnMchs=op_ids_on_mchs,
		)
		if action not in last_col:
			candidate_oprs[action // n_mch] += 1
		else:
			mask[action // n_mch] = True
		finished_mark[action // n_mch, action % n_mch] = 1

	orders_by_machine = []
	for machine in range(n_mch):
		valid = [int(op) for op in op_ids_on_mchs[machine].tolist() if op >= 0]
		orders_by_machine.append(valid)

	mch_zero_based = mch_one_based - 1
	op_start_times, makespan = rebuild_start_times_from_orders(
		orders_by_machine,
		duration,
		mch_zero_based,
	)
	return orders_by_machine, op_start_times, makespan


def rebuild_start_times_from_orders(orders_by_machine, duration, mch_zero_based):
	"""Rebuild globally feasible op start times from per-machine orders."""
	n_job, n_mch = duration.shape
	op_start_times = np.zeros((n_job, n_mch), dtype=np.int32)
	machine_ready = [0] * n_mch
	job_ready = [0] * n_job
	order_idx = [0] * n_mch
	scheduled = set()

	while len(scheduled) < n_job * n_mch:
		progress = False
		for machine in range(n_mch):
			if order_idx[machine] >= len(orders_by_machine[machine]):
				continue

			op_id = orders_by_machine[machine][order_idx[machine]]
			job, op = op_id // n_mch, op_id % n_mch
			if (job, op) in scheduled:
				order_idx[machine] += 1
				continue

			if op > 0 and (job, op - 1) not in scheduled:
				continue

			start_time = max(job_ready[job], machine_ready[machine])
			op_start_times[job, op] = start_time
			end_time = int(start_time + duration[job, op])
			job_ready[job] = end_time
			machine_ready[machine] = end_time
			scheduled.add((job, op))
			order_idx[machine] += 1
			progress = True

		if not progress:
			raise RuntimeError("Cannot rebuild feasible schedule from machine orders")

	makespan = int(max(op_start_times[j, n_mch - 1] + duration[j, n_mch - 1] for j in range(n_job)))
	return op_start_times, makespan


def compare_spt_initialization(duration, mch_one_based, seed=0):
	"""
	Compare extracted PDR-SPT and env _rules_solver-SPT on one instance.
	"""
	mch_zero_based = mch_one_based - 1
	pdr_orders, pdr_starts, pdr_ms = pdr_spt_init_extracted(duration, mch_zero_based)
	env_orders, env_starts, env_ms = env_rules_spt_init_extracted(duration, mch_one_based, high=99, seed=seed)

	print("=== Instance Summary ===")
	print(f"jobs={duration.shape[0]}, machines={duration.shape[1]}")
	print(f"PDR-SPT makespan: {pdr_ms}")
	print(f"ENV-SPT makespan: {env_ms}")
	print(f"Makespan equal? {pdr_ms == env_ms}")

	same_orders = all(pdr_orders[m] == env_orders[m] for m in range(len(pdr_orders)))
	same_starts = bool(np.array_equal(pdr_starts, env_starts))
	print(f"Machine orders equal? {same_orders}")
	print(f"Start times equal? {same_starts}")

	if not same_orders:
		for machine, (po, eo) in enumerate(zip(pdr_orders, env_orders)):
			if po != eo:
				print(f"First order mismatch on machine {machine}:")
				print(f"  pdr order: {po}")
				print(f"  env order: {eo}")
				break

	if not same_starts:
		diff_locs = np.argwhere(pdr_starts != env_starts)
		if diff_locs.size > 0:
			j, o = diff_locs[0]
			print(f"First start-time mismatch at (job={j}, op={o}):")
			print(f"  pdr start={int(pdr_starts[j, o])}, env start={int(env_starts[j, o])}")

	print("\n=== Why They Differ (Core Logic) ===")
	print("1) Tie-breaking rule differs:")
	print("   - pdr SPT uses deterministic tie offset tau (order-sensitive, no randomness).")
	print("   - env _rules_solver uses np.random.choice among equal shortest durations.")
	print("2) Insertion policy differs:")
	print("   - pdr appends on machine timeline when operation becomes dispatchable at current time.")
	print("   - env uses permissibleLeftShift, which may insert operation into an earlier feasible gap.")
	print("3) Time-advancing style differs:")
	print("   - pdr is event-driven with curr_time and active-job scan.")
	print("   - env repeatedly picks shortest candidate operation globally (subject to job-frontier),")
	print("     then repairs feasibility via left-shift placement.")

	return {
		"pdr_orders": pdr_orders,
		"env_orders": env_orders,
		"pdr_starts": pdr_starts,
		"env_starts": env_starts,
		"pdr_makespan": pdr_ms,
		"env_makespan": env_ms,
	}


def main():
	np.random.seed(0)
	n_job, n_mch = 10, 10
	low, high = 1, 99

	# duration, mch_one_based = uni_instance_gen(n_job, n_mch, low, high)
	dataset = JSPNumpyDataset(data_dir="./benchmark/TA")
	jsp_instance = dataset[70]
	duration = jsp_instance["duration"]
	mch_one_based = jsp_instance["mch"] + 1
	compare_spt_initialization(duration, mch_one_based, seed=0)


if __name__ == "__main__":
	main()
