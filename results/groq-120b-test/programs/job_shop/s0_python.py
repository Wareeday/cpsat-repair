import time
from typing import List, Tuple, Dict

def solve(instance: Dict) -> Dict:
    jobs: List[List[Tuple[int, int]]] = instance["jobs"]
    n_machines: int = instance["n_machines"]
    n_jobs = len(jobs)

    # ---------- data structures ----------
    # start times (filled progressively)
    starts: List[List[int]] = [[-1] * len(job) for job in jobs]

    # next operation index for each job
    next_op: List[int] = [0] * n_jobs
    # ready times (when the job / machine becomes free)
    job_ready: List[int] = [0] * n_jobs
    machine_ready: List[int] = [0] * n_machines

    # best solution found so far
    best_makespan = float('inf')
    best_starts: List[List[int]] = []

    # time limit (seconds)
    TIME_LIMIT = 28.0   # a bit less than the 30 s hard limit
    start_time = time.time()

    # ---------- helper lower bound ----------
    def lower_bound() -> int:
        """Compute a simple admissible lower bound for the current partial schedule."""
        # current longest completion among already scheduled ops
        cur = max(job_ready) if job_ready else 0

        # job based bound: each job must finish its remaining work sequentially
        job_bound = cur
        for j, idx in enumerate(next_op):
            if idx < len(jobs[j]):
                rem = sum(d for _, d in jobs[j][idx:])
                job_bound = max(job_bound, job_ready[j] + rem)

        # machine based bound: each machine must process its remaining work sequentially
        mach_bound = cur
        # accumulate remaining durations per machine
        rem_per_mach = [0] * n_machines
        for j, idx in enumerate(next_op):
            for op in jobs[j][idx:]:
                m, d = op
                rem_per_mach[m] += d
        for m in range(n_machines):
            mach_bound = max(mach_bound, machine_ready[m] + rem_per_mach[m])

        return max(cur, job_bound, mach_bound)

    # ---------- recursive search ----------
    def dfs(depth: int) -> None:
        nonlocal best_makespan, best_starts

        # time check
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError

        # all operations scheduled ?
        if depth == total_ops:
            makespan = max(job_ready) if job_ready else 0
            if makespan < best_makespan:
                best_makespan = makespan
                # copy starts
                best_starts = [row[:] for row in starts]
            return

        # prune by lower bound
        lb = lower_bound()
        if lb >= best_makespan:
            return

        # collect eligible operations (next op of each unfinished job)
        candidates = []
        for j in range(n_jobs):
            idx = next_op[j]
            if idx < len(jobs[j]):
                m, d = jobs[j][idx]
                earliest = max(job_ready[j], machine_ready[m])
                candidates.append((earliest, d, j, idx, m))

        # order heuristically: earliest start, then shortest processing time
        candidates.sort(key=lambda x: (x[0], x[1]))

        for earliest, dur, j, op_idx, m in candidates:
            # schedule this operation
            start = earliest
            finish = start + dur

            # update structures
            starts[j][op_idx] = start
            prev_job_ready = job_ready[j]
            prev_mach_ready = machine_ready[m]
            prev_next = next_op[j]

            job_ready[j] = finish
            machine_ready[m] = finish
            next_op[j] = op_idx + 1

            dfs(depth + 1)

            # backtrack
            starts[j][op_idx] = -1
            job_ready[j] = prev_job_ready
            machine_ready[m] = prev_mach_ready
            next_op[j] = prev_next

    total_ops = sum(len(job) for job in jobs)
    try:
        dfs(0)
    except TimeoutError:
        pass  # return the best solution found so far

    # if no solution was found (should not happen), fall back to a trivial schedule
    if not best_starts:
        # schedule jobs sequentially respecting machine order (very naive)
        cur_time_m = [0] * n_machines
        cur_time_j = [0] * n_jobs
        for j, job in enumerate(jobs):
            for k, (m, d) in enumerate(job):
                start = max(cur_time_j[j], cur_time_m[m])
                starts[j][k] = start
                cur_time_j[j] = start + d
                cur_time_m[m] = start + d
        best_starts = starts

    return {"starts": best_starts}