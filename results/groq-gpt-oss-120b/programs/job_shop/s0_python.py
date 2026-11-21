import time
import sys

def solve(instance: dict) -> dict:
    jobs = instance["jobs"]
    n_machines = instance["n_machines"]
    n_jobs = len(jobs)

    # Pre‑compute suffix sums of remaining processing times for each job
    suffix_sum = []
    for j in range(n_jobs):
        ops = jobs[j]
        s = [0] * (len(ops) + 1)
        for i in range(len(ops) - 1, -1, -1):
            s[i] = s[i + 1] + ops[i][1]
        suffix_sum.append(s)

    total_ops = sum(len(j) for j in jobs)

    # Best solution found so far
    best_makespan = float("inf")
    best_starts = None

    # Time limit handling
    start_time = time.time()
    TIME_LIMIT = 29.0  # seconds

    sys.setrecursionlimit(10000)

    def lower_bound(job_ready, machine_ready, next_op):
        # current makespan
        lb = max(job_ready) if job_ready else 0

        # job‑wise lower bound
        for j in range(n_jobs):
            idx = next_op[j]
            if idx < len(jobs[j]):
                lb = max(lb, job_ready[j] + suffix_sum[j][idx])

        # machine‑wise lower bound (simple sum of remaining work)
        for m in range(n_machines):
            rem = 0
            for j in range(n_jobs):
                idx = next_op[j]
                for k in range(idx, len(jobs[j])):
                    if jobs[j][k][0] == m:
                        rem += jobs[j][k][1]
            lb = max(lb, machine_ready[m] + rem)

        return lb

    def dfs(job_ready, machine_ready, next_op, starts, scheduled):
        nonlocal best_makespan, best_starts

        # time check
        if time.time() - start_time > TIME_LIMIT:
            return  # stop exploring further

        # all operations scheduled
        if scheduled == total_ops:
            makespan = max(job_ready) if job_ready else 0
            if makespan < best_makespan:
                best_makespan = makespan
                # copy starts
                best_starts = [row[:] for row in starts]
            return

        # prune by lower bound
        if lower_bound(job_ready, machine_ready, next_op) >= best_makespan:
            return

        # collect candidate jobs whose next operation is ready to be scheduled
        candidates = []
        for j in range(n_jobs):
            if next_op[j] < len(jobs[j]):
                m, d = jobs[j][next_op[j]]
                earliest = max(job_ready[j], machine_ready[m])
                candidates.append((earliest, j, m, d))

        # order candidates – earliest start first (helps pruning)
        candidates.sort(key=lambda x: x[0])

        for earliest, j, m, d in candidates:
            # schedule this operation at its earliest feasible start
            start_time_op = earliest
            finish = start_time_op + d

            # copy state
            new_job_ready = job_ready[:]
            new_machine_ready = machine_ready[:]
            new_next_op = next_op[:]
            new_starts = [row[:] for row in starts]

            new_job_ready[j] = finish
            new_machine_ready[m] = finish
            new_next_op[j] += 1
            new_starts[j][new_next_op[j] - 1] = start_time_op

            dfs(new_job_ready, new_machine_ready, new_next_op, new_starts, scheduled + 1)

            # early exit if time limit reached
            if time.time() - start_time > TIME_LIMIT:
                return

    # initial state
    init_job_ready = [0] * n_jobs
    init_machine_ready = [0] * n_machines
    init_next_op = [0] * n_jobs
    init_starts = [[0] * len(jobs[j]) for j in range(n_jobs)]

    dfs(init_job_ready, init_machine_ready, init_next_op, init_starts, 0)

    # If no solution was found within the time limit (should not happen for tiny instances),
    # fall back to a simple greedy schedule (list jobs in order).
    if best_starts is None:
        job_ready = [0] * n_jobs
        machine_ready = [0] * n_machines
        starts = [[0] * len(jobs[j]) for j in range(n_jobs)]
        for j in range(n_jobs):
            for k, (m, d) in enumerate(jobs[j]):
                s = max(job_ready[j], machine_ready[m])
                starts[j][k] = s
                f = s + d
                job_ready[j] = f
                machine_ready[m] = f
        best_starts = starts

    return {"starts": best_starts}