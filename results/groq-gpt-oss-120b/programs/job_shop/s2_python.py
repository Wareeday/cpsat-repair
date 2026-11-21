import time
from copy import deepcopy

def solve(instance):
    jobs = instance["jobs"]
    n_machines = instance["n_machines"]
    n_jobs = len(jobs)

    # total number of operations
    total_ops = sum(len(job) for job in jobs)

    # pre‑compute suffix sums of durations for each job (remaining work)
    suffix = []
    for job in jobs:
        s = [0] * (len(job) + 1)
        for i in range(len(job) - 1, -1, -1):
            s[i] = s[i + 1] + job[i][1]
        suffix.append(s)

    # data structures that will be mutated during search
    job_next = [0] * n_jobs                 # next operation index to schedule for each job
    job_ready = [0] * n_jobs                 # earliest time the next op of a job may start
    machine_ready = [0] * n_machines         # earliest free time of each machine
    starts = [[None] * len(job) for job in jobs]

    best_makespan = float('inf')
    best_starts = None

    start_time_limit = time.time() + 28.0   # a little margin before the 30 s hard limit

    # ----------------------------------------------------------------------
    # Helper: lower bound of the makespan for the current partial schedule
    # ----------------------------------------------------------------------
    def lower_bound():
        # job based bound
        job_bound = 0
        for j in range(n_jobs):
            remaining = suffix[j][job_next[j]]
            bound = job_ready[j] + remaining
            if bound > job_bound:
                job_bound = bound

        # machine based bound
        machine_bound = 0
        # compute remaining work per machine
        rem_per_machine = [0] * n_machines
        for j, job in enumerate(jobs):
            idx = job_next[j]
            for op in job[idx:]:
                rem_per_machine[op[0]] += op[1]
        for m in range(n_machines):
            bound = machine_ready[m] + rem_per_machine[m]
            if bound > machine_bound:
                machine_bound = bound

        return max(job_bound, machine_bound)

    # ----------------------------------------------------------------------
    # Depth‑first branch‑and‑bound search
    # ----------------------------------------------------------------------
    def dfs(scheduled):
        nonlocal best_makespan, best_starts

        # time‑out check
        if time.time() > start_time_limit:
            raise StopIteration

        if scheduled == total_ops:
            cur_mk = max(job_ready)   # all jobs finished, makespan = max finish time
            if cur_mk < best_makespan:
                best_makespan = cur_mk
                best_starts = deepcopy(starts)
            return

        # prune by lower bound
        if lower_bound() >= best_makespan:
            return

        # collect all operations that are ready to be scheduled now
        ready_ops = []
        for j in range(n_jobs):
            idx = job_next[j]
            if idx >= len(jobs[j]):
                continue
            m, d = jobs[j][idx]
            est = max(job_ready[j], machine_ready[m])
            ready_ops.append((est, j, idx, m, d))

        # order – try earlier start times first (helps pruning)
        ready_ops.sort(key=lambda x: x[0])

        for est, j, idx, m, d in ready_ops:
            # schedule operation (j, idx) at time est
            starts[j][idx] = est
            old_job_ready = job_ready[j]
            old_machine_ready = machine_ready[m]
            old_job_next = job_next[j]

            job_ready[j] = est + d
            machine_ready[m] = est + d
            job_next[j] = idx + 1

            dfs(scheduled + 1)

            # backtrack
            starts[j][idx] = None
            job_ready[j] = old_job_ready
            machine_ready[m] = old_machine_ready
            job_next[j] = old_job_next

    # ----------------------------------------------------------------------
    # Run the search, fall back to the best solution found if time runs out
    # ----------------------------------------------------------------------
    try:
        dfs(0)
    except StopIteration:
        pass

    # If no solution was found (should never happen) fall back to a trivial schedule
    if best_starts is None:
        # schedule each job sequentially, ignoring machine conflicts
        for j, job in enumerate(jobs):
            t = 0
            for k, (m, d) in enumerate(job):
                starts[j][k] = t
                t += d
        best_starts = starts

    # convert None to 0 (should not appear in optimal solution)
    for row in best_starts:
        for i, v in enumerate(row):
            if v is None:
                row[i] = 0

    return {"starts": best_starts}