import time
from typing import List, Dict, Tuple, Optional

def solve(instance: Dict) -> Dict:
    jobs = instance["jobs"]
    n_machines = instance["n_machines"]
    n_jobs = len(jobs)

    # Flatten operations
    offsets = [0] * n_jobs
    total_ops = 0
    for j, job in enumerate(jobs):
        offsets[j] = total_ops
        total_ops += len(job)

    machine: List[int] = [0] * total_ops
    duration: List[int] = [0] * total_ops
    prev_op: List[Optional[int]] = [None] * total_ops
    machine_ops: List[List[int]] = [[] for _ in range(n_machines)]

    for j, job in enumerate(jobs):
        for k, (m, d) in enumerate(job):
            oid = offsets[j] + k
            machine[oid] = m
            duration[oid] = d
            machine_ops[m].append(oid)
            if k > 0:
                prev_op[oid] = offsets[j] + k - 1

    # Helper: earliest feasible start on a machine given current intervals
    def find_earliest(intervals: List[Tuple[int, int]], earliest: int, dur: int) -> int:
        t = earliest
        for s, e in intervals:
            if t + dur <= s:
                return t
            if t < e:
                t = e
        return t

    # Simple list‑scheduling to obtain an initial feasible makespan
    start = [None] * total_ops
    machine_intervals: List[List[Tuple[int, int]]] = [[] for _ in range(n_machines)]

    unscheduled = set(range(total_ops))
    while unscheduled:
        # pick any ready operation (job predecessor already scheduled)
        ready = None
        for op in unscheduled:
            p = prev_op[op]
            if p is None or start[p] is not None:
                ready = op
                break
        if ready is None:  # should not happen
            break
        m = machine[ready]
        earliest = 0
        if prev_op[ready] is not None:
            earliest = start[prev_op[ready]] + duration[prev_op[ready]]
        earliest = find_earliest(machine_intervals[m], earliest, duration[ready])
        start[ready] = earliest
        # insert interval keeping order
        intervals = machine_intervals[m]
        i = 0
        while i < len(intervals) and intervals[i][0] < earliest:
            i += 1
        intervals.insert(i, (earliest, earliest + duration[ready]))
        unscheduled.remove(ready)

    best_makespan = max(start[i] + duration[i] for i in range(total_ops))
    best_start = start.copy()

    # ---------- Branch‑and‑Bound ----------
    start = [None] * total_ops
    machine_intervals = [[] for _ in range(n_machines)]
    start_time = time.time()
    TIME_LIMIT = 28.0  # seconds

    # Pre‑compute remaining duration per job from each position
    rem_job = []
    for j, job in enumerate(jobs):
        cum = [0] * (len(job) + 1)
        for k in range(len(job) - 1, -1, -1):
            cum[k] = cum[k + 1] + job[k][1]
        rem_job.append(cum)  # cum[k] = sum of durations from operation k onward

    # Pre‑compute remaining duration per machine (simple sum)
    total_dur_machine = [sum(duration[oid] for oid in machine_ops[m]) for m in range(n_machines)]

    def dfs(scheduled_cnt: int) -> None:
        nonlocal best_makespan, best_start
        # time check
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError

        if scheduled_cnt == total_ops:
            makespan = max(start[i] + duration[i] for i in range(total_ops))
            if makespan < best_makespan:
                best_makespan = makespan
                best_start = start.copy()
            return

        # ----- lower bound -----
        cur_max = 0
        for i in range(total_ops):
            if start[i] is not None:
                cur_max = max(cur_max, start[i] + duration[i])

        lb = cur_max

        # job based bound
        for j in range(n_jobs):
            last_finish = 0
            for k in range(len(jobs[j])):
                oid = offsets[j] + k
                if start[oid] is not None:
                    last_finish = max(last_finish, start[oid] + duration[oid])
                else:
                    # remaining work on this job
                    lb = max(lb, last_finish + rem_job[j][k])
                    break

        # machine based bound
        for m in range(n_machines):
            used = 0
            for s, e in machine_intervals[m]:
                used = max(used, e)
            # remaining work on this machine
            rem = 0
            for oid in machine_ops[m]:
                if start[oid] is None:
                    rem += duration[oid]
            lb = max(lb, used + rem)

        if lb >= best_makespan:
            return

        # ----- select ready operations -----
        ready_ops = []
        for op in range(total_ops):
            if start[op] is None:
                p = prev_op[op]
                if p is None or start[p] is not None:
                    ready_ops.append(op)

        # heuristic: choose those with smallest earliest feasible start
        min_ear = None
        candidates = []
        for op in ready_ops:
            m = machine[op]
            earliest = 0
            if prev_op[op] is not None:
                earliest = start[prev_op[op]] + duration[prev_op[op]]
            earliest = find_earliest(machine_intervals[m], earliest, duration[op])
            if (min_ear is None) or (earliest < min_ear):
                min_ear = earliest
                candidates = [op]
            elif earliest == min_ear:
                candidates.append(op)

        # branch on each candidate
        for op in candidates:
            m = machine[op]
            earliest = 0
            if prev_op[op] is not None:
                earliest = start[prev_op[op]] + duration[prev_op[op]]
            earliest = find_earliest(machine_intervals[m], earliest, duration[op])

            # schedule
            start[op] = earliest
            intervals = machine_intervals[m]
            i = 0
            while i < len(intervals) and intervals[i][0] < earliest:
                i += 1
            intervals.insert(i, (earliest, earliest + duration[op]))

            dfs(scheduled_cnt + 1)

            # undo
            intervals.pop(i)
            start[op] = None

    try:
        dfs(0)
    except TimeoutError:
        pass  # return best found so far

    # Build output per job
    out_starts: List[List[int]] = []
    for j in range(n_jobs):
        job_start = []
        for k in range(len(jobs[j])):
            oid = offsets[j] + k
            job_start.append(best_start[oid])
        out_starts.append(job_start)

    return {"starts": out_starts}