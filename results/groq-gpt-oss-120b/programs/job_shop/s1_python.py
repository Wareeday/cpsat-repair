import time
from copy import deepcopy

def solve(instance: dict) -> dict:
    jobs = instance["jobs"]
    n_machines = instance["n_machines"]
    num_jobs = len(jobs)
    ops_per_job = [len(j) for j in jobs]
    offsets = [0] * num_jobs
    for i in range(1, num_jobs):
        offsets[i] = offsets[i - 1] + ops_per_job[i - 1]
    total_ops = offsets[-1] + ops_per_job[-1]

    # flat arrays
    machine = [0] * total_ops
    duration = [0] * total_ops
    prev_op = [None] * total_ops
    job_of = [0] * total_ops
    idx_in_job = [0] * total_ops
    machine_ops = [[] for _ in range(n_machines)]

    for j, job in enumerate(jobs):
        for k, (m, d) in enumerate(job):
            oid = offsets[j] + k
            machine[oid] = m
            duration[oid] = d
            job_of[oid] = j
            idx_in_job[oid] = k
            if k > 0:
                prev_op[oid] = offsets[j] + k - 1
            machine_ops[m].append(oid)

    # helper: earliest feasible start on a machine given its current intervals
    def find_earliest(intervals, earliest, dur):
        t = earliest
        for s, e in intervals:
            if t + dur <= s:
                return t
            if t < e:
                t = e
        return t

    # simple greedy schedule to obtain an initial upper bound
    start = [None] * total_ops
    machine_intervals = [[] for _ in range(n_machines)]
    ready = {oid for oid in range(total_ops) if prev_op[oid] is None}
    while ready:
        # pick any ready operation (smallest id)
        oid = min(ready)
        ready.remove(oid)
        pred = prev_op[oid]
        earliest = 0
        if pred is not None:
            earliest = start[pred] + duration[pred]
        est = find_earliest(machine_intervals[machine[oid]], earliest, duration[oid])
        start[oid] = est
        # insert interval keeping sorted order
        intervals = machine_intervals[machine[oid]]
        i = 0
        while i < len(intervals) and intervals[i][0] < est:
            i += 1
        intervals.insert(i, (est, est + duration[oid]))
        # possibly unlock next operation of the same job
        j = job_of[oid]
        k = idx_in_job[oid] + 1
        if k < ops_per_job[j]:
            nxt = offsets[j] + k
            if prev_op[nxt] is not None and start[prev_op[nxt]] is not None:
                ready.add(nxt)
    greedy_makespan = max(start[oid] + duration[oid] for oid in range(total_ops))

    # best solution storage
    best_makespan = greedy_makespan
    best_start = start[:]  # copy

    # data for search
    start = [None] * total_ops
    machine_intervals = [[] for _ in range(n_machines)]
    start_time = time.time()
    TIME_LIMIT = 28.0  # seconds, margin for safety

    # lower‑bound helper
    def lower_bound(scheduled_cnt):
        # current max completion
        cur_max = 0
        for oid in range(total_ops):
            if start[oid] is not None:
                cur_max = max(cur_max, start[oid] + duration[oid])
        lb = cur_max

        # job based bound
        for j in range(num_jobs):
            last_finish = 0
            for k in range(ops_per_job[j]):
                oid = offsets[j] + k
                if start[oid] is not None:
                    last_finish = max(last_finish, start[oid] + duration[oid])
                else:
                    remaining = sum(duration[offsets[j] + t] for t in range(k, ops_per_job[j]))
                    lb = max(lb, last_finish + remaining)
                    break

        # machine based bound
        for m in range(n_machines):
            remaining = 0
            for oid in machine_ops[m]:
                if start[oid] is None:
                    remaining += duration[oid]
            if machine_intervals[m]:
                machine_free = max(e for (_, e) in machine_intervals[m])
            else:
                machine_free = 0
            lb = max(lb, machine_free + remaining)

        return lb

    # depth‑first branch and bound
    def dfs(scheduled_cnt):
        nonlocal best_makespan, best_start

        # time check
        if time.time() - start_time > TIME_LIMIT:
            return

        if scheduled_cnt == total_ops:
            makespan = max(start[oid] + duration[oid] for oid in range(total_ops))
            if makespan < best_makespan:
                best_makespan = makespan
                best_start = start[:]
            return

        if lower_bound(scheduled_cnt) >= best_makespan:
            return

        # collect ready operations (predecessor done)
        ready_ops = []
        for oid in range(total_ops):
            if start[oid] is None:
                pred = prev_op[oid]
                if pred is None or start[pred] is not None:
                    ready_ops.append(oid)

        # heuristic: choose those with smallest earliest feasible start
        best_est = None
        candidates = []
        for oid in ready_ops:
            pred = prev_op[oid]
            earliest = 0
            if pred is not None:
                earliest = start[pred] + duration[pred]
            est = find_earliest(machine_intervals[machine[oid]], earliest, duration[oid])
            if best_est is None or est < best_est:
                best_est = est
                candidates = [oid]
            elif est == best_est:
                candidates.append(oid)

        for oid in candidates:
            pred = prev_op[oid]
            earliest = 0
            if pred is not None:
                earliest = start[pred] + duration[pred]
            est = find_earliest(machine_intervals[machine[oid]], earliest, duration[oid])

            # assign
            start[oid] = est
            intervals = machine_intervals[machine[oid]]
            i = 0
            while i < len(intervals) and intervals[i][0] < est:
                i += 1
            intervals.insert(i, (est, est + duration[oid]))

            dfs(scheduled_cnt + 1)

            # backtrack
            intervals.pop(i)
            start[oid] = None

    dfs(0)

    # build output structure
    out = {"starts": []}
    for j in range(num_jobs):
        job_starts = []
        for k in range(ops_per_job[j]):
            oid = offsets[j] + k
            job_starts.append(best_start[oid])
        out["starts"].append(job_starts)
    return out