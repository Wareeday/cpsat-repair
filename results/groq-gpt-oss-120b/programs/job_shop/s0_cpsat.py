from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    jobs = instance["jobs"]
    n_machines = instance["n_machines"]

    # Horizon = total processing time (upper bound for all times)
    horizon = sum(d for job in jobs for _, d in job)

    model = cp_model.CpModel()

    # Variables and intervals
    job_intervals = []                     # per job: list of (start, end, interval, dur, machine)
    machine_intervals = [[] for _ in range(n_machines)]

    for j, job in enumerate(jobs):
        cur_job = []
        for k, (machine, dur) in enumerate(job):
            suffix = f"j{j}_o{k}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, dur, end, f"interval_{suffix}")
            model.Add(end == start + dur)

            cur_job.append((start, end, interval, dur, machine))
            machine_intervals[machine].append(interval)
        job_intervals.append(cur_job)

        # Precedence within the job
        for k in range(len(cur_job) - 1):
            model.Add(cur_job[k][1] <= cur_job[k + 1][0])   # end_k <= start_{k+1}

    # No‑overlap on each machine
    for m in range(n_machines):
        if machine_intervals[m]:
            model.AddNoOverlap(machine_intervals[m])

    # Makespan variable
    makespan = model.NewIntVar(0, horizon, "makespan")
    for job in job_intervals:
        for start, end, interval, dur, machine in job:
            model.Add(makespan >= end)
    model.Minimize(makespan)

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    # Optional: use multiple threads if available
    # solver.parameters.num_search_workers = 8
    solver.Solve(model)

    # Extract solution
    starts = []
    for job in job_intervals:
        starts.append([int(solver.Value(start)) for start, _, _, _, _ in job])

    return {"starts": starts}