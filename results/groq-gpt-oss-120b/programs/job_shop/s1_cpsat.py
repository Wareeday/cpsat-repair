from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    jobs = instance["jobs"]
    n_jobs = len(jobs)
    n_machines = instance["n_machines"]

    # Compute horizon as sum of all durations (safe upper bound)
    horizon = sum(op[1] for job in jobs for op in job)

    model = cp_model.CpModel()

    # Create variables
    start_vars = []   # start_vars[j][k] = start time of k-th op of job j
    end_vars = []     # end_vars[j][k] = end time (start + duration)
    interval_vars = []  # interval_vars[j][k] = interval variable
    for j, job in enumerate(jobs):
        job_starts = []
        job_ends = []
        job_intervals = []
        for k, (machine, duration) in enumerate(job):
            s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
            e = model.NewIntVar(0, horizon, f"e_{j}_{k}")
            interval = model.NewIntervalVar(s, duration, e, f"int_{j}_{k}")
            job_starts.append(s)
            job_ends.append(e)
            job_intervals.append((machine, interval))
        start_vars.append(job_starts)
        end_vars.append(job_ends)
        interval_vars.append(job_intervals)

        # Job order constraints
        for k in range(len(job) - 1):
            model.Add(end_vars[j][k] <= start_vars[j][k + 1])

    # Machine capacity constraints (no overlap)
    for m in range(n_machines):
        intervals_on_m = []
        for j in range(n_jobs):
            for k, (machine, interval) in enumerate(interval_vars[j]):
                if machine == m:
                    intervals_on_m.append(interval)
        if intervals_on_m:
            model.AddNoOverlap(intervals_on_m)

    # Makespan variable
    makespan = model.NewIntVar(0, horizon, "makespan")
    for j in range(n_jobs):
        for k in range(len(jobs[j])):
            model.Add(end_vars[j][k] <= makespan)

    model.Minimize(makespan)

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    # optional: use multiple threads if available
    solver.parameters.num_search_workers = max(1, solver.parameters.num_search_workers)

    status = solver.Solve(model)

    # Extract solution (if any)
    result_starts = []
    for j in range(n_jobs):
        job_start_times = []
        for k in range(len(jobs[j])):
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                job_start_times.append(int(solver.Value(start_vars[j][k])))
            else:
                # fallback to 0 if no solution (should not happen in tests)
                job_start_times.append(0)
        result_starts.append(job_start_times)

    return {"starts": result_starts}