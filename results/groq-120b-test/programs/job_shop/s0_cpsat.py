from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    jobs = instance["jobs"]
    n_machines = instance["n_machines"]

    # Compute a safe horizon (sum of all durations)
    horizon = sum(op[1] for job in jobs for op in job)

    model = cp_model.CpModel()

    # Variables
    starts = []
    intervals = []  # same shape as starts, store interval vars or None
    for j, job in enumerate(jobs):
        job_starts = []
        job_intervals = []
        for k, (machine, duration) in enumerate(job):
            s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
            e = model.NewIntVar(0, horizon, f"e_{j}_{k}")
            interval = model.NewIntervalVar(s, duration, e, f"int_{j}_{k}")
            job_starts.append(s)
            job_intervals.append(interval)
        starts.append(job_starts)
        intervals.append(job_intervals)

    # Precedence constraints within each job
    for j, job in enumerate(jobs):
        for k in range(1, len(job)):
            prev_dur = job[k - 1][1]
            model.Add(starts[j][k] >= starts[j][k - 1] + prev_dur)

    # Machine capacity constraints (no overlap)
    machine_to_intervals = {m: [] for m in range(n_machines)}
    for j, job in enumerate(jobs):
        for k, (machine, _) in enumerate(job):
            machine_to_intervals[machine].append(intervals[j][k])
    for m in range(n_machines):
        if machine_to_intervals[m]:
            model.AddNoOverlap(machine_to_intervals[m])

    # Makespan variable
    makespan = model.NewIntVar(0, horizon, "makespan")
    for j, job in enumerate(jobs):
        for k, (_, duration) in enumerate(job):
            model.Add(makespan >= starts[j][k] + duration)

    model.Minimize(makespan)

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    # Extract solution (if any)
    result_starts = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for j, job in enumerate(jobs):
            result_starts.append([solver.Value(starts[j][k]) for k in range(len(job))])
    else:
        # No feasible solution found within time limit; return empty starts
        result_starts = [[0 for _ in job] for job in jobs]

    return {"starts": result_starts}