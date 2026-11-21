from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    jobs = instance["jobs"]
    n_machines = instance["n_machines"]

    model = cp_model.CpModel()

    # Compute a horizon as the sum of all durations (safe upper bound)
    horizon = sum(op[1] for job in jobs for op in job)

    # Variables
    starts = []          # starts[j][k] = start time variable of operation k of job j
    ends = []            # corresponding end time variables
    intervals = []       # interval variables for NoOverlap constraints
    for j, job in enumerate(jobs):
        job_starts = []
        job_ends = []
        job_intervals = []
        for k, (machine, duration) in enumerate(job):
            s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
            e = model.NewIntVar(0, horizon, f"e_{j}_{k}")
            iv = model.NewIntervalVar(s, duration, e, f"iv_{j}_{k}")
            job_starts.append(s)
            job_ends.append(e)
            job_intervals.append((machine, iv))
        starts.append(job_starts)
        ends.append(job_ends)
        intervals.append(job_intervals)

    # Precedence constraints within each job
    for j, job in enumerate(jobs):
        for k in range(1, len(job)):
            model.Add(starts[j][k] >= ends[j][k - 1])

    # Machine capacity constraints (NoOverlap)
    for m in range(n_machines):
        machine_intervals = [iv for job_ints in intervals for mach, iv in job_ints if mach == m]
        if machine_intervals:
            model.AddNoOverlap(machine_intervals)

    # Makespan variable
    makespan = model.NewIntVar(0, horizon, "makespan")
    for j in range(len(jobs)):
        for k in range(len(jobs[j])):
            model.Add(makespan >= ends[j][k])
    model.Minimize(makespan)

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    # optional: suppress output
    solver.parameters.log_search_progress = False
    status = solver.Solve(model)

    # Extract solution (if any)
    result_starts = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for j, job in enumerate(jobs):
            result_starts.append([solver.Value(starts[j][k]) for k in range(len(job))])
    else:
        # Infeasible or unknown: return empty schedule (should not happen for valid instances)
        result_starts = [[0 for _ in job] for job in jobs]

    return {"starts": result_starts}