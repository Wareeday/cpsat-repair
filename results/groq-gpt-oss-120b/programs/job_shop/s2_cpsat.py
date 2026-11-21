from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    jobs = instance["jobs"]
    n_machines = instance["n_machines"]

    # Compute a horizon (upper bound for all start times)
    horizon = sum(op[1] for job in jobs for op in job)

    model = cp_model.CpModel()

    # Variables
    starts = []
    ends = []
    intervals = []          # parallel list of interval variables
    for j, job in enumerate(jobs):
        job_starts = []
        job_ends = []
        job_intervals = []
        for k, (machine, duration) in enumerate(job):
            s = model.NewIntVar(0, horizon, f"start_{j}_{k}")
            e = model.NewIntVar(0, horizon, f"end_{j}_{k}")
            interval = model.NewIntervalVar(s, duration, e, f"interval_{j}_{k}")
            job_starts.append(s)
            job_ends.append(e)
            job_intervals.append(interval)
        starts.append(job_starts)
        ends.append(job_ends)
        intervals.append(job_intervals)

    # Precedence constraints within each job
    for j, job in enumerate(jobs):
        for k in range(len(job) - 1):
            model.Add(ends[j][k] <= starts[j][k + 1])

    # Machine capacity constraints (no overlap)
    machine_to_intervals = [[] for _ in range(n_machines)]
    for j, job in enumerate(jobs):
        for k, (machine, _) in enumerate(job):
            machine_to_intervals[machine].append(intervals[j][k])

    for m in range(n_machines):
        if machine_to_intervals[m]:
            model.AddNoOverlap(machine_to_intervals[m])

    # Makespan variable
    makespan = model.NewIntVar(0, horizon, "makespan")
    all_ends = [e for job_ends in ends for e in job_ends]
    model.AddMaxEquality(makespan, all_ends)
    model.Minimize(makespan)

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    # Extract solution (if feasible)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # No feasible solution found within time limit – return zeros
        return {"starts": [[0] * len(job) for job in jobs]}

    result_starts = []
    for j, job in enumerate(jobs):
        result_starts.append([solver.Value(starts[j][k]) for k in range(len(job))])

    return {"starts": result_starts}