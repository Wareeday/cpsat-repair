from typing import List, Dict
from ortools.sat.python import cp_model


def solve(instance: Dict) -> Dict:
    jobs: List[List[List[int]]] = instance["jobs"]
    n_machines: int = instance["n_machines"]

    num_jobs = len(jobs)

    # Compute a safe horizon (upper bound on makespan)
    horizon = sum(op[1] for job in jobs for op in job)

    model = cp_model.CpModel()

    # Variables
    start_vars: List[List[cp_model.IntVar]] = []
    end_vars: List[List[cp_model.IntVar]] = []
    interval_vars: List[List[cp_model.IntervalVar]] = []

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
            job_intervals.append(interval)
        start_vars.append(job_starts)
        end_vars.append(job_ends)
        interval_vars.append(job_intervals)

    # Job precedence constraints
    for j, job in enumerate(jobs):
        for k in range(len(job) - 1):
            model.Add(end_vars[j][k] <= start_vars[j][k + 1])

    # Machine capacity constraints (no overlap)
    machine_to_intervals: Dict[int, List[cp_model.IntervalVar]] = {m: [] for m in range(n_machines)}
    for j, job in enumerate(jobs):
        for k, (machine, _) in enumerate(job):
            machine_to_intervals[machine].append(interval_vars[j][k])

    for m in range(n_machines):
        if machine_to_intervals[m]:
            model.AddNoOverlap(machine_to_intervals[m])

    # Makespan variable
    makespan = model.NewIntVar(0, horizon, "makespan")
    for j in range(num_jobs):
        for k in range(len(jobs[j])):
            model.Add(end_vars[j][k] <= makespan)

    model.Minimize(makespan)

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    # Extract solution (feasible or optimal)
    starts: List[List[int]] = []
    for j, job in enumerate(jobs):
        job_starts = []
        for k in range(len(job)):
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                job_starts.append(int(solver.Value(start_vars[j][k])))
            else:
                job_starts.append(0)  # fallback, should not happen
        starts.append(job_starts)

    return {"starts": starts}