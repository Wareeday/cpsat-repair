from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n_employees = instance["n_employees"]
    n_days = instance["n_days"]
    demand = instance["demand"]               # list of [m,a,n] per day
    requests_off = instance.get("requests_off", [])
    min_shifts = instance["min_shifts"]
    max_shifts = instance["max_shifts"]

    model = cp_model.CpModel()

    # Boolean variables: works[(e,d,s)] == 1 iff employee e works shift s on day d
    works = {}
    for e in range(n_employees):
        for d in range(n_days):
            for s in range(3):
                works[(e, d, s)] = model.NewBoolVar(f"w_{e}_{d}_{s}")

    # Boolean variables: off[(e,d)] == 1 iff employee e is off on day d
    off = {}
    for e in range(n_employees):
        for d in range(n_days):
            off[(e, d)] = model.NewBoolVar(f"off_{e}_{d}")

    # (1) Each employee works at most one shift per day (exactly one assignment)
    for e in range(n_employees):
        for d in range(n_days):
            model.Add(
                sum(works[(e, d, s)] for s in range(3)) + off[(e, d)] == 1
            )

    # (2) Daily demand for each shift
    for d in range(n_days):
        for s in range(3):
            model.Add(
                sum(works[(e, d, s)] for e in range(n_employees)) == demand[d][s]
            )

    # (3) Night‑to‑morning restriction
    for e in range(n_employees):
        for d in range(n_days - 1):
            model.Add(works[(e, d, 2)] + works[(e, d + 1, 0)] <= 1)

    # (4) Total shifts per employee within [min_shifts, max_shifts]
    for e in range(n_employees):
        total_shifts = sum(works[(e, d, s)] for d in range(n_days) for s in range(3))
        model.Add(total_shifts >= min_shifts)
        model.Add(total_shifts <= max_shifts)

    # Objective: minimise violated days‑off requests
    request_violations = []
    for (e, d) in requests_off:
        viol = model.NewIntVar(0, 1, f"req_{e}_{d}")
        model.Add(viol == sum(works[(e, d, s)] for s in range(3)))
        request_violations.append(viol)

    if request_violations:
        model.Minimize(sum(request_violations))

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # No feasible solution found (should not happen per statement)
        raise ValueError("No feasible roster found.")

    # Build roster output
    roster = []
    for e in range(n_employees):
        row = []
        for d in range(n_days):
            assigned_shift = -1
            for s in range(3):
                if solver.Value(works[(e, d, s)]) == 1:
                    assigned_shift = s
                    break
            row.append(assigned_shift)
        roster.append(row)

    return {"roster": roster}