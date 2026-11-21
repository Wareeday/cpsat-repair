from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    # Extract data
    n_employees = instance["n_employees"]
    n_days = instance["n_days"]
    demand = instance["demand"]               # list of [m,a,n] per day
    requests_off = instance.get("requests_off", [])
    min_shifts = instance["min_shifts"]
    max_shifts = instance["max_shifts"]

    model = cp_model.CpModel()

    # work[e][d][s] == 1 iff employee e works shift s on day d
    work = {}
    for e in range(n_employees):
        for d in range(n_days):
            for s in range(3):
                work[e, d, s] = model.NewBoolVar(f"w_{e}_{d}_{s}")

    # 1) At most one shift per employee per day
    for e in range(n_employees):
        for d in range(n_days):
            model.Add(sum(work[e, d, s] for s in range(3)) <= 1)

    # 2) Exact staffing demand per shift per day
    for d in range(n_days):
        for s in range(3):
            model.Add(
                sum(work[e, d, s] for e in range(n_employees))
                == demand[d][s]
            )

    # 3) Night‑to‑morning restriction
    for e in range(n_employees):
        for d in range(n_days - 1):
            model.Add(work[e, d, 2] + work[e, d + 1, 0] <= 1)

    # 4) Total shifts per employee within [min_shifts, max_shifts]
    for e in range(n_employees):
        total = sum(work[e, d, s] for d in range(n_days) for s in range(3))
        model.Add(total >= min_shifts)
        model.Add(total <= max_shifts)

    # Objective: minimise violated day‑off requests
    penalty_terms = []
    for e, d in requests_off:
        penalty_terms.append(sum(work[e, d, s] for s in range(3)))
    if penalty_terms:
        model.Minimize(sum(penalty_terms))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    # Build roster output
    roster = []
    for e in range(n_employees):
        row = []
        for d in range(n_days):
            assigned_shift = -1
            for s in range(3):
                if solver.Value(work[e, d, s]) == 1:
                    assigned_shift = s
                    break
            row.append(assigned_shift)
        roster.append(row)

    return {"roster": roster}