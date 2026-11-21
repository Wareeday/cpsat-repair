from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    # Unpack instance
    E = instance["n_employees"]
    D = instance["n_days"]
    demand = instance["demand"]            # list of [m,a,n] per day
    requests_off = instance.get("requests_off", [])
    min_shifts = instance["min_shifts"]
    max_shifts = instance["max_shifts"]

    model = cp_model.CpModel()

    # Decision variables: shift[e][d] in {-1,0,1,2}
    shift = {}
    for e in range(E):
        for d in range(D):
            shift[e, d] = model.NewIntVar(-1, 2, f"shift_e{e}_d{d}")

    # Helper booleans for each (e,d,s) and off
    is_shift = {}
    is_off = {}
    for e in range(E):
        for d in range(D):
            # off
            is_off[e, d] = model.NewBoolVar(f"off_e{e}_d{d}")
            model.Add(shift[e, d] == -1).OnlyEnforceIf(is_off[e, d])
            model.Add(shift[e, d] != -1).OnlyEnforceIf(is_off[e, d].Not())

            for s in range(3):
                is_shift[e, d, s] = model.NewBoolVar(f"shift_{e}_{d}_{s}")
                model.Add(shift[e, d] == s).OnlyEnforceIf(is_shift[e, d, s])
                model.Add(shift[e, d] != s).OnlyEnforceIf(is_shift[e, d, s].Not())

            # Exactly one of {off, 0,1,2} must be true
            model.AddExactlyOne(is_off[e, d],
                                is_shift[e, d, 0],
                                is_shift[e, d, 1],
                                is_shift[e, d, 2])

    # Demand constraints
    for d in range(D):
        for s in range(3):
            model.Add(
                sum(is_shift[e, d, s] for e in range(E)) == demand[d][s]
            )

    # Night‑to‑morning restriction
    for e in range(E):
        for d in range(D - 1):
            # if night today then not morning tomorrow
            model.Add(is_shift[e, d, 2] == 0).OnlyEnforceIf(is_shift[e, d, 2].Not())
            model.Add(is_shift[e, d + 1, 0] == 0).OnlyEnforceIf(is_shift[e, d, 2])

    # Shift count per employee
    for e in range(E):
        total_shifts = sum(is_shift[e, d, s] for d in range(D) for s in range(3))
        model.Add(total_shifts >= min_shifts)
        model.Add(total_shifts <= max_shifts)

    # Objective: minimise violated requests_off
    penalty_terms = []
    for (e, d) in requests_off:
        # penalty if employee works (i.e., not off)
        penalty = model.NewIntVar(0, 1, f"pen_e{e}_d{d}")
        model.Add(penalty == 1 - is_off[e, d])
        penalty_terms.append(penalty)

    if penalty_terms:
        model.Minimize(sum(penalty_terms))

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    # Allow non‑optimal solution if time runs out
    solver.parameters.enumerate_all_solutions = False

    status = solver.Solve(model)

    # Build roster from solution (fallback to -1 if not solved)
    roster = [[-1 for _ in range(D)] for _ in range(E)]
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for e in range(E):
            for d in range(D):
                roster[e][d] = int(solver.Value(shift[e, d]))

    return {"roster": roster}