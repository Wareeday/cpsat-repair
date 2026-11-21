from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n = instance["n"]
    blocked = instance.get("blocked", [])

    # Map rows to set of blocked columns
    blocked_in_row = {r: set() for r in range(n)}
    for r, c in blocked:
        blocked_in_row[r].add(c)

    model = cp_model.CpModel()

    # Decision variables: column of queen in each row
    cols = []
    for r in range(n):
        var = model.NewIntVar(0, n - 1, f"col_{r}")
        # forbid blocked columns
        for c in blocked_in_row[r]:
            model.Add(var != c)
        cols.append(var)

    # AllDifferent on columns
    model.AddAllDifferent(cols)

    # Auxiliary variables for the two diagonal families
    diag1 = []  # col + row
    diag2 = []  # col - row
    for r in range(n):
        d1 = model.NewIntVar(0, 2 * n - 2, f"diag1_{r}")
        d2 = model.NewIntVar(-n + 1, n - 1, f"diag2_{r}")
        model.Add(d1 == cols[r] + r)
        model.Add(d2 == cols[r] - r)
        diag1.append(d1)
        diag2.append(d2)

    model.AddAllDifferent(diag1)
    model.AddAllDifferent(diag2)

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    # Extract solution (feasible or optimal)
    queens = [solver.Value(cols[r]) for r in range(n)]

    return {"queens": queens}