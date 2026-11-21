from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n = instance["n"]
    blocked = instance.get("blocked", [])

    model = cp_model.CpModel()

    # One integer variable per row: column of the queen
    cols = [model.NewIntVar(0, n - 1, f"col_{r}") for r in range(n)]

    # Blocked squares: queen cannot be placed on them
    blocked_by_row = {}
    for r, c in blocked:
        blocked_by_row.setdefault(r, set()).add(c)
    for r, cs in blocked_by_row.items():
        for c in cs:
            model.Add(cols[r] != c)

    # No two queens share a column
    model.AddAllDifferent(cols)

    # Diagonal constraints using transformed variables
    diag1 = [model.NewIntVar(0, 2 * n, f"diag1_{r}") for r in range(n)]   # col + row
    diag2 = [model.NewIntVar(-n, n, f"diag2_{r}") for r in range(n)]      # col - row
    for r in range(n):
        model.Add(diag1[r] == cols[r] + r)
        model.Add(diag2[r] == cols[r] - r)

    model.AddAllDifferent(diag1)
    model.AddAllDifferent(diag2)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        queens = [solver.Value(cols[r]) for r in range(n)]
        return {"queens": queens}
    else:
        # No solution found within the time limit (should not happen per problem statement)
        return {"queens": []}