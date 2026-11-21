from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n = instance["n"]
    prefilled = instance.get("prefilled", [])

    model = cp_model.CpModel()

    # Decision variables: grid[r][c] in 1..n
    grid_vars = [
        [model.NewIntVar(1, n, f"x_{r}_{c}") for c in range(n)]
        for r in range(n)
    ]

    # Row all-different constraints
    for r in range(n):
        model.AddAllDifferent(grid_vars[r])

    # Column all-different constraints
    for c in range(n):
        model.AddAllDifferent([grid_vars[r][c] for r in range(n)])

    # Prefilled cells
    for r, c, v in prefilled:
        model.Add(grid_vars[r][c] == v)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No solution found within time limit.")

    result_grid = [
        [solver.Value(grid_vars[r][c]) for c in range(n)]
        for r in range(n)
    ]

    return {"grid": result_grid}