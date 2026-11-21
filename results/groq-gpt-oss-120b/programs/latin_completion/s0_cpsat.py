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
        col_vars = [grid_vars[r][c] for r in range(n)]
        model.AddAllDifferent(col_vars)

    # Prefilled cells
    for r, c, v in prefilled:
        model.Add(grid_vars[r][c] == v)

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    solver.parameters.random_seed = 0  # deterministic behaviour

    status = solver.Solve(model)

    # Extract solution (feasible or optimal)
    result_grid = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            result_grid[r][c] = solver.Value(grid_vars[r][c])

    return {"grid": result_grid}