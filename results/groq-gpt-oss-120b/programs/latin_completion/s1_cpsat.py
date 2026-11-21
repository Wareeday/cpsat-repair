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

    # Row and column all-different constraints
    for r in range(n):
        model.AddAllDifferent(grid_vars[r])
    for c in range(n):
        model.AddAllDifferent([grid_vars[r][c] for r in range(n)])

    # Prefilled cells
    for r, c, v in prefilled:
        model.Add(grid_vars[r][c] == v)

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    # Use multiple threads if available
    solver.parameters.num_search_workers = max(1, solver.parameters.num_search_workers)

    status = solver.Solve(model)

    # Extract solution if found
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        solution_grid = [
            [solver.Value(grid_vars[r][c]) for c in range(n)]
            for r in range(n)
        ]
    else:
        # No solution found within time limit (should not happen per problem statement)
        solution_grid = [[0] * n for _ in range(n)]

    return {"grid": solution_grid}