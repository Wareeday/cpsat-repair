from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n = instance.get("n", 0)
    edges = instance.get("edges", [])

    # Trivial case
    if n == 0:
        return {"colors": []}

    model = cp_model.CpModel()

    # Color variables: each vertex gets a colour in [0, n-1]
    colors = [model.NewIntVar(0, n - 1, f"c{i}") for i in range(n)]

    # Max colour variable (upper bound on used colours)
    max_color = model.NewIntVar(0, n - 1, "max_color")
    for c in colors:
        model.Add(max_color >= c)

    # Edge constraints: adjacent vertices must have different colours
    for u, v in edges:
        model.Add(colors[u] != colors[v])

    # Objective: minimise the highest colour used (hence the number of distinct colours)
    model.Minimize(max_color)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    # Retrieve solution if any feasible solution was found
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result_colors = [solver.Value(c) for c in colors]
    else:
        # Should not happen for valid inputs, but return a default colouring
        result_colors = list(range(n))

    return {"colors": result_colors}