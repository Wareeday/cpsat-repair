from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n = instance["n"]
    edges = instance.get("edges", [])

    model = cp_model.CpModel()

    # colour variables for each vertex, domain 0..n-1
    colors = [model.NewIntVar(0, n - 1, f"c{i}") for i in range(n)]

    # max colour variable
    max_color = model.NewIntVar(0, n - 1, "max_color")
    for i in range(n):
        model.Add(colors[i] <= max_color)

    # edge constraints: adjacent vertices must differ
    for u, v in edges:
        model.Add(colors[u] != colors[v])

    # objective: minimise the maximum colour used
    model.Minimize(max_color)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    # optional: use multiple threads if available
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    # Extract solution (even if not proven optimal)
    result_colors = [solver.Value(colors[i]) for i in range(n)]

    return {"colors": result_colors}