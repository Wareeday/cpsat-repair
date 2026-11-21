from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n = instance["n"]
    edges = instance.get("edges", [])

    model = cp_model.CpModel()

    # colour variables: 0 .. n-1 (upper bound can never be larger than n-1)
    colors = [model.NewIntVar(0, n - 1, f"c{i}") for i in range(n)]
    # variable representing the highest colour used
    max_color = model.NewIntVar(0, n - 1, "max_color")

    # each vertex colour must be <= max_color
    for v in range(n):
        model.Add(colors[v] <= max_color)

    # edge constraints: adjacent vertices must differ
    for u, v in edges:
        model.Add(colors[u] != colors[v])

    # symmetry breaking: fix the colour of the first vertex to 0
    if n > 0:
        model.Add(colors[0] == 0)

    # objective: minimise the highest colour used (=> minimise number of distinct colours)
    model.Minimize(max_color)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    # optional parallelism (does not hurt if not supported)
    solver.parameters.num_search_workers = 8

    solver.Solve(model)

    # extract solution
    solution = [int(solver.Value(col)) for col in colors]

    return {"colors": solution}