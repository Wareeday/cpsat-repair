from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n = instance["n"]
    dist = instance["dist"]

    model = cp_model.CpModel()

    # Decision variables: x[i][j] = 1 if edge i->j is used in the tour
    x = {}
    arcs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            var = model.NewBoolVar(f"x_{i}_{j}")
            x[(i, j)] = var
            arcs.append([i, j, var])

    # Circuit constraint enforces a single Hamiltonian cycle
    model.AddCircuit(arcs)

    # Objective: minimise total distance of the selected edges
    objective_terms = []
    for (i, j), var in x.items():
        objective_terms.append(dist[i][j] * var)
    model.Minimize(sum(objective_terms))

    # Solve with a time limit
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # No feasible solution found within the time limit; return empty tour
        return {"tour": []}

    # Reconstruct the tour from the selected arcs
    next_node = {}
    for (i, j), var in x.items():
        if solver.Value(var):
            next_node[i] = j

    # Build ordered tour starting from city 0
    tour = []
    current = 0
    for _ in range(n):
        tour.append(current)
        current = next_node[current]

    return {"tour": tour}