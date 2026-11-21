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

    # One Hamiltonian circuit covering all nodes
    model.AddCircuit(arcs)

    # Objective: minimize total distance
    objective = sum(dist[i][j] * x[(i, j)] for (i, j) in x)
    model.Minimize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    # Use multiple workers if available (optional)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # No feasible solution found within time limit
        return {"tour": []}

    # Reconstruct the tour starting from city 0
    tour = [0]
    current = 0
    while len(tour) < n:
        for j in range(n):
            if j == current:
                continue
            if solver.Value(x[(current, j)]) == 1:
                tour.append(j)
                current = j
                break

    return {"tour": tour}