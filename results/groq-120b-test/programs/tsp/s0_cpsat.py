from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n = instance["n"]
    dist = instance["dist"]

    # Trivial case
    if n == 1:
        return {"tour": [0]}

    model = cp_model.CpModel()

    # Decision variables: x[i][j] == 1 if we travel from i to j (i != j)
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

    # Add circuit constraint (Hamiltonian circuit)
    arcs = [(i, j, x[(i, j)]) for i in range(n) for j in range(n) if i != j]
    model.AddCircuit(arcs)

    # Objective: minimize total distance
    objective_terms = []
    for (i, j), var in x.items():
        objective_terms.append(dist[i][j] * var)
    model.Minimize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    solver.parameters.log_search_progress = False
    status = solver.Solve(model)

    # Extract solution if any
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # No solution found; return a default trivial tour
        return {"tour": list(range(n))}

    # Build successor map from chosen arcs
    succ = {}
    for (i, j), var in x.items():
        if solver.Value(var) == 1:
            succ[i] = j

    # Reconstruct tour starting from node 0
    tour = [0]
    while len(tour) < n:
        nxt = succ[tour[-1]]
        tour.append(nxt)

    return {"tour": tour}