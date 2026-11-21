from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n = instance["n"]
    dist = instance["dist"]

    # Trivial case n == 1
    if n == 1:
        return {"tour": [0]}

    model = cp_model.CpModel()

    # Decision variables: x[i][j] == 1 if edge i->j is used (i != j)
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

    # Each city has exactly one outgoing edge
    for i in range(n):
        model.Add(sum(x[(i, j)] for j in range(n) if i != j) == 1)

    # Each city has exactly one incoming edge
    for j in range(n):
        model.Add(sum(x[(i, j)] for i in range(n) if i != j) == 1)

    # Circuit constraint to eliminate subtours
    arcs = [(i, j, x[(i, j)]) for i in range(n) for j in range(n) if i != j]
    model.AddCircuit(arcs)

    # Objective: minimize total distance
    objective_terms = [dist[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j]
    model.Minimize(sum(objective_terms))

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    # If no feasible solution was found, return an empty tour (should not happen)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"tour": []}

    # Reconstruct the tour from the solution
    successor = {}
    for i in range(n):
        for j in range(n):
            if i != j and solver.BooleanValue(x[(i, j)]):
                successor[i] = j
                break

    # Build ordered tour starting from 0
    tour = [0]
    while len(tour) < n:
        tour.append(successor[tour[-1]])

    return {"tour": tour}