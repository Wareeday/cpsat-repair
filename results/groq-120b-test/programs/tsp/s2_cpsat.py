from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n = instance["n"]
    dist = instance["dist"]

    # Trivial case
    if n == 1:
        return {"tour": [0]}

    model = cp_model.CpModel()

    # Decision variables: x[i][j] == 1 if edge i->j is used in the tour
    x = [[model.NewBoolVar(f"x_{i}_{j}") for j in range(n)] for i in range(n)]

    # No self loops
    for i in range(n):
        model.Add(x[i][i] == 0)

    # Circuit constraint (exactly one outgoing and one incoming arc per node,
    # forming a single Hamiltonian cycle)
    arcs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                arcs.append([i, j, x[i][j]])
    model.AddCircuit(arcs)

    # Objective: minimize total distance
    objective_terms = []
    for i in range(n):
        for j in range(n):
            if i != j:
                objective_terms.append(dist[i][j] * x[i][j])
    model.Minimize(sum(objective_terms))

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    # Extract a feasible tour (if any)
    tour = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Build successor mapping
        succ = {}
        for i in range(n):
            for j in range(n):
                if i != j and solver.BooleanValue(x[i][j]):
                    succ[i] = j
                    break

        # Follow the cycle starting from node 0
        start = 0
        tour.append(start)
        while len(tour) < n:
            start = succ[start]
            tour.append(start)
    else:
        # In the unlikely event no solution is found, fall back to identity order
        tour = list(range(n))

    return {"tour": tour}