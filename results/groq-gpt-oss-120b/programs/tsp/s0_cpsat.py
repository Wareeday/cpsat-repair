from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n = instance["n"]
    dist = instance["dist"]

    model = cp_model.CpModel()
    x = {}
    arcs = []

    # Decision variables: x[i][j] = 1 if edge i->j is used in the tour
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            var = model.NewBoolVar(f"x_{i}_{j}")
            x[(i, j)] = var
            arcs.append([i, j, var])

    # Hamiltonian circuit constraint
    model.AddCircuit(arcs)

    # Objective: minimise total distance
    model.Minimize(
        sum(dist[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20
    # optional parallelism
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    # Build the tour from the solution (if any)
    tour = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # map each node to its successor
        succ = {}
        for i in range(n):
            for j in range(n):
                if i != j and solver.Value(x[(i, j)]) == 1:
                    succ[i] = j
                    break

        # reconstruct the permutation starting from node 0
        cur = 0
        for _ in range(n):
            tour.append(cur)
            cur = succ[cur]
    else:
        # fallback: return identity order (should not happen for valid inputs)
        tour = list(range(n))

    return {"tour": tour}