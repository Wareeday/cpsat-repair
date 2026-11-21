from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    capacity = instance["capacity"]
    sizes = instance["sizes"]
    n = len(sizes)
    max_bins = n  # worst case: one item per bin

    model = cp_model.CpModel()

    # Assignment variables: which bin each item goes to
    x = [model.NewIntVar(0, max_bins - 1, f"x_{i}") for i in range(n)]

    # Helper boolean variables: b[i,k] == 1 iff item i is placed in bin k
    b = {}
    for i in range(n):
        for k in range(max_bins):
            b[(i, k)] = model.NewBoolVar(f"b_{i}_{k}")
            model.Add(x[i] == k).OnlyEnforceIf(b[(i, k)])
            model.Add(x[i] != k).OnlyEnforceIf(b[(i, k)].Not())
        # each item must be assigned to exactly one bin
        model.Add(sum(b[(i, k)] for k in range(max_bins)) == 1)

    # Bin usage indicator variables
    y = [model.NewBoolVar(f"y_{k}") for k in range(max_bins)]

    # Capacity constraints and link y with b (y[k] = OR_i b[i,k])
    for k in range(max_bins):
        model.Add(
            sum(sizes[i] * b[(i, k)] for i in range(n)) <= capacity
        )
        model.AddMaxEquality(y[k], [b[(i, k)] for i in range(n)])

    # Objective: minimise number of used bins
    model.Minimize(sum(y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    solver.Solve(model)

    assignment = [int(solver.Value(x[i])) for i in range(n)]
    return {"assignment": assignment}