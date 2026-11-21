from ortools.sat.python import cp_model
import math

def solve(instance: dict) -> dict:
    capacity = instance["capacity"]
    sizes = instance["sizes"]
    n = len(sizes)

    # ---------- Upper bound via First-Fit Decreasing (quick heuristic) ----------
    sorted_sizes = sorted(sizes, reverse=True)
    ffd_bins = []
    for sz in sorted_sizes:
        placed = False
        for i in range(len(ffd_bins)):
            if sz <= ffd_bins[i]:
                ffd_bins[i] -= sz
                placed = True
                break
        if not placed:
            ffd_bins.append(capacity - sz)
    ub = len(ffd_bins) if ffd_bins else 0
    max_bins = max(1, ub)          # at least one bin variable

    # ---------- Lower bound (optional, not used directly) ----------
    total_size = sum(sizes)
    lb = math.ceil(total_size / capacity)

    model = cp_model.CpModel()

    # Assignment variables: bin index for each item
    x = [model.NewIntVar(0, max_bins - 1, f"x_{i}") for i in range(n)]

    # Boolean matrix linking items to bins
    b = {}
    for i in range(n):
        for k in range(max_bins):
            b[(i, k)] = model.NewBoolVar(f"b_{i}_{k}")
            model.Add(x[i] == k).OnlyEnforceIf(b[(i, k)])
            model.Add(x[i] != k).OnlyEnforceIf(b[(i, k)].Not())
        model.Add(sum(b[(i, k)] for k in range(max_bins)) == 1)

    # Bin usage indicators
    y = [model.NewBoolVar(f"y_{k}") for k in range(max_bins)]

    for k in range(max_bins):
        # capacity constraint
        model.Add(
            sum(sizes[i] * b[(i, k)] for i in range(n)) <= capacity
        )
        # y[k] = OR_i b[i,k]
        model.AddMaxEquality(y[k], [b[(i, k)] for i in range(n)])

    # Symmetry breaking: used bins form a prefix
    for k in range(max_bins - 1):
        model.Add(y[k] >= y[k + 1])

    # Objective: minimise number of used bins
    model.Minimize(sum(y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    solver.Solve(model)

    assignment = [int(solver.Value(x[i])) for i in range(n)]
    return {"assignment": assignment}