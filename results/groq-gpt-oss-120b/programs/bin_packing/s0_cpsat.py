from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    capacity = instance["capacity"]
    sizes = instance["sizes"]
    n = len(sizes)

    # Trivial case
    if n == 0:
        return {"assignment": []}

    max_bins = n  # worst case one item per bin

    model = cp_model.CpModel()

    # Decision variables: x[i,b] = 1 if item i is placed in bin b
    x = {}
    for i in range(n):
        for b in range(max_bins):
            x[i, b] = model.NewBoolVar(f"x_{i}_{b}")

    # Bin usage indicator
    bin_used = [model.NewBoolVar(f"bin_used_{b}") for b in range(max_bins)]

    # Each item must be assigned to exactly one bin
    for i in range(n):
        model.Add(sum(x[i, b] for b in range(max_bins)) == 1)

    # Capacity constraints and linking bin_used with assignments
    for b in range(max_bins):
        model.Add(
            sum(sizes[i] * x[i, b] for i in range(n)) <= capacity * bin_used[b]
        )
        for i in range(n):
            model.Add(x[i, b] <= bin_used[b])

    # Symmetry breaking: bins are used in order
    for b in range(max_bins - 1):
        model.Add(bin_used[b] >= bin_used[b + 1])

    # Optional: force first item into first bin to reduce symmetry further
    model.Add(x[0, 0] == 1)

    # Objective: minimise number of used bins
    model.Minimize(sum(bin_used))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    # Extract assignment (best found if feasible)
    assignment = [0] * n
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(n):
            for b in range(max_bins):
                if solver.BooleanValue(x[i, b]):
                    assignment[i] = b
                    break

    return {"assignment": assignment}