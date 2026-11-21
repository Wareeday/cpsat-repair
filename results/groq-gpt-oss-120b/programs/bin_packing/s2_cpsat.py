from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    capacity = instance["capacity"]
    sizes = instance["sizes"]
    n = len(sizes)

    model = cp_model.CpModel()

    # Decision variables
    # x[i][b] = 1 if item i is placed in bin b
    x = [[model.NewBoolVar(f"x_{i}_{b}") for b in range(n)] for i in range(n)]
    # y[b] = 1 if bin b is used
    y = [model.NewBoolVar(f"y_{b}") for b in range(n)]

    # Each item must be assigned to exactly one bin
    for i in range(n):
        model.Add(sum(x[i][b] for b in range(n)) == 1)

    # Capacity constraints
    for b in range(n):
        model.Add(
            sum(sizes[i] * x[i][b] for i in range(n)) <= capacity * y[b]
        )

    # Link x and y: if an item is placed in bin b then bin b must be used
    for i in range(n):
        for b in range(n):
            model.Add(x[i][b] <= y[b])

    # Objective: minimize number of used bins
    model.Minimize(sum(y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    # Extract assignment (even if status is not optimal)
    assignment = [-1] * n
    for i in range(n):
        for b in range(n):
            if solver.Value(x[i][b]) == 1:
                assignment[i] = b
                break
        # Fallback safety (should never happen)
        if assignment[i] == -1:
            assignment[i] = 0

    return {"assignment": assignment}