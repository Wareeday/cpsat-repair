from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    # Extract data
    capacity = int(instance.get("capacity", 0))
    weights = list(instance.get("weights", []))
    values = list(instance.get("values", []))
    conflicts = list(instance.get("conflicts", []))

    n = len(weights)
    # Model
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(n)]

    # Capacity constraint
    model.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity)

    # Conflict constraints
    for i, j in conflicts:
        if 0 <= i < n and 0 <= j < n:
            model.Add(x[i] + x[j] <= 1)

    # Objective: maximize total value
    model.Maximize(sum(values[i] * x[i] for i in range(n)))

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    selected = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(n):
            if solver.Value(x[i]) == 1:
                selected.append(i)

    return {"selected": selected}