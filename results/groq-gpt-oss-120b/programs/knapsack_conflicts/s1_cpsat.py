from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    capacity = instance["capacity"]
    weights = instance["weights"]
    values = instance["values"]
    conflicts = instance.get("conflicts", [])
    n = len(weights)

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(n)]

    # Capacity constraint
    model.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity)

    # Conflict constraints
    for i, j in conflicts:
        model.Add(x[i] + x[j] <= 1)

    # Objective: maximize total value
    model.Maximize(sum(values[i] * x[i] for i in range(n)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    selected = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(n):
            if solver.Value(x[i]) == 1:
                selected.append(i)

    return {"selected": selected}