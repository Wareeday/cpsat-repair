from ortools.sat.python import cp_model
from typing import List, Dict

def _first_fit_decreasing(sizes: List[int], capacity: int) -> int:
    """Simple heuristic to obtain an upper bound on the number of bins."""
    bins = []
    for s in sorted(sizes, reverse=True):
        placed = False
        for i in range(len(bins)):
            if bins[i] + s <= capacity:
                bins[i] += s
                placed = True
                break
        if not placed:
            bins.append(s)
    return len(bins)


def solve(instance: Dict) -> Dict:
    capacity = instance["capacity"]
    sizes = instance["sizes"]
    n = len(sizes)

    # Upper bound on number of bins using a fast heuristic
    ub = _first_fit_decreasing(sizes, capacity)

    model = cp_model.CpModel()

    # Decision variables
    # x[i][b] = 1 if item i is placed in bin b (b in 0..ub-1)
    x = [[model.NewBoolVar(f"x_{i}_{b}") for b in range(ub)] for i in range(n)]
    # y[b] = 1 if bin b is used
    y = [model.NewBoolVar(f"y_{b}") for b in range(ub)]

    # Each item assigned to exactly one bin
    for i in range(n):
        model.Add(sum(x[i][b] for b in range(ub)) == 1)

    # Capacity constraints and linking x to y
    for b in range(ub):
        model.Add(
            sum(sizes[i] * x[i][b] for i in range(n)) <= capacity * y[b]
        )
        for i in range(n):
            model.Add(x[i][b] <= y[b])

    # Symmetry breaking: used bins form a prefix (no gaps)
    for b in range(ub - 1):
        model.Add(y[b] >= y[b + 1])

    # Objective: minimise number of used bins
    model.Minimize(sum(y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    # Encourage finding a feasible solution quickly
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    # Build assignment from the solution (even if not proven optimal)
    assignment = [-1] * n
    for i in range(n):
        for b in range(ub):
            if solver.Value(x[i][b]) == 1:
                assignment[i] = b
                break
        if assignment[i] == -1:  # fallback, should not happen
            assignment[i] = 0

    return {"assignment": assignment}