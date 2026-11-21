from ortools.sat.python import cp_model

def solve(instance: dict) -> dict:
    n = instance["n"]
    blocked = set((r, c) for r, c in instance.get("blocked", []))

    model = cp_model.CpModel()

    # One integer variable per row: the column of the queen in that row.
    queens = [model.NewIntVar(0, n - 1, f"q_{r}") for r in range(n)]

    # Blocked squares: queen cannot be placed there.
    for r, c in blocked:
        model.Add(queens[r] != c)

    # No two queens share a column.
    model.AddAllDifferent(queens)

    # No two queens share a diagonal.
    for i in range(n):
        for j in range(i + 1, n):
            # Main diagonal (row - col) must differ.
            model.Add(queens[i] - i != queens[j] - j)
            # Anti-diagonal (row + col) must differ.
            model.Add(queens[i] + i != queens[j] + j)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0

    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution = [solver.Value(queens[r]) for r in range(n)]
    else:
        # No solution found within the time limit (should not happen per problem statement)
        solution = []

    return {"queens": solution}