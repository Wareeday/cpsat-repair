"""Small helper shared by the reference solvers."""
from __future__ import annotations

from ortools.sat.python import cp_model


TIME_LIMIT = 120.0     # seconds; cpsr.generate lowers it with --time-limit


def solve_reference(model: cp_model.CpModel, optimise: bool, time_limit: float | None = None) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit or TIME_LIMIT
    solver.parameters.num_workers = 8
    status = solver.solve(model)
    if optimise and status != cp_model.OPTIMAL:
        raise RuntimeError(f"reference solver did not prove optimality ({solver.status_name(status)})")
    if not optimise and status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"reference solver found no solution ({solver.status_name(status)})")
    return solver
