from __future__ import annotations

import random
from typing import Optional

from ortools.sat.python import cp_model

from ._cp import solve_reference
from .base import Family, Violation, as_int, as_list, get_field

SHIFT_NAMES = ["morning", "afternoon", "night"]


class ShiftScheduling(Family):
    name = "shift_scheduling"
    title = "Employee shift scheduling with days-off requests"
    sense = "min"
    objective_text = "the number of (employee, day) pairs in requests_off on which the employee works"
    sizes = [
        {"label": f"{e}emp x {d}d", "employees": e, "days": d}
        for e, d in ((4, 7), (6, 7), (9, 10), (14, 14), (20, 21))
    ]
    statement = (
        "Build a roster for employees 0..E-1 over days 0..D-1. There are three shifts per day: 0 = morning, "
        "1 = afternoon, 2 = night. Hard rules: (1) an employee works at most one shift per day; (2) on every "
        "day, shift s must be staffed by exactly demand[day][s] employees; (3) an employee who works the "
        "night shift on a day may not work the morning shift on the next day; (4) over the whole horizon each "
        "employee works between min_shifts and max_shifts shifts (inclusive). A feasible roster is guaranteed "
        "to exist. Soft rule: an entry [e, d] in requests_off means employee e would like to be off on day d."
    )
    input_schema = (
        '{"n_employees": int, "n_days": int, "demand": [[m, a, n], ...] (one triple per day), '
        '"requests_off": [[e, d], ...], "min_shifts": int, "max_shifts": int}'
    )
    output_schema = (
        '{"roster": [[x_{e,0}, ..., x_{e,D-1}], ...]}  one row per employee; x is -1 for a day off, '
        'otherwise the shift number 0, 1 or 2.'
    )
    example_input = {
        "n_employees": 2, "n_days": 2, "demand": [[1, 0, 0], [0, 1, 0]],
        "requests_off": [[0, 0]], "min_shifts": 1, "max_shifts": 1,
    }
    example_output = {"roster": [[-1, 1], [0, -1]]}

    def generate(self, size: dict, rng: random.Random) -> dict:
        E, D = size["employees"], size["days"]
        plan = [[-1] * D for _ in range(E)]            # planted roster, guarantees feasibility
        for e in range(E):
            for d in range(D):
                if rng.random() < 0.6:
                    s = rng.randrange(3)
                    if d and plan[e][d - 1] == 2 and s == 0:
                        s = 1
                    plan[e][d] = s
        demand = [[sum(1 for e in range(E) if plan[e][d] == s) for s in range(3)] for d in range(D)]
        counts = [sum(1 for x in row if x >= 0) for row in plan]
        requests = [[e, d] for e in range(E) for d in range(D) if rng.random() < 0.25]
        return {
            "n_employees": E, "n_days": D, "demand": demand, "requests_off": requests,
            "min_shifts": min(counts), "max_shifts": max(counts),
        }

    def reference(self, data: dict) -> tuple[dict, Optional[int]]:
        E, D = data["n_employees"], data["n_days"]
        m = cp_model.CpModel()
        x = {(e, d, s): m.new_bool_var(f"x{e}_{d}_{s}") for e in range(E) for d in range(D) for s in range(3)}
        for e in range(E):
            for d in range(D):
                m.add(sum(x[e, d, s] for s in range(3)) <= 1)
                if d + 1 < D:
                    m.add(x[e, d, 2] + x[e, d + 1, 0] <= 1)
            total = sum(x[e, d, s] for d in range(D) for s in range(3))
            m.add(total >= data["min_shifts"])
            m.add(total <= data["max_shifts"])
        for d in range(D):
            for s in range(3):
                m.add(sum(x[e, d, s] for e in range(E)) == data["demand"][d][s])
        m.minimize(sum(x[e, d, s] for e, d in data["requests_off"] for s in range(3)))
        sol = solve_reference(m, optimise=True)
        roster = [[next((s for s in range(3) if sol.value(x[e, d, s])), -1) for d in range(D)] for e in range(E)]
        return {"roster": roster}, int(sol.objective_value)

    def evaluate(self, data: dict, solution: dict) -> Optional[int]:
        E, D = data["n_employees"], data["n_days"]
        raw = as_list(get_field(solution, "roster"), "roster", E)
        r = [
            [as_int(v, f"roster[{e}][{d}]") for d, v in enumerate(as_list(row, f"roster[{e}]", D))]
            for e, row in enumerate(raw)
        ]
        for e in range(E):
            for d in range(D):
                if r[e][d] not in (-1, 0, 1, 2):
                    raise Violation(f"roster[{e}][{d}] is {r[e][d]}; allowed values are -1, 0, 1, 2.")
        for d in range(D):
            for s in range(3):
                got = sum(1 for e in range(E) if r[e][d] == s)
                if got != data["demand"][d][s]:
                    raise Violation(
                        f"On day {d} the {SHIFT_NAMES[s]} shift has {got} employees but the demand is "
                        f"{data['demand'][d][s]}."
                    )
        for e in range(E):
            for d in range(D - 1):
                if r[e][d] == 2 and r[e][d + 1] == 0:
                    raise Violation(f"Employee {e} works the night shift on day {d} and the morning shift on day {d + 1}.")
            total = sum(1 for x in r[e] if x >= 0)
            if not data["min_shifts"] <= total <= data["max_shifts"]:
                raise Violation(
                    f"Employee {e} works {total} shifts; it must be between {data['min_shifts']} and {data['max_shifts']}."
                )
        return sum(1 for e, d in data["requests_off"] if r[e][d] >= 0)
