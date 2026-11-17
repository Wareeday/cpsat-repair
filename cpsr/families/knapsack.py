from __future__ import annotations

import random
from typing import Optional

from ortools.sat.python import cp_model

from ._cp import solve_reference
from .base import Family, Violation, as_int, as_list, get_field


class KnapsackConflicts(Family):
    name = "knapsack_conflicts"
    title = "0/1 knapsack with conflicting item pairs"
    sense = "max"
    objective_text = "the total value of the selected items"
    sizes = [{"label": f"n={n}", "n": n} for n in (10, 20, 40, 100, 200)]
    statement = (
        "Choose a subset of items. The total weight of the chosen items must not exceed the capacity. "
        "Some pairs of items are in conflict: two items of a conflicting pair may not both be chosen. "
        "Items are numbered 0..n-1."
    )
    input_schema = (
        '{"capacity": int, "weights": [int, ...], "values": [int, ...], "conflicts": [[i, j], ...]}'
    )
    output_schema = '{"selected": [i, ...]}  the indices of the chosen items, without repetition.'
    example_input = {"capacity": 10, "weights": [4, 5, 6], "values": [7, 8, 9], "conflicts": [[0, 1]]}
    example_output = {"selected": [0, 2]}

    def generate(self, size: dict, rng: random.Random) -> dict:
        n = size["n"]
        w = [rng.randint(10, 60) for _ in range(n)]
        v = [max(1, wi + rng.randint(-5, 15)) for wi in w]
        pairs = sorted({tuple(sorted(rng.sample(range(n), 2))) for _ in range(n // 2)})
        return {"capacity": int(0.35 * sum(w)), "weights": w, "values": v, "conflicts": [list(p) for p in pairs]}

    def reference(self, data: dict) -> tuple[dict, Optional[int]]:
        n = len(data["weights"])
        m = cp_model.CpModel()
        x = [m.new_bool_var(f"x{i}") for i in range(n)]
        m.add(sum(w * xi for w, xi in zip(data["weights"], x)) <= data["capacity"])
        for i, j in data["conflicts"]:
            m.add(x[i] + x[j] <= 1)
        m.maximize(sum(v * xi for v, xi in zip(data["values"], x)))
        s = solve_reference(m, optimise=True)
        sel = [i for i in range(n) if s.value(x[i])]
        return {"selected": sel}, int(s.objective_value)

    def evaluate(self, data: dict, solution: dict) -> Optional[int]:
        n = len(data["weights"])
        sel = [as_int(i, "selected item") for i in as_list(get_field(solution, "selected"), "selected")]
        if len(set(sel)) != len(sel):
            raise Violation("An item index appears more than once in 'selected'.")
        for i in sel:
            if not 0 <= i < n:
                raise Violation(f"Item index {i} is outside 0..{n - 1}.")
        weight = sum(data["weights"][i] for i in sel)
        if weight > data["capacity"]:
            raise Violation(f"The selected items weigh {weight}, which exceeds the capacity {data['capacity']}.")
        chosen = set(sel)
        for i, j in data["conflicts"]:
            if i in chosen and j in chosen:
                raise Violation(f"Items {i} and {j} are in conflict but both are selected.")
        return sum(data["values"][i] for i in sel)
