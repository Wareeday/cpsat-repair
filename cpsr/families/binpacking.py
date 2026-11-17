from __future__ import annotations

import random
from typing import Optional

from ortools.sat.python import cp_model

from ._cp import solve_reference
from .base import Family, Violation, as_int, as_list, get_field


class BinPacking(Family):
    name = "bin_packing"
    title = "Bin packing (minimum number of bins)"
    sense = "min"
    objective_text = "the number of bins that contain at least one item"
    sizes = [{"label": f"n={n}", "n": n} for n in (8, 12, 20, 30, 45)]
    statement = (
        "Assign every item to a bin so that the total size of the items in each bin does not exceed the "
        "bin capacity. Use as few bins as possible. Items are numbered 0..n-1; bins are labelled with "
        "arbitrary non-negative integers."
    )
    input_schema = '{"capacity": int, "sizes": [int, ...]}  (every size is at most the capacity)'
    output_schema = '{"assignment": [b_0, ..., b_{n-1}]}  b_i is the (non-negative integer) label of the bin of item i.'
    example_input = {"capacity": 10, "sizes": [6, 5, 4, 3]}
    example_output = {"assignment": [0, 1, 0, 1]}

    def generate(self, size: dict, rng: random.Random) -> dict:
        return {"capacity": 100, "sizes": [rng.randint(15, 60) for _ in range(size["n"])]}

    def reference(self, data: dict) -> tuple[dict, Optional[int]]:
        sizes, cap = data["sizes"], data["capacity"]
        n = len(sizes)
        ub = n
        m = cp_model.CpModel()
        x = [[m.new_bool_var(f"x{i}_{b}") for b in range(ub)] for i in range(n)]
        used = [m.new_bool_var(f"u{b}") for b in range(ub)]
        for i in range(n):
            m.add_exactly_one(x[i])
        for b in range(ub):
            m.add(sum(sizes[i] * x[i][b] for i in range(n)) <= cap * used[b])
            if b + 1 < ub:
                m.add(used[b] >= used[b + 1])   # symmetry breaking
        m.minimize(sum(used))
        s = solve_reference(m, optimise=True)
        assign = [next(b for b in range(ub) if s.value(x[i][b])) for i in range(n)]
        return {"assignment": assign}, len(set(assign))

    def evaluate(self, data: dict, solution: dict) -> Optional[int]:
        sizes, cap = data["sizes"], data["capacity"]
        raw = as_list(get_field(solution, "assignment"), "assignment", len(sizes))
        a = [as_int(b, f"assignment[{i}]") for i, b in enumerate(raw)]
        if any(b < 0 for b in a):
            raise Violation("Bin labels must be non-negative integers.")
        load: dict[int, int] = {}
        for i, b in enumerate(a):
            load[b] = load.get(b, 0) + sizes[i]
        for b, total in sorted(load.items()):
            if total > cap:
                items = [i for i, bb in enumerate(a) if bb == b]
                raise Violation(f"Bin {b} holds items {items} with total size {total}, which exceeds the capacity {cap}.")
        return len(load)
