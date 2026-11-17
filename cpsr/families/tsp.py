from __future__ import annotations

import math
import random
from typing import Optional

from ortools.sat.python import cp_model

from ._cp import solve_reference
from .base import Family, Violation, as_int, as_list, get_field


class TSP(Family):
    name = "tsp"
    title = "Travelling salesman problem (closed tour)"
    sense = "min"
    objective_text = "the total length of the closed tour, including the edge back to the start"
    sizes = [{"label": f"n={n}", "n": n} for n in (6, 8, 11, 15, 20)]
    statement = (
        "Visit every city exactly once and return to the starting city. dist[i][j] is the integer distance "
        "from city i to city j (the matrix is symmetric). Cities are numbered 0..n-1."
    )
    input_schema = '{"n": int, "dist": [[int, ...], ...]}  (n x n symmetric matrix, zero diagonal)'
    output_schema = '{"tour": [c_0, ..., c_{n-1}]}  a permutation of 0..n-1; the tour returns from c_{n-1} to c_0.'
    example_input = {"n": 3, "dist": [[0, 2, 9], [2, 0, 4], [9, 4, 0]]}
    example_output = {"tour": [0, 1, 2]}

    def generate(self, size: dict, rng: random.Random) -> dict:
        n = size["n"]
        pts = [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
        dist = [[round(math.dist(p, q)) for q in pts] for p in pts]
        return {"n": n, "dist": dist}

    def reference(self, data: dict) -> tuple[dict, Optional[int]]:
        n, d = data["n"], data["dist"]
        m = cp_model.CpModel()
        lit = {(i, j): m.new_bool_var(f"a{i}_{j}") for i in range(n) for j in range(n) if i != j}
        m.add_circuit([(i, j, v) for (i, j), v in lit.items()])
        m.minimize(sum(d[i][j] * v for (i, j), v in lit.items()))
        s = solve_reference(m, optimise=True)
        nxt = {i: j for (i, j), v in lit.items() if s.value(v)}
        tour, cur = [0], nxt[0]
        while cur != 0:
            tour.append(cur)
            cur = nxt[cur]
        return {"tour": tour}, int(s.objective_value)

    def evaluate(self, data: dict, solution: dict) -> Optional[int]:
        n, d = data["n"], data["dist"]
        tour = [as_int(c, "tour entry") for c in as_list(get_field(solution, "tour"), "tour")]
        for c in tour:
            if not 0 <= c < n:
                raise Violation(f"The tour contains city {c}, outside 0..{n - 1}.")
        if len(tour) != n or len(set(tour)) != n:
            missing = sorted(set(range(n)) - set(tour))
            dup = sorted({c for c in tour if tour.count(c) > 1})
            raise Violation(
                f"The tour must visit each of the {n} cities exactly once; it has {len(tour)} entries, "
                f"missing cities {missing}, repeated cities {dup}."
            )
        return sum(d[tour[i]][tour[(i + 1) % n]] for i in range(n))
