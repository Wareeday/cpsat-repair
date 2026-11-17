from __future__ import annotations

import random
from typing import Optional

from ortools.sat.python import cp_model

from ._cp import solve_reference
from .base import Family, Violation, as_int, as_list, get_field


def _planted_queens(n: int, rng: random.Random) -> list[int]:
    """Random n-queens solution via min-conflicts."""
    def attackers(cols: list[int], r: int, c: int) -> int:
        return sum(1 for r2 in range(n) if r2 != r and (cols[r2] == c or abs(cols[r2] - c) == abs(r2 - r)))

    while True:
        cols = [rng.randrange(n) for _ in range(n)]
        for _ in range(50 * n):
            bad = [r for r in range(n) if attackers(cols, r, cols[r])]
            if not bad:
                return cols
            r = rng.choice(bad)
            scores = [attackers(cols, r, c) for c in range(n)]
            best = min(scores)
            cols[r] = rng.choice([c for c in range(n) if scores[c] == best])


class NQueensBlocked(Family):
    name = "nqueens_blocked"
    title = "N-Queens with blocked squares"
    sense = None
    sizes = [{"label": f"n={n}", "n": n} for n in (8, 12, 16, 24, 32)]
    statement = (
        "Place n queens on an n x n chessboard, exactly one queen in every row. No two queens may share a "
        "column or a diagonal (in either direction). Some squares are blocked and may not hold a queen. "
        "Rows and columns are numbered 0..n-1. A valid placement is guaranteed to exist."
    )
    input_schema = '{"n": int, "blocked": [[row, col], ...]}'
    output_schema = '{"queens": [c_0, c_1, ..., c_{n-1}]}  where c_r is the column of the queen in row r.'
    example_input = {"n": 4, "blocked": [[0, 0], [3, 3]]}
    example_output = {"queens": [1, 3, 0, 2]}

    def generate(self, size: dict, rng: random.Random) -> dict:
        n = size["n"]
        planted = _planted_queens(n, rng)
        free = [(r, c) for r in range(n) for c in range(n) if planted[r] != c]
        blocked = sorted(rng.sample(free, 2 * n))
        return {"n": n, "blocked": [list(b) for b in blocked]}

    def reference(self, data: dict) -> tuple[dict, Optional[int]]:
        n = data["n"]
        blocked = {(r, c) for r, c in data["blocked"]}
        m = cp_model.CpModel()
        q = []
        for r in range(n):
            allowed = [c for c in range(n) if (r, c) not in blocked]
            q.append(m.new_int_var_from_domain(cp_model.Domain.from_values(allowed), f"q{r}"))
        m.add_all_different(q)
        m.add_all_different([q[r] + r for r in range(n)])
        m.add_all_different([q[r] - r for r in range(n)])
        s = solve_reference(m, optimise=False)
        return {"queens": [s.value(v) for v in q]}, None

    def evaluate(self, data: dict, solution: dict) -> Optional[int]:
        n = data["n"]
        blocked = {(r, c) for r, c in data["blocked"]}
        raw = as_list(get_field(solution, "queens"), "queens", n)
        q = [as_int(c, f"queens[{r}]") for r, c in enumerate(raw)]
        for r, c in enumerate(q):
            if not 0 <= c < n:
                raise Violation(f"The queen in row {r} is in column {c}, outside 0..{n - 1}.")
            if (r, c) in blocked:
                raise Violation(f"The queen in row {r} is on the blocked square ({r}, {c}).")
        for r1 in range(n):
            for r2 in range(r1 + 1, n):
                if q[r1] == q[r2]:
                    raise Violation(f"The queens in rows {r1} and {r2} are both in column {q[r1]}.")
                if abs(q[r1] - q[r2]) == r2 - r1:
                    raise Violation(
                        f"The queens in rows {r1} (column {q[r1]}) and {r2} (column {q[r2]}) attack each other diagonally."
                    )
        return None
