from __future__ import annotations

import random
from typing import Optional

from ortools.sat.python import cp_model

from ._cp import solve_reference
from .base import Family, Violation, as_int, as_list, get_field


class LatinCompletion(Family):
    name = "latin_completion"
    title = "Latin square completion"
    sense = None
    sizes = [{"label": f"n={n}", "n": n} for n in (5, 8, 10, 14, 20)]
    statement = (
        "Fill an n x n grid with the integers 1..n so that every row and every column contains each of "
        "1..n exactly once. Some cells are pre-filled and must keep their value. Rows and columns are "
        "numbered 0..n-1. A completion is guaranteed to exist."
    )
    input_schema = '{"n": int, "prefilled": [[row, col, value], ...]}'
    output_schema = '{"grid": [[...], ...]}  an n x n list of lists, grid[r][c] in 1..n.'
    example_input = {"n": 3, "prefilled": [[0, 0, 1], [1, 1, 3]]}
    example_output = {"grid": [[1, 2, 3], [2, 3, 1], [3, 1, 2]]}

    def generate(self, size: dict, rng: random.Random) -> dict:
        n = size["n"]
        rows, cols, syms = list(range(n)), list(range(n)), list(range(1, n + 1))
        rng.shuffle(rows), rng.shuffle(cols), rng.shuffle(syms)
        grid = [[syms[(rows[r] + cols[c]) % n] for c in range(n)] for r in range(n)]
        cells = [(r, c) for r in range(n) for c in range(n)]
        keep = rng.sample(cells, round(0.42 * n * n))
        return {"n": n, "prefilled": [[r, c, grid[r][c]] for r, c in sorted(keep)]}

    def reference(self, data: dict) -> tuple[dict, Optional[int]]:
        n = data["n"]
        m = cp_model.CpModel()
        g = [[m.new_int_var(1, n, f"g{r}_{c}") for c in range(n)] for r in range(n)]
        for i in range(n):
            m.add_all_different(g[i])
            m.add_all_different([g[r][i] for r in range(n)])
        for r, c, v in data["prefilled"]:
            m.add(g[r][c] == v)
        s = solve_reference(m, optimise=False)
        return {"grid": [[s.value(x) for x in row] for row in g]}, None

    def evaluate(self, data: dict, solution: dict) -> Optional[int]:
        n = data["n"]
        raw = as_list(get_field(solution, "grid"), "grid", n)
        g = [[as_int(v, f"grid[{r}][{c}]") for c, v in enumerate(as_list(row, f"grid[{r}]", n))] for r, row in enumerate(raw)]
        for r in range(n):
            for c in range(n):
                if not 1 <= g[r][c] <= n:
                    raise Violation(f"Cell ({r}, {c}) holds {g[r][c]}, which is outside 1..{n}.")
        for r, c, v in data["prefilled"]:
            if g[r][c] != v:
                raise Violation(f"Cell ({r}, {c}) is pre-filled with {v} but the grid has {g[r][c]}.")
        for r in range(n):
            seen: dict[int, int] = {}
            for c in range(n):
                if g[r][c] in seen:
                    raise Violation(f"Row {r} contains the value {g[r][c]} twice (columns {seen[g[r][c]]} and {c}).")
                seen[g[r][c]] = c
        for c in range(n):
            seen = {}
            for r in range(n):
                if g[r][c] in seen:
                    raise Violation(f"Column {c} contains the value {g[r][c]} twice (rows {seen[g[r][c]]} and {r}).")
                seen[g[r][c]] = r
        return None
