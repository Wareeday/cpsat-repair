from __future__ import annotations

import random
from typing import Optional

from ortools.sat.python import cp_model

from ._cp import solve_reference
from .base import Family, Violation, as_int, as_list, get_field


def _greedy_colors(n: int, adj: list[set[int]]) -> int:
    colors = [-1] * n
    for v in sorted(range(n), key=lambda v: -len(adj[v])):
        used = {colors[u] for u in adj[v] if colors[u] >= 0}
        colors[v] = next(c for c in range(n) if c not in used)
    return max(colors) + 1


def _greedy_clique(n: int, adj: list[set[int]]) -> list[int]:
    best: list[int] = []
    for start in range(n):
        clique, cand = [start], set(adj[start])
        while cand:
            v = max(cand, key=lambda u: len(adj[u] & cand))
            clique.append(v)
            cand &= adj[v]
        if len(clique) > len(best):
            best = clique
    return best


class GraphColoring(Family):
    name = "graph_coloring"
    title = "Graph colouring (minimum number of colours)"
    sense = "min"
    objective_text = "the number of distinct colours used"
    sizes = [{"label": f"n={n}", "n": n, "p": p} for n, p in ((10, 0.35), (20, 0.30), (30, 0.25), (45, 0.25), (60, 0.25))]
    statement = (
        "Given an undirected graph on vertices 0..n-1, assign a colour (a non-negative integer) to every "
        "vertex so that the two endpoints of every edge get different colours. Use as few distinct colours "
        "as possible."
    )
    input_schema = '{"n": int, "edges": [[u, v], ...]}  (undirected, no self-loops, no duplicates)'
    output_schema = '{"colors": [c_0, ..., c_{n-1}]}  non-negative integers; the objective is the number of distinct values.'
    example_input = {"n": 4, "edges": [[0, 1], [1, 2], [2, 3], [3, 0]]}
    example_output = {"colors": [0, 1, 0, 1]}

    def generate(self, size: dict, rng: random.Random) -> dict:
        n, p = size["n"], size["p"]
        edges = [[u, v] for u in range(n) for v in range(u + 1, n) if rng.random() < p]
        return {"n": n, "edges": edges}

    def reference(self, data: dict) -> tuple[dict, Optional[int]]:
        n = data["n"]
        adj: list[set[int]] = [set() for _ in range(n)]
        for u, v in data["edges"]:
            adj[u].add(v), adj[v].add(u)
        k = _greedy_colors(n, adj)
        m = cp_model.CpModel()
        x = [m.new_int_var(0, k - 1, f"x{v}") for v in range(n)]
        for u, v in data["edges"]:
            m.add(x[u] != x[v])
        for i, v in enumerate(_greedy_clique(n, adj)):   # symmetry breaking
            m.add(x[v] == i)
        top = m.new_int_var(0, k - 1, "top")
        m.add_max_equality(top, x)
        m.minimize(top)
        s = solve_reference(m, optimise=True)
        sol = [s.value(v) for v in x]
        return {"colors": sol}, len(set(sol))

    def evaluate(self, data: dict, solution: dict) -> Optional[int]:
        n = data["n"]
        raw = as_list(get_field(solution, "colors"), "colors", n)
        col = [as_int(c, f"colors[{v}]") for v, c in enumerate(raw)]
        if any(c < 0 for c in col):
            raise Violation("Colours must be non-negative integers.")
        for u, v in data["edges"]:
            if col[u] == col[v]:
                raise Violation(f"Vertices {u} and {v} are adjacent but both have colour {col[u]}.")
        return len(set(col))
