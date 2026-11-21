import time
from typing import List, Dict, Set

def solve(instance: Dict) -> Dict:
    # ---------- data preparation ----------
    n: int = instance["n"]
    edges: List[List[int]] = instance["edges"]
    adj: List[Set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    degrees = [len(nei) for nei in adj]

    # ---------- greedy upper bound (largest degree ordering) ----------
    order = sorted(range(n), key=lambda x: -degrees[x])
    greedy_colors = [-1] * n
    for v in order:
        forbidden = {greedy_colors[u] for u in adj[v] if greedy_colors[u] != -1}
        c = 0
        while c in forbidden:
            c += 1
        greedy_colors[v] = c
    best_num = max(greedy_colors) + 1 if n else 0
    best_colors = greedy_colors[:]

    # ---------- DSATUR exact search with branch‑and‑bound ----------
    start_time = time.time()
    TIME_LIMIT = 29.5  # seconds

    colors = [-1] * n                     # current partial coloring
    uncolored: Set[int] = set(range(n))   # vertices not yet colored
    used_colors: Set[int] = set()         # colors currently in use

    # Helper: select next vertex using DSATUR rule
    def select_vertex():
        best_v = -1
        best_sat = -1
        best_deg = -1
        best_nei_colors = set()
        for v in uncolored:
            nei_cols = {colors[u] for u in adj[v] if colors[u] != -1}
            sat = len(nei_cols)
            deg = degrees[v]
            if sat > best_sat or (sat == best_sat and deg > best_deg):
                best_sat = sat
                best_deg = deg
                best_v = v
                best_nei_colors = nei_cols
        return best_v, best_nei_colors

    # Recursive depth‑first search
    def dfs():
        nonlocal best_num, best_colors
        # time check
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError

        if not uncolored:                     # all vertices colored
            cur_num = len(used_colors)
            if cur_num < best_num:
                best_num = cur_num
                best_colors = colors[:]
            return

        v, forbidden = select_vertex()

        # Try existing colors first (smallest first)
        for c in sorted(used_colors):
            if c in forbidden:
                continue
            # pruning
            if len(used_colors) >= best_num:
                continue
            # assign
            colors[v] = c
            uncolored.remove(v)
            dfs()
            # backtrack
            uncolored.add(v)
            colors[v] = -1

        # Try a new color
        new_color = len(used_colors)          # smallest unused non‑negative integer
        if new_color < best_num:
            colors[v] = new_color
            used_colors.add(new_color)
            uncolored.remove(v)
            dfs()
            # backtrack
            uncolored.add(v)
            colors[v] = -1
            used_colors.remove(new_color)

    try:
        dfs()
    except TimeoutError:
        pass  # return the best solution found so far

    return {"colors": best_colors}