import time
from typing import List, Dict, Set

TIME_LIMIT = 28.0  # seconds


def solve(instance: Dict) -> Dict:
    n: int = instance.get("n", 0)
    edges: List[List[int]] = instance.get("edges", [])

    if n == 0:
        return {"colors": []}

    # adjacency list
    adj: List[Set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # ---------- greedy coloring for initial upper bound ----------
    order = sorted(range(n), key=lambda x: -len(adj[x]))  # descending degree
    greedy_colors = [-1] * n
    max_color = -1
    for v in order:
        used = {greedy_colors[nb] for nb in adj[v] if greedy_colors[nb] != -1}
        c = 0
        while c in used:
            c += 1
        greedy_colors[v] = c
        max_color = max(max_color, c)
    best_num = max_color + 1
    best_solution = greedy_colors[:]

    # ---------- data structures for DSATUR ----------
    colors = [-1] * n                     # current partial coloring
    neighbor_color_sets: List[Set[int]] = [set() for _ in range(n)]
    uncolored: Set[int] = set(range(n))

    start_time = time.time()

    # Helper to select next vertex using DSATUR rule
    def select_vertex() -> int:
        # max saturation degree, break ties by degree
        best_v = -1
        best_sat = -1
        best_deg = -1
        for v in uncolored:
            sat = len(neighbor_color_sets[v])
            deg = len(adj[v])
            if sat > best_sat or (sat == best_sat and deg > best_deg):
                best_sat, best_deg, best_v = sat, deg, v
        return best_v

    # Recursive backtracking with branch‑and‑bound
    def dfs(colored_cnt: int, used_colors: int):
        nonlocal best_num, best_solution

        # time check
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError

        if colored_cnt == n:
            # found a complete coloring
            if used_colors < best_num:
                best_num = used_colors
                best_solution = colors[:]
            return

        # prune if already not better than best
        if used_colors >= best_num:
            return

        v = select_vertex()
        # possible existing colors
        forbidden = neighbor_color_sets[v]
        for c in range(used_colors):
            if c in forbidden:
                continue
            # assign
            colors[v] = c
            uncolored.remove(v)
            # update neighbor_color_sets
            changed = []
            for nb in adj[v]:
                if c not in neighbor_color_sets[nb]:
                    neighbor_color_sets[nb].add(c)
                    changed.append(nb)
            dfs(colored_cnt + 1, used_colors)
            # undo
            for nb in changed:
                neighbor_color_sets[nb].remove(c)
            uncolored.add(v)
            colors[v] = -1

        # try a new color
        if used_colors + 1 < best_num:
            c = used_colors
            colors[v] = c
            uncolored.remove(v)
            changed = []
            for nb in adj[v]:
                if c not in neighbor_color_sets[nb]:
                    neighbor_color_sets[nb].add(c)
                    changed.append(nb)
            dfs(colored_cnt + 1, used_colors + 1)
            for nb in changed:
                neighbor_color_sets[nb].remove(c)
            uncolored.add(v)
            colors[v] = -1

    # start the search
    try:
        dfs(0, 0)
    except TimeoutError:
        pass  # return the best solution found so far

    # ensure output uses non‑negative ints
    return {"colors": best_solution}