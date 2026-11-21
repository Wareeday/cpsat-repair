import time
from typing import List, Dict

def solve(instance: Dict) -> Dict:
    # ---------- helpers ----------
    def greedy_upper_bound(order: List[int]) -> List[int]:
        """Simple greedy coloring following given vertex order."""
        col = [-1] * n
        for v in order:
            used = 0
            for nb in adj[v]:
                c = col[nb]
                if c != -1:
                    used |= 1 << c
            # smallest free colour
            c = 0
            while used & (1 << c):
                c += 1
            col[v] = c
        return col

    def heuristic_clique_size() -> int:
        """Very cheap heuristic for a lower bound (size of a large clique)."""
        # vertices sorted by decreasing degree
        verts = sorted(range(n), key=lambda x: -deg[x])
        best = 0
        for v in verts:
            clique = [v]
            for u in verts:
                if u == v:
                    continue
                if all(u in adj[w] for w in clique):
                    clique.append(u)
            best = max(best, len(clique))
        return best

    # ---------- backtracking for k‑colourability ----------
    def try_k(k: int) -> List[int] | None:
        """Return a valid colouring using at most k colours, or None."""
        colors = [-1] * n
        sat_mask = [0] * n  # bitmask of neighbour colours
        start = time.time()

        def dfs(coloured_cnt: int) -> bool:
            # time check
            if time.time() - start > TIME_LIMIT * 0.95:
                raise TimeoutError
            if coloured_cnt == n:
                return True

            # choose vertex with highest saturation, break ties by degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in range(n):
                if colors[v] == -1:
                    sat = sat_mask[v].bit_count()
                    d = deg[v]
                    if sat > best_sat or (sat == best_sat and d > best_deg):
                        best_sat = sat
                        best_deg = d
                        best_v = v

            forbidden = sat_mask[best_v]
            for c in range(k):
                if not (forbidden >> c) & 1:
                    colors[best_v] = c
                    changed = []
                    bit = 1 << c
                    for nb in adj[best_v]:
                        if colors[nb] == -1 and not (sat_mask[nb] & bit):
                            sat_mask[nb] |= bit
                            changed.append(nb)
                    if dfs(coloured_cnt + 1):
                        return True
                    # revert
                    for nb in changed:
                        sat_mask[nb] &= ~bit
                    colors[best_v] = -1
            return False

        try:
            if dfs(0):
                return colors
        except TimeoutError:
            return None
        return None

    # ---------- main ----------
    TIME_LIMIT = 30.0  # seconds
    start_time = time.time()

    n = instance.get("n", 0)
    edges = instance.get("edges", [])
    if n == 0:
        return {"colors": []}

    # adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    deg = [len(adj[i]) for i in range(n)]

    # initial upper bound via greedy (order by decreasing degree)
    order = sorted(range(n), key=lambda x: -deg[x])
    greedy_colors = greedy_upper_bound(order)
    best_num = max(greedy_colors) + 1
    best_colors = greedy_colors[:]

    # lower bound heuristic (clique size)
    lower = heuristic_clique_size()
    lower = max(lower, 1)

    # iterative deepening on k
    for k in range(lower, best_num):
        if time.time() - start_time > TIME_LIMIT * 0.9:
            break
        sol = try_k(k)
        if sol is not None:
            best_num = k
            best_colors = sol
            break  # optimal found

    # ensure colours are non‑negative ints and compact (optional)
    # (they already are)
    return {"colors": best_colors}