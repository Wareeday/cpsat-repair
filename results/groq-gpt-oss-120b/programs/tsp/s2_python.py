import time
from typing import List, Dict

def solve(instance: Dict) -> Dict:
    n: int = instance["n"]
    dist: List[List[int]] = instance["dist"]
    if n == 0:
        return {"tour": []}
    if n == 1:
        return {"tour": [0]}

    deadline = time.time() + 28.0  # safety margin for the 30‑second limit

    # ---------- helper ----------
    def tour_length(tour: List[int]) -> int:
        total = 0
        for i in range(n):
            total += dist[tour[i]][tour[(i + 1) % n]]
        return total

    # ---------- exact DP (Held‑Karp) ----------
    best_tour: List[int] = list(range(n))  # fallback
    best_len: int = tour_length(best_tour)

    if n <= 13:  # exact DP feasible
        ALL = (1 << n) - 1
        dp = [{} for _ in range(1 << n)]
        parent = [{} for _ in range(1 << n)]
        start_mask = 1 << 0
        dp[start_mask][0] = 0

        for mask in range(1 << n):
            if not (mask & start_mask):
                continue
            # early exit if time is up
            if time.time() > deadline:
                break
            for i, cur_cost in dp[mask].items():
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    new_mask = mask | (1 << j)
                    new_cost = cur_cost + dist[i][j]
                    if j not in dp[new_mask] or new_cost < dp[new_mask][j]:
                        dp[new_mask][j] = new_cost
                        parent[new_mask][j] = i

        if time.time() <= deadline:
            # retrieve best closing edge
            min_cost = None
            last_city = -1
            for i, cost in dp[ALL].items():
                total = cost + dist[i][0]
                if min_cost is None or total < min_cost:
                    min_cost = total
                    last_city = i
            if min_cost is not None:
                # reconstruct path
                path = [0]
                mask = ALL
                cur = last_city
                while mask != start_mask:
                    path.append(cur)
                    prev = parent[mask][cur]
                    mask ^= (1 << cur)
                    cur = prev
                path.reverse()
                best_tour = path
                best_len = min_cost

    # ---------- heuristic for larger instances ----------
    # nearest‑neighbor + 2‑opt, run from several start cities
    def nearest_neighbor(start: int) -> List[int]:
        visited = [False] * n
        tour = [start]
        visited[start] = True
        cur = start
        for _ in range(n - 1):
            nxt = min((j for j in range(n) if not visited[j]),
                      key=lambda j: dist[cur][j])
            visited[nxt] = True
            tour.append(nxt)
            cur = nxt
        return tour

    def two_opt(tour: List[int]) -> List[int]:
        improved = True
        while improved and time.time() <= deadline:
            improved = False
            for i in range(n - 1):
                for k in range(i + 2, n):
                    if i == 0 and k == n - 1:
                        continue  # don't break the tour closure
                    a, b = tour[i], tour[(i + 1) % n]
                    c, d = tour[k], tour[(k + 1) % n]
                    delta = (dist[a][b] + dist[c][d]) - (dist[a][c] + dist[b][d])
                    if delta > 0:
                        # reverse segment (i+1 .. k)
                        tour[i + 1:k + 1] = reversed(tour[i + 1:k + 1])
                        improved = True
                        break
                if improved:
                    break
        return tour

    # try a few starts (all if small, otherwise a subset)
    starts = range(n) if n <= 8 else range(min(n, 8))
    for s in starts:
        if time.time() > deadline:
            break
        t = nearest_neighbor(s)
        t = two_opt(t)
        l = tour_length(t)
        if l < best_len:
            best_len = l
            best_tour = t

    return {"tour": best_tour}