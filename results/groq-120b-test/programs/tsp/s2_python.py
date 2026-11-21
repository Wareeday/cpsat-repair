import time
from itertools import combinations

def solve(instance: dict) -> dict:
    start_time = time.time()
    TIME_LIMIT = 28.0  # safety margin

    n = instance["n"]
    dist = instance["dist"]

    if n == 1:
        return {"tour": [0]}

    # ---------- Helper functions ----------
    def tour_length(tour):
        total = 0
        for i in range(n):
            total += dist[tour[i]][tour[(i + 1) % n]]
        return total

    # ---------- Exact DP (Held‑Karp) ----------
    def held_karp():
        # fix start at 0
        size = 1 << n
        dp = [{} for _ in range(size)]
        # dp[mask][i] = (cost, prev)
        dp[1][0] = (0, -1)  # only city 0 visited, end at 0

        for mask in range(1, size):
            if not (mask & 1):  # must contain start city 0
                continue
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                if last == 0 and mask != 1:
                    continue
                cur_cost, _ = dp[mask].get(last, (float('inf'), -1))
                if cur_cost == float('inf'):
                    continue
                # try to extend to a new city nxt
                nxt_mask = mask
                for nxt in range(1, n):
                    if mask & (1 << nxt):
                        continue
                    nxt_mask = mask | (1 << nxt)
                    new_cost = cur_cost + dist[last][nxt]
                    old = dp[nxt_mask].get(nxt, (float('inf'), -1))[0]
                    if new_cost < old:
                        dp[nxt_mask][nxt] = (new_cost, last)

        full = (1 << n) - 1
        best_cost = float('inf')
        best_last = -1
        for last in range(1, n):
            cost, _ = dp[full].get(last, (float('inf'), -1))
            cost += dist[last][0]  # return to start
            if cost < best_cost:
                best_cost = cost
                best_last = last

        # reconstruct path
        tour = [0] * n
        mask = full
        last = best_last
        for idx in range(n - 1, 0, -1):
            tour[idx] = last
            _, prev = dp[mask][last]
            mask ^= (1 << last)
            last = prev
        tour[0] = 0
        return tour, best_cost

    # ---------- Heuristic: Nearest Neighbor + 2‑opt ----------
    def nearest_neighbor(start):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            nxt = min(unvisited, key=lambda x: dist[cur][x])
            unvisited.remove(nxt)
            tour.append(nxt)
            cur = nxt
        return tour

    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i, best_k = -1, -1
            for i in range(n - 1):
                a, b = tour[i], tour[(i + 1) % n]
                for k in range(i + 2, n if i else n - 1):
                    c, d = tour[k], tour[(k + 1) % n]
                    delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_k = i + 1, k
                        improved = True
            if improved:
                # reverse segment [best_i, best_k]
                tour[best_i:best_k + 1] = reversed(tour[best_i:best_k + 1])
        return tour

    # ---------- Main solving flow ----------
    best_tour = None
    best_len = float('inf')

    # Try exact DP for small n
    if n <= 13:
        try:
            tour, length = held_karp()
            best_tour, best_len = tour, length
        except Exception:
            pass
        if time.time() - start_time > TIME_LIMIT:
            return {"tour": best_tour}

    # Heuristic search (run while time permits)
    starts = list(range(n))
    # shuffle start order a bit to diversify (deterministic)
    for s in starts:
        if time.time() - start_time > TIME_LIMIT:
            break
        tour = nearest_neighbor(s)
        tour = two_opt(tour)
        length = tour_length(tour)
        if length < best_len:
            best_len = length
            best_tour = tour

    # If still no tour (should not happen), fallback to simple order
    if best_tour is None:
        best_tour = list(range(n))

    return {"tour": best_tour}