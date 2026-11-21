import time
import random
from math import inf

def solve(instance: dict) -> dict:
    """Return an optimal (or best‑found) closed TSP tour for the given instance."""
    start_time = time.time()
    TIME_LIMIT = 28.0                     # a bit less than the external 30 s limit

    n = instance["n"]
    dist = instance["dist"]

    # ----------------------------------------------------------------------
    # Helper: total length of a tour (including return to start)
    # ----------------------------------------------------------------------
    def tour_length(tour):
        total = 0
        for i in range(n):
            total += dist[tour[i]][tour[(i + 1) % n]]
        return total

    # ----------------------------------------------------------------------
    # Exact Held‑Karp DP (works up to about n = 14‑15)
    # ----------------------------------------------------------------------
    if n <= 14:
        # dp[mask][i] = (cost, prev)   mask includes city i and city 0
        dp = [{} for _ in range(1 << n)]
        dp[1][0] = (0, -1)                     # mask 1 == only city 0 visited

        for mask in range(1, 1 << n):
            if not (mask & 1):                 # must contain start city 0
                continue
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                if i == 0 and mask != 1:
                    continue
                cur_cost, _ = dp[mask].get(i, (inf, -1))
                if cur_cost == inf:
                    continue
                # try to go to a new city j
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    nmask = mask | (1 << j)
                    ndist = cur_cost + dist[i][j]
                    old = dp[nmask].get(j, (inf, -1))[0]
                    if ndist < old:
                        dp[nmask][j] = (ndist, i)

        full_mask = (1 << n) - 1
        best_cost = inf
        best_last = -1
        for i in range(1, n):
            cost_i, _ = dp[full_mask].get(i, (inf, -1))
            total = cost_i + dist[i][0]
            if total < best_cost:
                best_cost = total
                best_last = i

        # reconstruct path
        tour = [0] * n
        mask = full_mask
        cur = best_last
        for pos in range(n - 1, 0, -1):
            tour[pos] = cur
            _, prev = dp[mask][cur]
            mask ^= (1 << cur)
            cur = prev
        tour[0] = 0
        return {"tour": tour}

    # ----------------------------------------------------------------------
    # Heuristic for larger instances: Nearest‑Neighbour + 2‑opt
    # ----------------------------------------------------------------------
    # nearest neighbour starting from a given city
    def nearest_neighbour(start):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            nxt = min(unvisited, key=lambda v: dist[cur][v])
            unvisited.remove(nxt)
            tour.append(nxt)
            cur = nxt
        return tour

    # 2‑opt improvement (returns True if an improvement was made)
    def two_opt(tour):
        improved = False
        for i in range(n - 1):
            a, b = tour[i], tour[(i + 1) % n]
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue          # would break the closed tour edge
                c, d = tour[j], tour[(j + 1) % n]
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < -1e-9:
                    # perform 2‑opt flip
                    if i < j:
                        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    else:
                        # shouldn't happen with the loops above
                        pass
                    improved = True
                    return True        # restart search after first improvement
        return improved

    # Main heuristic loop
    best_tour = None
    best_len = inf
    # try a few random starts (or all starts if n is modest)
    starts = list(range(n))
    if n > 30:
        random.shuffle(starts)
        starts = starts[:30]               # limit number of restarts for speed
    for s in starts:
        if time.time() - start_time > TIME_LIMIT:
            break
        tour = nearest_neighbour(s)
        # improve with 2‑opt until no improvement or time runs out
        while True:
            if time.time() - start_time > TIME_LIMIT:
                break
            if not two_opt(tour):
                break
        cur_len = tour_length(tour)
        if cur_len < best_len:
            best_len = cur_len
            best_tour = tour[:]

    # Fallback: if nothing was generated (should not happen) return identity
    if best_tour is None:
        best_tour = list(range(n))

    return {"tour": best_tour}