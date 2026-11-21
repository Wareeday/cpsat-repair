import time
from typing import List, Dict

def solve(instance: Dict) -> Dict:
    """
    Solve the symmetric TSP (closed tour) exactly for small instances
    using the Held‑Karp dynamic programming algorithm.
    For larger instances a fast nearest‑neighbor + 2‑opt heuristic is used,
    but the DP will be attempted first and stopped if the time limit is
    approached.
    """
    start_time = time.time()
    TIME_LIMIT = 28.0          # a bit less than the external 30 s limit

    n: int = instance["n"]
    dist: List[List[int]] = instance["dist"]

    # Trivial case
    if n == 1:
        return {"tour": [0]}

    # ------------------------------------------------------------
    # Exact DP (Held‑Karp) – works well up to about n = 17‑18.
    # ------------------------------------------------------------
    max_exact_n = 17  # adjust if you want to allow larger exact solves
    if n <= max_exact_n:
        full_mask = (1 << n) - 1

        # dp[mask] = dict {last_city: min_cost}
        dp = [dict() for _ in range(1 << n)]
        # parent[mask][city] = previous city on the optimal path to (mask,city)
        parent = [dict() for _ in range(1 << n)]

        # Initialise paths that start at city 0 and go directly to city i
        for i in range(1, n):
            mask = (1 << 0) | (1 << i)
            dp[mask][i] = dist[0][i]
            parent[mask][i] = 0

        # Iterate over all masks that contain the start city (bit 0)
        for mask in range(1 << n):
            if not (mask & 1):          # start city must be in the set
                continue
            # early exit if time is almost up
            if time.time() - start_time > TIME_LIMIT:
                break

            cur_dp = dp[mask]
            if not cur_dp:
                continue
            for last, cur_cost in cur_dp.items():
                # try to append a new city j not yet visited
                not_visited = (~mask) & full_mask
                # iterate over bits set in not_visited
                j = 1
                while j < n:
                    bit = 1 << j
                    if not_visited & bit:
                        new_mask = mask | bit
                        new_cost = cur_cost + dist[last][j]
                        nd = dp[new_mask]
                        if j not in nd or new_cost < nd[j]:
                            nd[j] = new_cost
                            parent[new_mask][j] = last
                    j += 1

        # Retrieve best tour cost and ending city
        best_cost = None
        best_last = None
        final_dp = dp[full_mask]
        for last, cost in final_dp.items():
            total = cost + dist[last][0]   # return to start
            if best_cost is None or total < best_cost:
                best_cost = total
                best_last = last

        # If DP finished (best_last is not None) reconstruct the tour
        if best_last is not None:
            tour = [0] * n
            mask = full_mask
            cur = best_last
            idx = n - 1
            while mask != (1 << 0):
                tour[idx] = cur
                prev = parent[mask][cur]
                mask ^= (1 << cur)
                cur = prev
                idx -= 1
            tour[0] = 0
            return {"tour": tour}

    # ------------------------------------------------------------
    # Heuristic fallback (nearest neighbour + 2‑opt)
    # ------------------------------------------------------------
    def nearest_neighbour(start: int) -> List[int]:
        visited = [False] * n
        tour = [start]
        visited[start] = True
        cur = start
        for _ in range(n - 1):
            nxt = min(
                (j for j in range(n) if not visited[j]),
                key=lambda j: dist[cur][j],
            )
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt
        return tour

    def tour_length(t: List[int]) -> int:
        total = 0
        for i in range(n):
            total += dist[t[i]][t[(i + 1) % n]]
        return total

    # 2‑opt improvement
    def two_opt(t: List[int]) -> List[int]:
        improved = True
        while improved and (time.time() - start_time) < TIME_LIMIT:
            improved = False
            for i in range(1, n - 1):
                for k in range(i + 1, n):
                    # compute the gain of swapping edges (i‑1,i) and (k,k+1)
                    a, b = t[i - 1], t[i]
                    c, d = t[k], t[(k + 1) % n]
                    delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])
                    if delta < 0:
                        # perform the 2‑opt swap
                        t[i:k + 1] = reversed(t[i:k + 1])
                        improved = True
            # break early if time is low
        return t

    # start from city 0 (any start works because the tour is cyclic)
    best_tour = nearest_neighbour(0)
    best_len = tour_length(best_tour)

    # try a few random starts if time permits
    import random
    attempts = 0
    while (time.time() - start_time) < TIME_LIMIT and attempts < 8:
        start_city = random.randrange(n)
        tour = nearest_neighbour(start_city)
        tour = two_opt(tour)
        length = tour_length(tour)
        if length < best_len:
            best_len = length
            best_tour = tour
        attempts += 1

    return {"tour": best_tour}