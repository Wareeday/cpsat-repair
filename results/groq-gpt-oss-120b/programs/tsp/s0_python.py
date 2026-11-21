import time
from typing import List, Dict

def solve(instance: Dict) -> Dict:
    n: int = instance["n"]
    dist: List[List[int]] = instance["dist"]
    if n == 1:
        return {"tour": [0]}

    start_time = time.time()
    TIME_LIMIT = 28.0  # seconds, leave margin for reconstruction

    # ---------- Held‑Karp DP (exact) ----------
    # dp[mask][i] = min cost to start at 0, visit set mask (including 0 and i), end at i
    # predecessor for reconstruction
    INF = 10 ** 18
    size = 1 << n
    dp = [dict() for _ in range(size)]
    parent = [dict() for _ in range(size)]

    dp[1 << 0][0] = 0  # only start city visited

    for mask in range(size):
        if not (mask & 1):  # must contain start city
            continue
        # early termination if time is over
        if time.time() - start_time > TIME_LIMIT:
            break
        cur_dp = dp[mask]
        for last, cur_cost in cur_dp.items():
            # try to go to a new city nxt not yet visited
            remaining = (~mask) & ((1 << n) - 1)
            nxt = remaining
            while nxt:
                lsb = nxt & -nxt
                city = (lsb.bit_length() - 1)
                new_mask = mask | lsb
                new_cost = cur_cost + dist[last][city]
                nd = dp[new_mask]
                if new_cost < nd.get(city, INF):
                    nd[city] = new_cost
                    parent[new_mask][city] = last
                nxt -= lsb

    full_mask = (1 << n) - 1
    # If DP finished early due to timeout, fall back to greedy
    if time.time() - start_time > TIME_LIMIT or not dp[full_mask]:
        # ---------- Simple nearest‑neighbor heuristic ----------
        unvisited = set(range(n))
        tour = [0]
        unvisited.remove(0)
        while unvisited:
            last = tour[-1]
            nxt = min(unvisited, key=lambda v: dist[last][v])
            tour.append(nxt)
            unvisited.remove(nxt)
        return {"tour": tour}

    # Find best end city
    best_end = -1
    best_cost = INF
    for last, cost in dp[full_mask].items():
        total = cost + dist[last][0]  # return to start
        if total < best_cost:
            best_cost = total
            best_end = last

    # Reconstruct path backwards
    path = [0] * n
    mask = full_mask
    cur = best_end
    for idx in range(n - 1, -1, -1):
        path[idx] = cur
        prev = parent[mask].get(cur)
        if prev is None:
            break
        mask ^= (1 << cur)
        cur = prev

    return {"tour": path}