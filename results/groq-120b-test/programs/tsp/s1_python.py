import time
from math import inf

def solve(instance: dict) -> dict:
    start_time = time.time()
    n = instance["n"]
    dist = instance["dist"]

    # Edge case: single city
    if n == 1:
        return {"tour": [0]}

    # Fix start city to 0 to reduce symmetry
    full_mask = (1 << n) - 1

    # DP tables
    dp = [[inf] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    dp[1][0] = 0  # mask with only city 0 visited

    # Iterate over all masks that include city 0
    for mask in range(1 << n):
        if not (mask & 1):  # must contain start city
            continue
        # optional time check to avoid overrun
        if time.time() - start_time > 28:
            break
        for i in range(n):
            cur_cost = dp[mask][i]
            if cur_cost == inf:
                continue
            # try to go to an unvisited city j
            remaining = (~mask) & full_mask
            while remaining:
                j = (remaining & -remaining).bit_length() - 1  # index of lowest set bit
                remaining ^= (1 << j)
                new_mask = mask | (1 << j)
                new_cost = cur_cost + dist[i][j]
                if new_cost < dp[new_mask][j]:
                    dp[new_mask][j] = new_cost
                    parent[new_mask][j] = i

    # Close the tour back to start (city 0)
    best_cost = inf
    last_city = -1
    for i in range(1, n):
        cost = dp[full_mask][i] + dist[i][0]
        if cost < best_cost:
            best_cost = cost
            last_city = i

    # If DP was interrupted (unlikely for intended sizes), fall back to a simple heuristic
    if last_city == -1:
        # Nearest neighbour heuristic starting from 0
        visited = [False] * n
        tour = [0]
        visited[0] = True
        cur = 0
        for _ in range(n - 1):
            nxt = min((j for j in range(n) if not visited[j]),
                      key=lambda j: dist[cur][j])
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt
        return {"tour": tour}

    # Reconstruct optimal tour
    tour = [0] * n
    mask = full_mask
    city = last_city
    for pos in range(n - 1, 0, -1):
        tour[pos] = city
        prev_city = parent[mask][city]
        mask ^= (1 << city)
        city = prev_city
    tour[0] = 0

    return {"tour": tour}