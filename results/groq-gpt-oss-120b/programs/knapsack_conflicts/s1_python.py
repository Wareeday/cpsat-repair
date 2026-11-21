import time
import sys

def solve(instance: dict) -> dict:
    start_time = time.time()
    TIME_LIMIT = 28.0  # seconds, leave margin for overhead

    capacity = instance["capacity"]
    weights = instance["weights"]
    values = instance["values"]
    conflicts_input = instance.get("conflicts", [])
    n = len(weights)

    # conflict adjacency sets
    conflicts = [set() for _ in range(n)]
    for i, j in conflicts_input:
        conflicts[i].add(j)
        conflicts[j].add(i)

    # items sorted by value/weight density (zero weight -> infinite density)
    items = []
    for i in range(n):
        w = weights[i]
        v = values[i]
        density = float('inf') if w == 0 else v / w
        items.append((i, w, v, density))
    items.sort(key=lambda x: x[3], reverse=True)

    sorted_idx = [it[0] for it in items]
    sorted_w = [it[1] for it in items]
    sorted_v = [it[2] for it in items]

    best_value = 0
    best_selected = []

    sys.setrecursionlimit(10000)

    # optimistic bound using fractional knapsack (ignores conflicts)
    def optimistic_bound(pos: int, remaining_cap: int) -> float:
        total = 0.0
        cap = remaining_cap
        for k in range(pos, n):
            w = sorted_w[k]
            v = sorted_v[k]
            if w == 0:
                total += v
                continue
            if w <= cap:
                cap -= w
                total += v
            else:
                total += v * (cap / w)
                break
        return total

    def dfs(pos: int, cur_w: int, cur_v: int, selected: list, forbidden: set):
        nonlocal best_value, best_selected

        # time check
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError

        # update best solution
        if cur_v > best_value:
            best_value = cur_v
            best_selected = selected.copy()

        if pos == n:
            return

        # bound pruning
        if cur_v + optimistic_bound(pos, capacity - cur_w) <= best_value:
            return

        idx = sorted_idx[pos]
        w = sorted_w[pos]
        v = sorted_v[pos]

        # try to take the item
        if idx not in forbidden and cur_w + w <= capacity:
            new_forbidden = forbidden.union(conflicts[idx])
            dfs(pos + 1, cur_w + w, cur_v + v, selected + [idx], new_forbidden)

        # skip the item
        dfs(pos + 1, cur_w, cur_v, selected, forbidden)

    try:
        dfs(0, 0, 0, [], set())
    except TimeoutError:
        pass

    return {"selected": sorted(best_selected)}