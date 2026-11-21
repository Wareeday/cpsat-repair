import time
from typing import List, Dict, Set, Tuple

def solve(instance: Dict) -> Dict:
    # ---- unpack instance -------------------------------------------------
    capacity: int = instance["capacity"]
    weights: List[int] = instance["weights"]
    values: List[int] = instance["values"]
    conflict_pairs: List[List[int]] = instance.get("conflicts", [])

    n = len(weights)
    # ---- build conflict adjacency -----------------------------------------
    conflict_adj: List[Set[int]] = [set() for _ in range(n)]
    for i, j in conflict_pairs:
        conflict_adj[i].add(j)
        conflict_adj[j].add(i)

    # ---- item structure ---------------------------------------------------
    # each item: (original_index, weight, value, ratio, conflict_set)
    items: List[Tuple[int, int, int, float, Set[int]]] = []
    for idx in range(n):
        w, v = weights[idx], values[idx]
        ratio = v / w if w != 0 else float('inf')
        items.append((idx, w, v, ratio, conflict_adj[idx]))

    # sort by decreasing value/weight ratio (helps bound)
    items.sort(key=lambda x: x[3], reverse=True)

    # ---- pre‑compute for fractional bound ---------------------------------
    # arrays in sorted order
    sorted_weights = [it[1] for it in items]
    sorted_values = [it[2] for it in items]
    sorted_indices = [it[0] for it in items]

    # ---- global best ------------------------------------------------------
    best_value = 0
    best_selected: List[int] = []

    # ---- time limit -------------------------------------------------------
    start_time = time.time()
    TIME_LIMIT = 28.0  # a bit below 30 s to be safe

    # ---- helper: fractional knapsack bound ignoring conflicts ------------
    def fractional_bound(pos: int, remaining_cap: int, forbidden: Set[int]) -> float:
        """Upper bound on additional value from items[pos:] respecting capacity
        and skipping items whose original index is in forbidden."""
        bound = 0.0
        cap = remaining_cap
        for i in range(pos, n):
            idx = sorted_indices[i]
            if idx in forbidden:
                continue
            w = sorted_weights[i]
            v = sorted_values[i]
            if w <= cap:
                cap -= w
                bound += v
            else:
                # take fraction
                bound += v * (cap / w)
                break
        return bound

    # ---- depth‑first branch and bound --------------------------------------
    def dfs(pos: int,
            cur_weight: int,
            cur_value: int,
            selected: List[int],
            forbidden: Set[int]) -> None:
        nonlocal best_value, best_selected

        # time check
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError

        # reached end
        if pos == n:
            if cur_value > best_value:
                best_value = cur_value
                best_selected = selected.copy()
            return

        # upper bound pruning
        ub = cur_value + fractional_bound(pos, capacity - cur_weight, forbidden)
        if ub <= best_value:
            return

        idx, w, v, _, _ = items[pos]

        # Option 1: try to take the item (if feasible)
        if w + cur_weight <= capacity and idx not in forbidden:
            # update structures
            new_selected = selected + [idx]
            new_forbidden = forbidden.union(conflict_adj[idx])
            new_forbidden.add(idx)  # also forbid itself for later positions
            dfs(pos + 1, cur_weight + w, cur_value + v, new_selected, new_forbidden)

        # Option 2: skip the item
        # we keep current forbidden set (item may still be forbidden by earlier picks)
        dfs(pos + 1, cur_weight, cur_value, selected, forbidden)

    # ---- run search --------------------------------------------------------
    try:
        dfs(0, 0, 0, [], set())
    except TimeoutError:
        pass  # return best found so far

    # ---- output -----------------------------------------------------------
    return {"selected": sorted(best_selected)}