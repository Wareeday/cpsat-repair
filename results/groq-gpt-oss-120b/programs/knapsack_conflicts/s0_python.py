import time
from typing import List, Dict, Set

def solve(instance: Dict) -> Dict:
    capacity: int = instance["capacity"]
    weights: List[int] = instance["weights"]
    values: List[int] = instance["values"]
    conflict_pairs: List[List[int]] = instance["conflicts"]

    n = len(weights)

    # adjacency list for conflicts
    conflicts: List[Set[int]] = [set() for _ in range(n)]
    for i, j in conflict_pairs:
        conflicts[i].add(j)
        conflicts[j].add(i)

    # order items by value/weight ratio (descending) for better bounding
    order = list(range(n))
    order.sort(key=lambda i: values[i] / weights[i] if weights[i] != 0 else float('inf'), reverse=True)

    # state
    selected_bool = [False] * n          # currently selected items
    current_selected: List[int] = []     # list of selected indices (in original numbering)

    best_value = 0
    best_selected: List[int] = []

    start_time = time.time()
    TIME_LIMIT = 29.0   # a bit less than the external 30 s guard

    # ---------- helper functions ----------
    def conflicts_with_selected(idx: int) -> bool:
        """Return True if idx conflicts with any already selected item."""
        for nb in conflicts[idx]:
            if selected_bool[nb]:
                return True
        return False

    def fractional_bound(pos: int, remaining_capacity: int) -> float:
        """Optimistic bound using fractional knapsack on items from pos onward,
        ignoring conflicts among those items but respecting conflicts with already selected."""
        cap = remaining_capacity
        bound = 0.0
        for p in range(pos, n):
            idx = order[p]
            if selected_bool[idx] or conflicts_with_selected(idx):
                continue
            w = weights[idx]
            v = values[idx]
            if w <= cap:
                cap -= w
                bound += v
            else:
                if w > 0:
                    bound += v * (cap / w)
                break
        return bound

    # ---------- depth‑first search ----------
    def dfs(pos: int, cur_weight: int, cur_value: int):
        nonlocal best_value, best_selected

        # time check
        if time.time() - start_time > TIME_LIMIT:
            return

        # bound check
        ub = cur_value + fractional_bound(pos, capacity - cur_weight)
        if ub <= best_value:
            return

        if pos == n:
            if cur_value > best_value:
                best_value = cur_value
                best_selected = current_selected.copy()
            return

        idx = order[pos]

        # Branch 1: skip the item
        dfs(pos + 1, cur_weight, cur_value)

        # Branch 2: take the item (if feasible)
        if (not selected_bool[idx] and
            not conflicts_with_selected(idx) and
            cur_weight + weights[idx] <= capacity):
            # select
            selected_bool[idx] = True
            current_selected.append(idx)

            dfs(pos + 1, cur_weight + weights[idx], cur_value + values[idx])

            # backtrack
            current_selected.pop()
            selected_bool[idx] = False

    # start the search
    dfs(0, 0, 0)

    # ensure deterministic order in output
    best_selected.sort()
    return {"selected": best_selected}