import time
import math
from typing import List, Dict

def solve(instance: Dict) -> Dict:
    capacity: int = instance["capacity"]
    sizes: List[int] = instance["sizes"]
    n = len(sizes)

    # trivial case
    if n == 0:
        return {"assignment": []}
    if capacity == 0:
        # all sizes must be 0, put everything in one bin
        return {"assignment": [0] * n}

    # order items by decreasing size (helps pruning)
    order = sorted(range(n), key=lambda i: sizes[i], reverse=True)
    sorted_sizes = [sizes[i] for i in order]

    # suffix sums of remaining sizes (for lower bound)
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_sizes[i]

    # basic lower bound from total volume
    total_volume = suffix_sum[0]
    lower_bound = (total_volume + capacity - 1) // capacity

    # ----- initial solution: First Fit Decreasing (gives an upper bound) -----
    bin_loads_fdd: List[int] = []
    assignment_fdd = [-1] * n
    for idx in range(n):
        item = order[idx]
        size = sizes[item]
        placed = False
        for b, load in enumerate(bin_loads_fdd):
            if load + size <= capacity:
                bin_loads_fdd[b] += size
                assignment_fdd[item] = b
                placed = True
                break
        if not placed:
            bin_loads_fdd.append(size)
            assignment_fdd[item] = len(bin_loads_fdd) - 1

    best_bins = len(bin_loads_fdd)
    best_assignment = assignment_fdd[:]  # copy

    # ----- branch‑and‑bound search -----
    start_time = time.time()
    time_limit = 28.0  # seconds

    # mutable state during recursion
    bin_loads: List[int] = []          # loads of currently opened bins
    assignment = [-1] * n              # final assignment being built
    bins_used = 0                      # number of bins currently opened

    # flag to abort recursion when time is out
    timeout = False

    def dfs(idx: int):
        nonlocal best_bins, best_assignment, bins_used, timeout
        # time check
        if time.time() - start_time > time_limit:
            timeout = True
            return

        if idx == n:                     # all items placed
            if bins_used < best_bins:
                best_bins = bins_used
                best_assignment = assignment[:]   # copy
            return

        # prune by optimistic lower bound
        remaining = suffix_sum[idx]
        optimistic = max(lower_bound, bins_used + (remaining + capacity - 1) // capacity)
        if optimistic >= best_bins:
            return

        size = sorted_sizes[idx]
        orig_item = order[idx]

        # try to put into existing bins (avoid symmetric tries)
        tried_loads = set()
        for b in range(bins_used):
            load = bin_loads[b]
            if load + size <= capacity and load not in tried_loads:
                tried_loads.add(load)
                bin_loads[b] += size
                assignment[orig_item] = b
                dfs(idx + 1)
                if timeout:
                    return
                bin_loads[b] = load
                assignment[orig_item] = -1

        # try a new bin
        if bins_used + 1 < best_bins:
            bin_loads.append(size)
            assignment[orig_item] = bins_used
            bins_used += 1
            dfs(idx + 1)
            if timeout:
                return
            bins_used -= 1
            bin_loads.pop()
            assignment[orig_item] = -1

    dfs(0)

    # In the unlikely event that timeout occurred before any solution,
    # fall back to the FFD solution already stored in best_assignment.
    return {"assignment": best_assignment}