def solve(instance: dict) -> dict:
    import time, sys
    sys.setrecursionlimit(10000)

    cap = instance["capacity"]
    sizes = instance["sizes"]
    n = len(sizes)

    # order items decreasingly (helps pruning)
    order = sorted(range(n), key=lambda i: -sizes[i])
    sorted_sizes = [sizes[i] for i in order]

    best_bins = n + 1          # current best number of bins found
    best_assign = None         # assignment for the best solution

    start = time.time()
    TIME_LIMIT = 29.0          # seconds, a bit below the allowed 30 s

    class TimeoutError(RuntimeError):
        pass

    def dfs(idx: int, bins_rem: list[int], assign: list[int]) -> None:
        """Depth‑first search with branch‑and‑bound."""
        nonlocal best_bins, best_assign

        # time check
        if time.time() - start > TIME_LIMIT:
            raise TimeoutError

        # all items placed
        if idx == n:
            used = len(bins_rem)
            if used < best_bins:
                best_bins = used
                # translate assignment back to original item order
                result = [0] * n
                for pos, b in enumerate(assign):
                    result[order[pos]] = b
                best_assign = result
            return

        # lower bound: remaining total size needs at least this many bins
        remaining_total = sum(sorted_sizes[idx:])
        lower_needed = (remaining_total + cap - 1) // cap
        if len(bins_rem) + lower_needed >= best_bins:
            return

        size = sorted_sizes[idx]

        # try to put the item into an existing bin
        for b_idx, rem in enumerate(bins_rem):
            if size <= rem:
                bins_rem[b_idx] -= size
                assign.append(b_idx)
                dfs(idx + 1, bins_rem, assign)
                assign.pop()
                bins_rem[b_idx] = rem

        # try a new bin (only after all existing bins have been tried)
        assign.append(len(bins_rem))
        bins_rem.append(cap - size)
        dfs(idx + 1, bins_rem, assign)
        assign.pop()
        bins_rem.pop()

    # run the exact search, but fall back to a greedy solution if we run out of time
    try:
        dfs(0, [], [])
    except TimeoutError:
        pass

    if best_assign is None:
        # simple first‑fit greedy fallback (guarantees a feasible solution)
        assign = [-1] * n
        bins = []          # remaining capacity of each bin
        for i, sz in enumerate(sizes):
            placed = False
            for b_idx, rem in enumerate(bins):
                if sz <= rem:
                    bins[b_idx] -= sz
                    assign[i] = b_idx
                    placed = True
                    break
            if not placed:
                bins.append(cap - sz)
                assign[i] = len(bins) - 1
        best_assign = assign

    return {"assignment": best_assign}