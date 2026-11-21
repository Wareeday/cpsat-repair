import time

def solve(instance: dict) -> dict:
    capacity = instance["capacity"]
    sizes = instance["sizes"]
    n = len(sizes)

    if n == 0:
        return {"assignment": []}

    # order items decreasing by size
    order = sorted(range(n), key=lambda i: -sizes[i])
    sorted_sizes = [sizes[i] for i in order]

    start = time.time()
    TIME_LIMIT = 28.0  # a little margin before the external 30 s limit

    # ---------- simple upper bound (first‑fit decreasing) ----------
    bins_rem = []
    assign_best = [-1] * n
    for idx, sz in zip(order, sorted_sizes):
        placed = False
        for b, rem in enumerate(bins_rem):
            if sz <= rem:
                bins_rem[b] -= sz
                assign_best[idx] = b
                placed = True
                break
        if not placed:
            bins_rem.append(capacity - sz)
            assign_best[idx] = len(bins_rem) - 1

    best_num_bins = len(bins_rem)

    # lower bound from total size
    lower = (sum(sizes) + capacity - 1) // capacity

    # ---------- decision routine: can we pack into k bins? ----------
    def can_pack(k: int):
        """Return (True, assignment) if items fit into k bins, else (False, None)."""
        bins = [capacity] * k
        assign = [-1] * n

        # depth‑first search with symmetry pruning
        def dfs(pos: int) -> bool:
            if time.time() - start > TIME_LIMIT:
                raise TimeoutError
            if pos == n:
                return True

            sz = sorted_sizes[pos]
            for i in range(k):
                if bins[i] < sz:
                    continue
                # skip bins with the same remaining capacity as a previous one
                if i > 0 and bins[i] == bins[i - 1]:
                    continue

                before = bins[i]
                bins[i] -= sz
                assign[order[pos]] = i

                if dfs(pos + 1):
                    return True

                # backtrack
                bins[i] = before
                assign[order[pos]] = -1

                # if the item exactly filled an empty bin, no need to try other empty bins
                if before == sz:          # item exactly filled the bin
                    break
                if before == capacity:    # we tried an empty bin
                    break
            return False

        try:
            feasible = dfs(0)
        except TimeoutError:
            feasible = False
        return feasible, assign if feasible else None

    # ---------- iterative search for the optimum ----------
    for k in range(lower, best_num_bins):
        if time.time() - start > TIME_LIMIT:
            break
        feasible, assign = can_pack(k)
        if feasible:
            best_num_bins = k
            assign_best = assign
            break

    return {"assignment": assign_best}