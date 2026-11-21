import time

def solve(instance: dict) -> dict:
    n = instance["n"]
    blocked = set((r, c) for r, c in instance.get("blocked", []))

    # allowed columns per row as bit masks
    allowed = [0] * n
    for r in range(n):
        mask = 0
        for c in range(n):
            if (r, c) not in blocked:
                mask |= 1 << c
        allowed[r] = mask

    # order rows by fewest available columns (MRV heuristic)
    rows_order = list(range(n))
    rows_order.sort(key=lambda r: allowed[r].bit_count())

    solution = [None] * n          # current partial solution (row -> column)
    best_solution = None           # will hold the first complete solution found

    start_time = time.time()
    TIME_LIMIT = 28.0               # seconds, a bit less than the overall limit

    class Found(Exception):
        """Used for fast unwind when a solution is found."""
        pass

    class Timeout(Exception):
        """Raised when the allowed time is exceeded."""
        pass

    def dfs(pos: int, cols: int, d1: int, d2: int):
        """Depth‑first search with bit masks."""
        nonlocal best_solution
        if time.time() - start_time > TIME_LIMIT:
            raise Timeout()
        if pos == n:
            best_solution = solution.copy()
            raise Found()

        r = rows_order[pos]
        # columns that are free in this row (respect blocked squares)
        avail = allowed[r] & ~cols

        while avail:
            # isolate lowest set bit and obtain its index
            low_bit = avail & -avail
            c = low_bit.bit_length() - 1

            d1_idx = r + c
            d2_idx = r - c + n - 1

            if ((d1 >> d1_idx) & 1) == 0 and ((d2 >> d2_idx) & 1) == 0:
                solution[r] = c
                dfs(
                    pos + 1,
                    cols | (1 << c),
                    d1 | (1 << d1_idx),
                    d2 | (1 << d2_idx),
                )
                solution[r] = None

            # clear the processed bit
            avail &= avail - 1

    # Primary search (ordered rows, time‑limited)
    try:
        dfs(0, 0, 0, 0)
    except Found:
        pass
    except Timeout:
        pass

    # If time ran out before a solution, fall back to a simple row‑by‑row backtrack
    if best_solution is None:
        def simple_backtrack(row: int, cols: int, d1: int, d2: int):
            if row == n:
                return []
            for c in range(n):
                if (allowed[row] >> c) & 1 == 0:
                    continue
                if (cols >> c) & 1:
                    continue
                if (d1 >> (row + c)) & 1:
                    continue
                if (d2 >> (row - c + n - 1)) & 1:
                    continue
                nxt = simple_backtrack(row + 1,
                                       cols | (1 << c),
                                       d1 | (1 << (row + c)),
                                       d2 | (1 << (row - c + n - 1)))
                if nxt is not None:
                    return [c] + nxt
            return None

        best_solution = simple_backtrack(0, 0, 0, 0)

    return {"queens": best_solution}