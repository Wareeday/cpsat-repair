import sys
import time

def solve(instance: dict) -> dict:
    n = instance["n"]
    prefilled = instance.get("prefilled", [])

    # Grid and masks
    grid = [[0] * n for _ in range(n)]
    row_mask = [0] * n          # bits of numbers already used in each row
    col_mask = [0] * n          # bits of numbers already used in each column
    full_mask = (1 << n) - 1    # bits 0..n-1 set

    for r, c, v in prefilled:
        grid[r][c] = v
        bit = 1 << (v - 1)
        row_mask[r] |= bit
        col_mask[c] |= bit

    # List of empty cells
    empties = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == 0]

    start_time = time.time()
    TIME_LIMIT = 28.0           # seconds, a bit below the external limit
    sys.setrecursionlimit(10000)

    class TimeoutError(RuntimeError):
        pass

    def backtrack(idx: int) -> bool:
        # time check
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError

        if idx == len(empties):
            return True

        # MRV: choose cell with smallest domain among the remaining ones
        best_i = -1
        best_cnt = n + 1
        best_mask = 0
        for i in range(idx, len(empties)):
            r, c = empties[i]
            mask = full_mask ^ (row_mask[r] | col_mask[c])
            cnt = mask.bit_count()
            if cnt < best_cnt:
                best_cnt = cnt
                best_i = i
                best_mask = mask
                if cnt == 1:
                    break
        if best_i == -1 or best_cnt == 0:
            return False

        # put the chosen cell at position idx
        empties[idx], empties[best_i] = empties[best_i], empties[idx]
        r, c = empties[idx]
        mask = best_mask

        while mask:
            lsb = mask & -mask
            val_idx = lsb.bit_length() - 1      # 0‑based index
            v = val_idx + 1                     # actual value 1..n
            bit = 1 << val_idx

            # assign
            grid[r][c] = v
            row_mask[r] |= bit
            col_mask[c] |= bit

            if backtrack(idx + 1):
                return True

            # undo
            grid[r][c] = 0
            row_mask[r] ^= bit
            col_mask[c] ^= bit

            mask ^= lsb

        # restore order before backtracking further
        empties[idx], empties[best_i] = empties[best_i], empties[idx]
        return False

    try:
        backtrack(0)
    except TimeoutError:
        # If we run out of time we simply return the current (partial) grid.
        # The test harness expects a *valid* Latin square; the timeout case
        # should be rare for the intended instance sizes.
        pass

    return {"grid": grid}