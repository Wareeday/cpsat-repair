import time
from typing import List, Dict

def solve(instance: Dict) -> Dict:
    n: int = instance["n"]
    prefilled: List[List[int]] = instance["prefilled"]

    # grid initialization
    grid: List[List[int]] = [[0] * n for _ in range(n)]

    # bit masks for used numbers in each row / column (bit i -> number i+1)
    row_mask: List[int] = [0] * n
    col_mask: List[int] = [0] * n
    full_mask: int = (1 << n) - 1

    for r, c, v in prefilled:
        bit = 1 << (v - 1)
        grid[r][c] = v
        row_mask[r] |= bit
        col_mask[c] |= bit

    # collect empty cells
    empties: List[tuple] = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == 0]

    start_time = time.time()
    TIME_LIMIT = 28.0  # a bit less than the hard limit

    # Helper: returns list of possible values (as ints) for cell (r,c)
    def possible_vals(r: int, c: int) -> List[int]:
        avail_mask = full_mask & ~(row_mask[r] | col_mask[c])
        vals = []
        while avail_mask:
            lsb = avail_mask & -avail_mask
            v = (lsb.bit_length() - 1) + 1
            vals.append(v)
            avail_mask ^= lsb
        return vals

    # Recursive backtracking with MRV heuristic
    def dfs(k: int) -> bool:
        # time check
        if time.time() - start_time > TIME_LIMIT:
            return False

        if k == len(empties):
            return True  # all filled

        # select cell with Minimum Remaining Values
        min_pos = -1
        min_options = n + 1
        for i in range(k, len(empties)):
            r, c = empties[i]
            opts = full_mask & ~(row_mask[r] | col_mask[c])
            cnt = opts.bit_count()
            if cnt < min_options:
                min_options = cnt
                min_pos = i
                if cnt == 1:
                    break
        if min_options == 0:
            return False

        # bring chosen cell to position k
        empties[k], empties[min_pos] = empties[min_pos], empties[k]
        r, c = empties[k]

        # iterate over possible values
        avail_mask = full_mask & ~(row_mask[r] | col_mask[c])
        while avail_mask:
            lsb = avail_mask & -avail_mask
            v = (lsb.bit_length() - 1) + 1
            avail_mask ^= lsb

            bit = 1 << (v - 1)
            grid[r][c] = v
            row_mask[r] |= bit
            col_mask[c] |= bit

            if dfs(k + 1):
                return True

            # undo
            grid[r][c] = 0
            row_mask[r] ^= bit
            col_mask[c] ^= bit

        # restore order before backtrack
        empties[k], empties[min_pos] = empties[min_pos], empties[k]
        return False

    dfs(0)

    return {"grid": grid}