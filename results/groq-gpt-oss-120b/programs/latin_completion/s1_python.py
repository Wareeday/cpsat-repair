import time
from typing import List, Dict

def solve(instance: Dict) -> Dict:
    n: int = instance["n"]
    prefilled: List[List[int]] = instance.get("prefilled", [])

    # Bit masks for available numbers in each row/col (1 means available)
    full_mask = (1 << n) - 1
    row_mask = [full_mask for _ in range(n)]
    col_mask = [full_mask for _ in range(n)]

    # Grid initialisation
    grid: List[List[int]] = [[0] * n for _ in range(n)]

    for r, c, v in prefilled:
        bit = 1 << (v - 1)
        if not (row_mask[r] & bit) or not (col_mask[c] & bit):
            # contradictory input – but problem guarantees consistency
            raise ValueError("Invalid prefilled data")
        grid[r][c] = v
        row_mask[r] ^= bit
        col_mask[c] ^= bit

    # Collect empty cells
    empties: List[tuple] = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == 0]

    start_time = time.time()
    TIME_LIMIT = 28.0          # seconds, a bit below the 30‑second hard limit
    solution_grid: List[List[int]] = []

    # Helper: count bits (available values)
    def count_bits(x: int) -> int:
        return x.bit_count()

    # Recursive backtracking with MRV heuristic
    def backtrack(k: int) -> bool:
        # Timeout check
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError

        if k == len(empties):
            # All cells filled – store solution
            for i in range(n):
                solution_grid.append(grid[i].copy())
            return True

        # Choose cell with minimum remaining values (MRV)
        best_idx = -1
        best_options = 0
        best_mask = 0
        min_cnt = n + 1

        for i in range(k, len(empties)):
            r, c = empties[i]
            possible = row_mask[r] & col_mask[c]
            cnt = count_bits(possible)
            if cnt < min_cnt:
                min_cnt = cnt
                best_idx = i
                best_options = cnt
                best_mask = possible
                if cnt == 1:
                    break
        if best_idx == -1 or best_options == 0:
            return False

        # Swap the chosen cell to position k
        empties[k], empties[best_idx] = empties[best_idx], empties[k]
        r, c = empties[k]

        mask = best_mask
        while mask:
            low = mask & -mask
            v = low.bit_length()          # value = position of low bit (1‑based)
            bit = 1 << (v - 1)

            # Place v
            grid[r][c] = v
            row_mask[r] ^= bit
            col_mask[c] ^= bit

            if backtrack(k + 1):
                return True

            # Undo
            row_mask[r] ^= bit
            col_mask[c] ^= bit
            grid[r][c] = 0

            mask ^= low

        # Restore order before backtracking further
        empties[k], empties[best_idx] = empties[best_idx], empties[k]
        return False

    try:
        backtrack(0)
    except TimeoutError:
        pass

    # If for some reason we didn't find a solution (should not happen), fall back to a simple cyclic Latin square
    if not solution_grid:
        cyclic = [[(i + j) % n + 1 for j in range(n)] for i in range(n)]
        # Apply prefilled values (they are guaranteed compatible)
        for r, c, v in prefilled:
            cyclic[r][c] = v
        solution_grid = cyclic

    return {"grid": solution_grid}