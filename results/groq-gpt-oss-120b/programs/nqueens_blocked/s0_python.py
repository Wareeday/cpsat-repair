import time

def solve(instance: dict) -> dict:
    n = instance["n"]
    blocked = set((r, c) for r, c in instance.get("blocked", []))

    # allowed columns per row (excluding blocked squares)
    allowed_per_row = [
        [c for c in range(n) if (r, c) not in blocked] for r in range(n)
    ]

    solution = [None] * n          # column chosen for each row
    used_cols = 0                  # bitmask of occupied columns
    used_diag1 = 0                 # bitmask of occupied main diagonals (r - c + n - 1)
    used_diag2 = 0                 # bitmask of occupied anti diagonals (r + c)

    start_time = time.time()
    TIME_LIMIT = 28.0  # seconds, a bit less than the overall limit

    def dfs(placed: int) -> bool:
        """Return True if a complete placement is found."""
        nonlocal used_cols, used_diag1, used_diag2

        # timeout check
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError

        if placed == n:
            return True

        # Choose the unfilled row with the fewest legal columns (MRV heuristic)
        best_row = -1
        best_options = None
        for r in range(n):
            if solution[r] is not None:
                continue
            opts = []
            for c in allowed_per_row[r]:
                col_bit = 1 << c
                d1 = r - c + n - 1
                d2 = r + c
                if (used_cols & col_bit) or (used_diag1 & (1 << d1)) or (used_diag2 & (1 << d2)):
                    continue
                opts.append(c)
            if not opts:          # dead end
                return False
            if best_options is None or len(opts) < len(best_options):
                best_options = opts
                best_row = r
                if len(best_options) == 1:
                    break

        r = best_row
        for c in best_options:
            col_bit = 1 << c
            d1 = r - c + n - 1
            d2 = r + c

            # place queen
            solution[r] = c
            used_cols ^= col_bit
            used_diag1 ^= (1 << d1)
            used_diag2 ^= (1 << d2)

            if dfs(placed + 1):
                return True

            # backtrack
            used_cols ^= col_bit
            used_diag1 ^= (1 << d1)
            used_diag2 ^= (1 << d2)
            solution[r] = None

        return False

    try:
        dfs(0)
    except TimeoutError:
        pass

    # If the search timed out before finding a full solution, try a simple greedy fill.
    if any(v is None for v in solution):
        used_cols = 0
        used_diag1 = 0
        used_diag2 = 0
        for r, c in enumerate(solution):
            if c is not None:
                used_cols |= 1 << c
                used_diag1 |= 1 << (r - c + n - 1)
                used_diag2 |= 1 << (r + c)

        for r in range(n):
            if solution[r] is not None:
                continue
            for c in allowed_per_row[r]:
                if (used_cols >> c) & 1:
                    continue
                d1 = r - c + n - 1
                d2 = r + c
                if (used_diag1 >> d1) & 1 or (used_diag2 >> d2) & 1:
                    continue
                solution[r] = c
                used_cols |= 1 << c
                used_diag1 |= 1 << d1
                used_diag2 |= 1 << d2
                break

    return {"queens": solution}