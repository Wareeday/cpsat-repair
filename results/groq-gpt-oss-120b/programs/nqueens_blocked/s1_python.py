import time
import sys

def solve(instance: dict) -> dict:
    n = instance["n"]
    blocked = instance.get("blocked", [])
    # Bitmask of allowed columns for each row
    full_mask = (1 << n) - 1
    allowed = [full_mask for _ in range(n)]
    for r, c in blocked:
        allowed[r] &= ~(1 << c)

    # Order rows by fewest possibilities (MRV heuristic)
    row_order = list(range(n))
    row_order.sort(key=lambda r: allowed[r].bit_count())

    # State arrays
    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)   # index = r + c
    diag2_used = [False] * (2 * n - 1)   # index = r - c + n - 1
    solution = [-1] * n

    TIME_LIMIT = 28.0  # seconds
    start_time = time.time()
    sys.setrecursionlimit(max(1000, n + 10))

    def dfs(idx: int) -> bool:
        # time check
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError
        if idx == n:
            return True
        r = row_order[idx]
        mask = allowed[r]
        while mask:
            lowbit = mask & -mask
            c = lowbit.bit_length() - 1
            mask ^= lowbit
            d1 = r + c
            d2 = r - c + n - 1
            if not col_used[c] and not diag1_used[d1] and not diag2_used[d2]:
                col_used[c] = diag1_used[d1] = diag2_used[d2] = True
                solution[r] = c
                if dfs(idx + 1):
                    return True
                col_used[c] = diag1_used[d1] = diag2_used[d2] = False
                solution[r] = -1
        return False

    try:
        dfs(0)
    except TimeoutError:
        # In the unlikely event of a timeout, return the partial (invalid) result;
        # problem guarantees a solution exists and typical instances are solved fast.
        pass

    return {"queens": solution}