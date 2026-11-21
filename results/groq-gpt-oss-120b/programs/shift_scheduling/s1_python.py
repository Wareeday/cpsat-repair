import time
import itertools
from copy import deepcopy

def solve(instance: dict) -> dict:
    # ---------- data ----------
    E = instance["n_employees"]
    D = instance["n_days"]
    demand = instance["demand"]               # list of [m,a,n] per day
    req_off = set((e, d) for e, d in instance["requests_off"])
    min_shifts = instance["min_shifts"]
    max_shifts = instance["max_shifts"]

    # ---------- helpers ----------
    # total demand remaining from each day (including that day)
    total_demand_rem = [0] * (D + 1)
    for day in range(D - 1, -1, -1):
        total_demand_rem[day] = total_demand_rem[day + 1] + sum(demand[day])

    # quick employee list
    employees = list(range(E))

    # ---------- state ----------
    # -2 = not yet decided for that day, -1 = off, 0/1/2 = shift
    assign = [[-2] * D for _ in range(E)]
    shifts_assigned = [0] * E          # total shifts already given to employee
    best_solution = None
    best_viol = float('inf')
    start_time = time.time()
    TIME_LIMIT = 28.0   # a bit under 30s to be safe

    # ---------- feasibility checks ----------
    def feasible_remaining(day_idx: int) -> bool:
        """Check simple capacity constraints for days >= day_idx."""
        # remaining demand
        rem_demand = total_demand_rem[day_idx]
        # capacity left according to max_shifts
        cap_left = sum(max_shifts - shifts_assigned[e] for e in employees)
        if rem_demand > cap_left:
            return False
        # minimum needed to satisfy min_shifts
        min_needed = sum(max(0, min_shifts - shifts_assigned[e]) for e in employees)
        if min_needed > rem_demand:
            return False
        return True

    def can_work(e: int, day: int, shift: int) -> bool:
        """Check if employee e can take `shift` on `day` respecting local constraints."""
        # already at max?
        if shifts_assigned[e] >= max_shifts:
            return False
        # night->morning restriction
        if shift == 0 and day > 0 and assign[e][day - 1] == 2:
            return False
        return True

    # ---------- recursive search ----------
    def dfs_day(day: int, cur_viol: int):
        nonlocal best_solution, best_viol

        # time check
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError

        if day == D:
            # all days assigned – check min_shifts
            if all(min_shifts <= shifts_assigned[e] <= max_shifts for e in employees):
                if cur_viol < best_viol:
                    best_viol = cur_viol
                    best_solution = deepcopy(assign)
            return

        # order shifts by decreasing demand to reduce branching
        shift_order = sorted([0, 1, 2], key=lambda s: -demand[day][s])

        # recursive assignment of the three shift groups
        def assign_shifts(idx: int, used_set: set, cur_viol_local: int):
            nonlocal best_solution, best_viol

            # prune by current violations
            if cur_viol_local >= best_viol:
                return
            # time check
            if time.time() - start_time > TIME_LIMIT:
                raise TimeoutError

            if idx == 3:  # all shifts for this day decided
                # set off for the rest
                off_emps = [e for e in employees if e not in used_set]
                for e in off_emps:
                    assign[e][day] = -1
                # after fixing the day, check simple feasibility for the rest
                if feasible_remaining(day + 1):
                    dfs_day(day + 1, cur_viol_local)
                # backtrack off marks
                for e in off_emps:
                    assign[e][day] = -2
                return

            shift = shift_order[idx]
            need = demand[day][shift]
            if need == 0:
                assign_shifts(idx + 1, used_set, cur_viol_local)
                return

            # candidate employees for this shift
            candidates = [e for e in employees
                          if e not in used_set and can_work(e, day, shift)]

            # If not enough candidates, prune
            if len(candidates) < need:
                return

            # iterate over combinations
            for combo in itertools.combinations(candidates, need):
                added_viol = 0
                for e in combo:
                    assign[e][day] = shift
                    shifts_assigned[e] += 1
                    if (e, day) in req_off:
                        added_viol += 1
                new_used = used_set.union(combo)
                # quick prune: respect max_shifts already handled, but also check capacity left for remaining days
                if feasible_remaining(day):
                    assign_shifts(idx + 1, new_used, cur_viol_local + added_viol)
                # backtrack
                for e in combo:
                    assign[e][day] = -2
                    shifts_assigned[e] -= 1

        assign_shifts(0, set(), cur_viol)

    # ---------- run ----------
    try:
        dfs_day(0, 0)
    except TimeoutError:
        pass

    # If no solution found (should not happen), fall back to any feasible one via greedy
    if best_solution is None:
        # simple greedy: fill each day respecting constraints, ignoring requests
        # (guaranteed feasible by problem statement)
        assign = [[-1] * D for _ in range(E)]
        shifts_assigned = [0] * E
        for day in range(D):
            # order shifts by demand descending
            shift_order = sorted([0, 1, 2], key=lambda s: -demand[day][s])
            used = set()
            for shift in shift_order:
                need = demand[day][shift]
                if need == 0:
                    continue
                cand = [e for e in employees if e not in used and can_work(e, day, shift)]
                for e in cand[:need]:
                    assign[e][day] = shift
                    shifts_assigned[e] += 1
                    used.add(e)
            # rest stay off (already -1)
        best_solution = assign

    # ---------- output ----------
    return {"roster": best_solution}