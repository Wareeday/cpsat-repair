from __future__ import annotations

import random
from typing import Optional

from ortools.sat.python import cp_model

from ._cp import solve_reference
from .base import Family, Violation, as_int, as_list, get_field


class JobShop(Family):
    name = "job_shop"
    title = "Job-shop scheduling (minimum makespan)"
    sense = "min"
    objective_text = "the makespan, i.e. the largest completion time over all operations"
    sizes = [
        {"label": f"{j}x{m}", "jobs": j, "machines": m}
        for j, m in ((3, 3), (4, 4), (5, 5), (7, 6), (10, 8))
    ]
    statement = (
        "Each job is a list of operations that must be executed in the given order. An operation is a pair "
        "(machine, duration). An operation may start only after the previous operation of its job has "
        "finished. A machine processes one operation at a time and operations cannot be interrupted. Start "
        "times are non-negative integers. The makespan is the maximum of start + duration over all "
        "operations. Jobs, operations and machines are numbered from 0."
    )
    input_schema = '{"jobs": [[[machine, duration], ...], ...], "n_machines": int}'
    output_schema = '{"starts": [[s_00, s_01, ...], ...]}  starts[j][k] is the start time of operation k of job j.'
    example_input = {"jobs": [[[0, 3], [1, 2]], [[1, 2], [0, 4]]], "n_machines": 2}
    example_output = {"starts": [[0, 3], [0, 5]]}

    def generate(self, size: dict, rng: random.Random) -> dict:
        jobs = []
        for _ in range(size["jobs"]):
            order = list(range(size["machines"]))
            rng.shuffle(order)
            jobs.append([[mach, rng.randint(1, 9)] for mach in order])
        return {"jobs": jobs, "n_machines": size["machines"]}

    def reference(self, data: dict) -> tuple[dict, Optional[int]]:
        jobs = data["jobs"]
        horizon = sum(d for job in jobs for _, d in job)
        m = cp_model.CpModel()
        start, ivs = [], {mach: [] for mach in range(data["n_machines"])}
        for j, job in enumerate(jobs):
            row = []
            for k, (mach, dur) in enumerate(job):
                s = m.new_int_var(0, horizon, f"s{j}_{k}")
                ivs[mach].append(m.new_interval_var(s, dur, s + dur, f"i{j}_{k}"))
                if k:
                    m.add(s >= row[-1] + job[k - 1][1])
                row.append(s)
            start.append(row)
        for mach in ivs:
            m.add_no_overlap(ivs[mach])
        mk = m.new_int_var(0, horizon, "makespan")
        for j, job in enumerate(jobs):
            m.add(mk >= start[j][-1] + job[-1][1])
        m.minimize(mk)
        s = solve_reference(m, optimise=True)
        return {"starts": [[s.value(v) for v in row] for row in start]}, int(s.objective_value)

    def evaluate(self, data: dict, solution: dict) -> Optional[int]:
        jobs = data["jobs"]
        raw = as_list(get_field(solution, "starts"), "starts", len(jobs))
        st = [
            [as_int(v, f"starts[{j}][{k}]") for k, v in enumerate(as_list(row, f"starts[{j}]", len(jobs[j])))]
            for j, row in enumerate(raw)
        ]
        for j, job in enumerate(jobs):
            for k, (_, dur) in enumerate(job):
                if st[j][k] < 0:
                    raise Violation(f"Operation {k} of job {j} starts at {st[j][k]}, which is negative.")
                if k and st[j][k] < st[j][k - 1] + job[k - 1][1]:
                    raise Violation(
                        f"Operation {k} of job {j} starts at {st[j][k]} but operation {k - 1} of the same job "
                        f"only finishes at {st[j][k - 1] + job[k - 1][1]}."
                    )
        by_machine: dict[int, list[tuple[int, int, int, int]]] = {}
        for j, job in enumerate(jobs):
            for k, (mach, dur) in enumerate(job):
                by_machine.setdefault(mach, []).append((st[j][k], st[j][k] + dur, j, k))
        for mach, ops in sorted(by_machine.items()):
            ops.sort()
            for (s1, e1, j1, k1), (s2, e2, j2, k2) in zip(ops, ops[1:]):
                if s2 < e1:
                    raise Violation(
                        f"On machine {mach}, operation {k1} of job {j1} runs during [{s1}, {e1}) and operation "
                        f"{k2} of job {j2} runs during [{s2}, {e2}); they overlap."
                    )
        return max(st[j][k] + jobs[j][k][1] for j in range(len(jobs)) for k in range(len(jobs[j])))
