"""Prompt construction, code extraction and counterexample messages."""
from __future__ import annotations

import json
import re
from typing import Optional

from .families.base import Family
from .verdict import (FORMAT_ERROR, INVALID, NO_CODE, RUNTIME_ERROR, SUBOPTIMAL, TIMEOUT, Verdict)

KINDS = ("cpsat", "python")
TIME_LIMIT_HINT = 20

SYSTEM = (
    "You are an expert in combinatorial optimisation and Python. You write correct, self-contained code. "
    "You reply with exactly one ```python code block and no other text."
)

_CPSAT_RULES = f"""Write a Python module that defines `solve(instance: dict) -> dict`.
- Formulate the problem as a CP-SAT model with Google OR-Tools (`from ortools.sat.python import cp_model`): \
create variables, add the constraints, add the objective if there is one, and let the solver do the search. \
Do not write your own search algorithm.
- Set `solver.parameters.max_time_in_seconds = {TIME_LIMIT_HINT}`. If the solver stops without proving \
optimality, return the best solution it found.
- Return plain Python ints and lists exactly as described in OUTPUT (no OR-Tools objects). Do not print anything.
- The function will be tested on hidden instances of several sizes, so do not hard-code any data."""

_PYTHON_RULES = f"""Write a Python module that defines `solve(instance: dict) -> dict`.
- Use only the Python standard library. Do not import ortools, numpy, scipy, pulp or any other package.
- Implement the algorithm yourself (exact search, dynamic programming, greedy, local search: your choice).
- The function is stopped after about {TIME_LIMIT_HINT + 10} seconds. Keep track of time and return the best \
valid solution you have before that. Do not print anything.
- Return plain Python ints and lists exactly as described in OUTPUT.
- The function will be tested on hidden instances of several sizes, so do not hard-code any data."""


def initial_messages(fam: Family, kind: str) -> list[dict]:
    rules = _CPSAT_RULES if kind == "cpsat" else _PYTHON_RULES
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{fam.prompt_block()}\n\n{rules}"},
    ]


_BLOCK = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> Optional[str]:
    blocks = [b.strip("\n") for b in _BLOCK.findall(reply)]
    with_solve = [b for b in blocks if re.search(r"def\s+solve\s*\(", b)]
    if with_solve:
        return max(with_solve, key=len)
    if re.search(r"^\s*def\s+solve\s*\(", reply, re.MULTILINE) and "```" not in reply:
        return reply
    return None


def counterexample_message(fam: Family, data: dict, verdict: Verdict, timeout: float,
                           ref_objective: Optional[int] = None, reveal_optimum: bool = False) -> str:
    head = "Your program was run on a hidden test instance and it did NOT pass."
    if verdict.status == NO_CODE:
        body = "Your reply did not contain a ```python code block that defines solve(instance)."
    elif verdict.status == RUNTIME_ERROR:
        body = f"The program crashed or returned nothing usable:\n{verdict.message}"
    elif verdict.status == TIMEOUT:
        body = (f"The program did not finish within {timeout:g} seconds. Make the model or the search more "
                "efficient, and always return the best valid solution before the time runs out.")
    elif verdict.status == FORMAT_ERROR:
        body = f"The returned value does not match the OUTPUT format: {verdict.message}"
    elif verdict.status == INVALID:
        body = f"The returned solution is not valid. Counterexample: {verdict.message}"
    elif verdict.status == SUBOPTIMAL:
        body = (f"The returned solution is valid (objective = {verdict.objective}) but it is not optimal: "
                "a strictly better solution exists.")
        if reveal_optimum and ref_objective is not None:
            body += f" The optimal objective value is {ref_objective}."
    else:
        body = verdict.message
    parts = [head, "", f"Instance size: {fam.describe(data)}", body]
    text = json.dumps(data)
    if len(text) <= 1500:
        parts += ["", f"The instance was: {text}"]
    parts += [
        "",
        "Find the mistake and fix it so that the program is correct for ALL instances, not just this one. "
        "Reply with the complete corrected module in a single ```python code block and nothing else.",
    ]
    return "\n".join(parts)
