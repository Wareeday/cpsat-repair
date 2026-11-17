"""Common interface for a problem family.

Every family provides
  * a generator with five size levels,
  * a trusted reference solver (CP-SAT) used only to obtain ground-truth optima,
  * an independent checker written in plain Python (no OR-Tools import) that
    verifies feasibility and recomputes the objective from scratch.
"""
from __future__ import annotations

import json
import random
from typing import Any, Optional

from ..verdict import FORMAT_ERROR, INVALID, OK, SUBOPTIMAL, Verdict


class FormatError(Exception):
    """Output does not match the documented output schema."""


class Violation(Exception):
    """Output is well-formed but breaks a constraint. The message is the counterexample."""


def as_int(x: Any, what: str) -> int:
    if isinstance(x, bool):
        raise FormatError(f"{what} must be an integer, got a boolean")
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    raise FormatError(f"{what} must be an integer, got {x!r}")


def as_list(x: Any, what: str, length: Optional[int] = None) -> list:
    if not isinstance(x, (list, tuple)):
        raise FormatError(f"{what} must be a list, got {type(x).__name__}")
    if length is not None and len(x) != length:
        raise FormatError(f"{what} must have length {length}, got {len(x)}")
    return list(x)


class Family:
    name: str = ""
    title: str = ""
    sense: Optional[str] = None          # "min", "max" or None (pure satisfaction)
    objective_text: str = ""
    sizes: list[dict] = []               # five entries, each with a "label"
    statement: str = ""
    input_schema: str = ""
    output_schema: str = ""
    example_input: dict = {}
    example_output: dict = {}

    # -- to be provided by each family ------------------------------------
    def generate(self, size: dict, rng: random.Random) -> dict:
        raise NotImplementedError

    def reference(self, data: dict) -> tuple[dict, Optional[int]]:
        """Trusted CP-SAT model. Must prove optimality for optimisation families."""
        raise NotImplementedError

    def evaluate(self, data: dict, solution: dict) -> Optional[int]:
        """Raise FormatError / Violation, else return the recomputed objective (or None)."""
        raise NotImplementedError

    # -- shared logic -----------------------------------------------------
    def check(self, data: dict, solution: Any, ref_objective: Optional[int]) -> Verdict:
        if not isinstance(solution, dict):
            return Verdict(FORMAT_ERROR, f"solve() must return a dict, got {type(solution).__name__}")
        try:
            obj = self.evaluate(data, solution)
        except FormatError as e:
            return Verdict(FORMAT_ERROR, str(e))
        except Violation as e:
            return Verdict(INVALID, str(e))
        if self.sense is None or ref_objective is None:
            return Verdict(OK, "valid", obj)
        better = obj < ref_objective if self.sense == "min" else obj > ref_objective
        worse = obj > ref_objective if self.sense == "min" else obj < ref_objective
        if worse:
            return Verdict(SUBOPTIMAL, f"valid, objective = {obj}, but a strictly better solution exists", obj)
        if better:
            # The checker recomputed a better value than the proven optimum: a bug on our side.
            raise AssertionError(f"{self.name}: objective {obj} beats the reference optimum {ref_objective}")
        return Verdict(OK, f"valid and optimal (objective = {obj})", obj)

    def prompt_block(self) -> str:
        parts = [
            f"PROBLEM: {self.title}",
            self.statement.strip(),
            "",
            "INPUT: the `instance` dict that solve() receives:",
            self.input_schema.strip(),
            "",
            "OUTPUT: the dict that solve() must return:",
            self.output_schema.strip(),
            "",
        ]
        if self.sense:
            word = "minimise" if self.sense == "min" else "maximise"
            parts.append(f"OBJECTIVE: {word} {self.objective_text}. An answer only counts as correct if it is valid AND optimal.")
        else:
            parts.append("OBJECTIVE: none. Any valid answer is correct.")
        parts += [
            "",
            "Example instance:",
            json.dumps(self.example_input),
            "Example of a valid output for it:",
            json.dumps(self.example_output),
        ]
        return "\n".join(parts)

    def describe(self, data: dict) -> str:
        """One-line size summary used in counterexample messages."""
        return ", ".join(f"{k}={v}" for k, v in data.items() if isinstance(v, int))


def get_field(solution: dict, key: str) -> Any:
    if key not in solution:
        raise FormatError(f"the returned dict is missing the key '{key}'")
    return solution[key]
