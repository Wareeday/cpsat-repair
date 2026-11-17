"""Outcome of running one generated program on one instance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

OK = "ok"                        # valid, and optimal when the family is an optimisation problem
SUBOPTIMAL = "suboptimal"        # valid but a strictly better solution exists
INVALID = "invalid"              # output is well-formed but violates a constraint
FORMAT_ERROR = "format_error"    # output does not match the output schema
RUNTIME_ERROR = "runtime_error"  # exception, syntax error, missing solve(), solver returned nothing
TIMEOUT = "timeout"              # exceeded the wall-clock limit
NO_CODE = "no_code"              # no python code block in the model's reply

ALL_STATUSES = [OK, SUBOPTIMAL, INVALID, FORMAT_ERROR, RUNTIME_ERROR, TIMEOUT, NO_CODE]


@dataclass
class Verdict:
    status: str
    message: str = ""
    objective: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.status == OK
