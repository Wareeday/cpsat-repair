"""Run generated code in a child process with a wall-clock limit.

This is a resource guard, not a security sandbox: the code comes from a language model and runs
with your user's permissions. Use a container if you do not trust the model.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
POSIX = os.name == "posix"


@dataclass
class RunResult:
    status: str                # "ok", "timeout" or "runtime_error"
    output: Any = None
    error: str = ""
    seconds: float = 0.0


def _limit_memory(mem_mb: int):
    def apply() -> None:
        import resource
        limit = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    return apply


def run_program(code: str, data: dict, timeout: float = 30.0, mem_mb: Optional[int] = 4096) -> RunResult:
    with tempfile.TemporaryDirectory(prefix="cpsr_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "prog.py").write_text(code)
        (tmp_path / "inst.json").write_text(json.dumps(data))
        out_file = tmp_path / "out.json"
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        cmd = [sys.executable, "-m", "cpsr._runner", str(tmp_path / "prog.py"), str(tmp_path / "inst.json"), str(out_file)]
        kwargs: dict[str, Any] = {}
        if POSIX:
            kwargs["start_new_session"] = True
            if mem_mb:
                kwargs["preexec_fn"] = _limit_memory(mem_mb)
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=tmp, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kwargs)
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if POSIX:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.communicate()
            return RunResult("timeout", error=f"no answer within {timeout:g} seconds", seconds=time.time() - t0)
        seconds = time.time() - t0
        if not out_file.exists():
            tail = (stderr or b"").decode(errors="replace").strip()[-800:]
            return RunResult("runtime_error", error=f"the process died (exit code {proc.returncode}). {tail}", seconds=seconds)
        payload = json.loads(out_file.read_text())
        if not payload["ok"]:
            return RunResult("runtime_error", error=payload["error"], seconds=seconds)
        return RunResult("ok", output=payload["output"], seconds=seconds)
