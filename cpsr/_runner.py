"""Executed in a child process: import the generated module, call solve(instance), write JSON."""
from __future__ import annotations

import importlib.util
import json
import sys
import traceback


def _default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    try:
        return int(o)          # numpy integers and the like
    except Exception:
        return f"<unserialisable {type(o).__name__}>"


def main(prog_path: str, inst_path: str, out_path: str) -> None:
    with open(inst_path) as f:
        data = json.load(f)
    try:
        spec = importlib.util.spec_from_file_location("candidate", prog_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "solve", None)):
            raise RuntimeError("the module does not define a function solve(instance)")
        payload = {"ok": True, "output": module.solve(data)}
        text = json.dumps(payload, default=_default)
    except BaseException as e:
        if isinstance(e, SyntaxError):
            msg = "".join(traceback.format_exception_only(type(e), e))
        else:
            tb = e.__traceback__.tb_next or e.__traceback__      # drop the runner's own frame
            msg = "".join(traceback.format_exception(type(e), e, tb))
        msg = msg.replace(prog_path, "solution.py").strip()
        text = json.dumps({"ok": False, "error": "\n".join(msg.splitlines()[-12:])[-1500:]})
    with open(out_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main(*sys.argv[1:4])
