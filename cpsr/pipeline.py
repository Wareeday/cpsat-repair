"""The experiment: three conditions per (instance, sample).

  python        the model writes plain-Python search code; one shot
  cpsat         the model writes a CP-SAT model; one shot
  cpsat_repair  same program as `cpsat` in round 0; if it fails, the checker's counterexample is sent back
                and the model rewrites the program, at most `repair_rounds` times

`cpsat` and `cpsat_repair` share their round-0 program, so the comparison between them is paired.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .families import get_family
from .families.base import Family
from .llm import Backend, LLMCache
from .prompts import counterexample_message, extract_code, initial_messages
from .sandbox import run_program
from .verdict import NO_CODE, RUNTIME_ERROR, TIMEOUT, Verdict

CONDITIONS = ("python", "cpsat", "cpsat_repair")


@dataclass
class Settings:
    samples: int = 3
    temperature: float = 0.7
    repair_rounds: int = 3
    timeout: float = 30.0
    mem_mb: Optional[int] = 4096
    sleep: float = 0.0
    reveal_optimum: bool = False


@dataclass
class Program:
    messages: list[dict]      # the conversation so far, ending with the model's reply
    code: Optional[str]


class Experiment:
    def __init__(self, backend: Backend, out_dir: Path, settings: Settings):
        self.backend, self.out_dir, self.cfg = backend, out_dir, settings
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "programs").mkdir(exist_ok=True)
        self.cache = LLMCache(out_dir / "llm_cache.jsonl")
        self.runs_path = out_dir / "runs.jsonl"
        self.done: set[tuple[str, int, str]] = set()
        self.last: dict[tuple[str, int, str], str] = {}      # for the progress line only
        if self.runs_path.exists():
            for line in self.runs_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.done.add((r["id"], r["sample"], r["condition"]))
                    self.last[(r["id"], r["sample"], r["condition"])] = r["status"]

    # -- model calls ------------------------------------------------------
    def _ask(self, messages: list[dict], sample: int, rnd: int, meta: dict) -> str:
        seed = 1000 * sample + rnd
        key = LLMCache.key(self.backend.id, messages, self.cfg.temperature, seed)
        reply = self.cache.get(key)
        if reply is None:
            reply = self.backend.chat(messages, temperature=self.cfg.temperature, seed=seed,
                                      meta={**meta, "sample": sample, "round": rnd})
            self.cache.put(key, reply)
            if self.cfg.sleep:
                time.sleep(self.cfg.sleep)
        return reply

    def _first_program(self, fam: Family, kind: str, sample: int) -> Program:
        messages = initial_messages(fam, kind)
        reply = self._ask(messages, sample, 0, {"family": fam.name, "kind": kind})
        code = extract_code(reply)
        self._save(fam.name, f"s{sample}_{kind}.py", code or f"# no code block found\n# {reply[:2000]!r}\n")
        return Program(messages + [{"role": "assistant", "content": reply}], code)

    def _save(self, family: str, name: str, text: str) -> None:
        folder = self.out_dir / "programs" / family
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text)

    # -- running and checking --------------------------------------------
    def _evaluate(self, fam: Family, code: Optional[str], inst: dict) -> tuple[Verdict, float]:
        if code is None:
            return Verdict(NO_CODE, "no python code block that defines solve() in the reply"), 0.0
        res = run_program(code, inst["data"], self.cfg.timeout, self.cfg.mem_mb)
        if res.status == "timeout":
            return Verdict(TIMEOUT, res.error), res.seconds
        if res.status == "runtime_error":
            return Verdict(RUNTIME_ERROR, res.error), res.seconds
        return fam.check(inst["data"], res.output, inst["ref_objective"]), res.seconds

    def _repair(self, fam: Family, inst: dict, sample: int, prog: Program, verdict: Verdict, seconds: float):
        messages, code = list(prog.messages), prog.code
        trace, rounds = [verdict.status], 0
        while not verdict.ok and rounds < self.cfg.repair_rounds:
            rounds += 1
            messages.append({"role": "user", "content": counterexample_message(
                fam, inst["data"], verdict, self.cfg.timeout, inst["ref_objective"], self.cfg.reveal_optimum,
                inst["size_label"])})
            reply = self._ask(messages, sample, rounds, {"family": fam.name, "kind": "cpsat"})
            messages.append({"role": "assistant", "content": reply})
            code = extract_code(reply)
            verdict, seconds = self._evaluate(fam, code, inst)
            trace.append(verdict.status)
            self._save(fam.name, f"s{sample}_{inst['id']}_repair{rounds}.py", code or "# no code block found\n")
        return verdict, seconds, rounds, trace, code

    def _heldout(self, fam: Family, code: Optional[str], inst: dict, insts: list[dict]) -> Optional[str]:
        """Does a repaired program also solve another instance of the same family and size level?

        The counterexample came from `inst`, so passing `inst` again proves little: the model may have
        patched that one instance. The sibling instance was never shown to the model.
        """
        sibling = next((i for i in insts if i["size_idx"] == inst["size_idx"] and i["id"] != inst["id"]), None)
        if sibling is None:
            return None
        verdict, _ = self._evaluate(fam, code, sibling)
        return verdict.status

    def _log(self, inst: dict, sample: int, condition: str, verdict: Verdict, seconds: float,
             rounds: int = 0, trace: Optional[list] = None, heldout: Optional[str] = None) -> None:
        row = {
            "id": inst["id"], "family": inst["family"], "size_idx": inst["size_idx"],
            "size_label": inst["size_label"], "sample": sample, "condition": condition,
            "status": verdict.status, "message": verdict.message[:600], "objective": verdict.objective,
            "ref_objective": inst["ref_objective"], "rounds": rounds,
            "trace": trace or [verdict.status], "heldout": heldout,
            "seconds": round(seconds, 3), "model": self.backend.id,
        }
        with open(self.runs_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.done.add((inst["id"], sample, condition))
        self.last[(inst["id"], sample, condition)] = verdict.status + (f"({rounds}r)" if condition == "cpsat_repair" else "")

    # -- main loop --------------------------------------------------------
    def run(self, instances: list[dict]) -> None:
        by_family: dict[str, list[dict]] = {}
        for inst in instances:
            by_family.setdefault(inst["family"], []).append(inst)
        for name, insts in by_family.items():
            fam = get_family(name)
            for sample in range(self.cfg.samples):
                todo = [i for i in insts if any((i["id"], sample, c) not in self.done for c in CONDITIONS)]
                if not todo:
                    continue
                py = self._first_program(fam, "python", sample)
                cp = self._first_program(fam, "cpsat", sample)
                for inst in todo:
                    if (inst["id"], sample, "python") not in self.done:
                        v, s = self._evaluate(fam, py.code, inst)
                        self._log(inst, sample, "python", v, s)
                    if any((inst["id"], sample, c) not in self.done for c in ("cpsat", "cpsat_repair")):
                        v0, s0 = self._evaluate(fam, cp.code, inst)
                        if (inst["id"], sample, "cpsat") not in self.done:
                            self._log(inst, sample, "cpsat", v0, s0)
                        if (inst["id"], sample, "cpsat_repair") not in self.done:
                            v, s, rounds, trace, final_code = self._repair(fam, inst, sample, cp, v0, s0)
                            heldout = self._heldout(fam, final_code, inst, insts) if v.ok and rounds else None
                            self._log(inst, sample, "cpsat_repair", v, s, rounds, trace, heldout)
                    shown = "  ".join(f"{c}={self.last[(inst['id'], sample, c)]}" for c in CONDITIONS)
                    print(f"{inst['id']:28s} sample {sample}  {shown}", flush=True)
