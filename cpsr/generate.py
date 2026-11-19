"""Create data/instances.jsonl: every family x five sizes x k instances, with proven optima.

    python -m cpsr.generate --per-size 2

Instances for which the reference CP-SAT model cannot prove optimality within the time limit are
discarded and re-drawn with a different seed (otherwise there would be no ground truth).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .families import all_families, get_family
from .families import _cp

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "instances.jsonl"


def build_instances(per_size: int, families: list[str] | None, time_limit: float, verbose: bool = True) -> list[dict]:
    chosen = [get_family(n) for n in families] if families else all_families()
    _cp.TIME_LIMIT = time_limit
    rows = []
    for fam in chosen:
        for size_idx, size in enumerate(fam.sizes):
            for k in range(per_size):
                for attempt in range(30):
                    rng = random.Random(f"{fam.name}|{size_idx}|{k}|{attempt}")
                    data = fam.generate(size, rng)
                    t0 = time.time()
                    try:
                        solution, ref_obj = fam.reference(data)
                    except RuntimeError:
                        continue
                    assert fam.check(data, solution, ref_obj).ok, f"reference solution rejected by checker: {fam.name}"
                    rows.append({
                        "id": f"{fam.name}-{size_idx + 1}-{k}", "family": fam.name, "size_idx": size_idx + 1,
                        "size_label": size["label"], "seed": k, "attempt": attempt,
                        "ref_seconds": round(time.time() - t0, 3), "ref_objective": ref_obj, "data": data,
                    })
                    break
                else:
                    raise RuntimeError(f"no solvable instance found for {fam.name} size {size['label']}")
            if verbose:
                print(f"{fam.name:20s} {size['label']:12s} done", flush=True)
    return rows


def load_instances(path: Path = DEFAULT_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-size", type=int, default=2, help="instances per family and size level")
    ap.add_argument("--families", help="comma-separated subset")
    ap.add_argument("--time-limit", type=float, default=60.0, help="reference solver limit in seconds")
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args()
    rows = build_instances(args.per_size, args.families.split(",") if args.families else None, args.time_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} instances to {args.out}")


if __name__ == "__main__":
    main()
