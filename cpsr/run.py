"""Command line entry point.

Local model with Ollama:
    python -m cpsr.run --backend ollama --model qwen2.5-coder:7b --out results/qwen7b

Any OpenAI-compatible API (key in an environment variable):
    export OPENAI_API_KEY=...
    python -m cpsr.run --backend openai --base-url https://api.groq.com/openai/v1 \
        --model llama-3.3-70b-versatile --sleep 2 --out results/llama70b

Pipeline test without any model:
    python -m cpsr.run --backend mock --mock-mode flaky --out results/mock_test
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import ortools

from . import __version__
from .generate import DEFAULT_PATH, load_instances
from .llm import make_backend
from .pipeline import Experiment, Settings


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the three conditions on all instances.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--backend", choices=["ollama", "openai", "mock"], required=True)
    ap.add_argument("--model", help="model name as the backend knows it")
    ap.add_argument("--base-url", help="Ollama host or OpenAI-compatible base URL")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--mock-mode", choices=["oracle", "flaky", "bad"], default="flaky")
    ap.add_argument("--out", type=Path, required=True, help="output directory (resumable)")
    ap.add_argument("--instances", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--families", help="comma-separated subset of families")
    ap.add_argument("--max-size", type=int, default=5, help="only use size levels 1..N")
    ap.add_argument("--samples", type=int, default=3, help="independent programs per family and condition")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repair-rounds", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=30.0, help="wall-clock limit per program run (s)")
    ap.add_argument("--mem-mb", type=int, default=4096, help="memory cap per run, POSIX only (0 = none)")
    ap.add_argument("--sleep", type=float, default=0.0, help="pause after each model call (rate limits)")
    ap.add_argument("--reveal-optimum", action="store_true",
                    help="tell the model the optimal value when its answer is suboptimal (off by default)")
    args = ap.parse_args()

    instances = load_instances(args.instances)
    if args.families:
        keep = set(args.families.split(","))
        instances = [i for i in instances if i["family"] in keep]
    instances = [i for i in instances if i["size_idx"] <= args.max_size]
    if not instances:
        sys.exit("no instances selected")

    backend = make_backend(args.backend, args.model, args.base_url, args.api_key_env, args.mock_mode)
    settings = Settings(args.samples, args.temperature, args.repair_rounds, args.timeout,
                        args.mem_mb or None, args.sleep, args.reveal_optimum)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({
        "backend": backend.id, "settings": vars(settings), "n_instances": len(instances),
        "python": platform.python_version(), "ortools": ortools.__version__, "cpsr": __version__,
    }, indent=2))
    Experiment(backend, args.out, settings).run(instances)
    print(f"finished. Next: python -m cpsr.report --results {args.out}")


if __name__ == "__main__":
    main()
