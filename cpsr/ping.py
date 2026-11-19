"""Check your key, base URL and model name before a long run.

    python -m cpsr.ping --backend openai --base-url https://api.groq.com/openai/v1 --model openai/gpt-oss-120b
    python -m cpsr.ping --backend openai --base-url https://api.groq.com/openai/v1 --list-models
"""
from __future__ import annotations

import argparse
import time

from .llm import BackendError, make_backend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["ollama", "openai"], required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--list-models", action="store_true", help="print the model ids the provider offers")
    ap.add_argument("--no-seed", action="store_true")
    args = ap.parse_args()
    backend = make_backend(args.backend, args.model or "list-only", args.base_url, args.api_key_env, "oracle",
                           max_tokens=64, send_seed=not args.no_seed)
    try:
        if args.list_models:
            print("\n".join(backend.list_models()))
            return
        t0 = time.time()
        reply = backend.chat([{"role": "user", "content": "Reply with the single word OK."}], temperature=0.0, seed=0)
        print(f"reply: {reply.strip()[:200]!r}  ({time.time() - t0:.1f}s)")
    except BackendError as e:
        raise SystemExit(f"FAILED: {e}")


if __name__ == "__main__":
    main()
