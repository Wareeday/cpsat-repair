"""LLM backends. All of them take chat messages and return the reply text.

  ollama  - local models served by Ollama (http://localhost:11434)
  openai  - any OpenAI-compatible endpoint (Groq, OpenRouter, Together, Gemini's compat API, vLLM, ...)
  mock    - no model at all; used for testing the pipeline
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


class Backend:
    id = "base"

    def chat(self, messages: list[dict], *, temperature: float, seed: int, meta: Optional[dict] = None) -> str:
        raise NotImplementedError


def _post_json(url: str, payload: dict, headers: dict, timeout: float, retries: int = 6) -> dict:
    body = json.dumps(payload).encode()
    delay = 5.0
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            retryable = e.code in (429, 500, 502, 503, 504)
            if not retryable or attempt == retries - 1:
                raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode(errors='replace')[:300]}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"cannot reach {url}: {e}") from e
        time.sleep(delay)
        delay = min(delay * 2, 90)
    raise RuntimeError("unreachable")


class OllamaBackend(Backend):
    def __init__(self, model: str, host: str = "http://localhost:11434", num_ctx: int = 8192, timeout: float = 900):
        self.model, self.host, self.num_ctx, self.timeout = model, host.rstrip("/"), num_ctx, timeout
        self.id = f"ollama:{model}"

    def chat(self, messages, *, temperature, seed, meta=None) -> str:
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_ctx": self.num_ctx},
        }
        return _post_json(f"{self.host}/api/chat", payload, {}, self.timeout)["message"]["content"]


class OpenAICompatBackend(Backend):
    def __init__(self, model: str, base_url: str, api_key: str, timeout: float = 300, send_seed: bool = True):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.timeout, self.send_seed = timeout, send_seed
        self.id = f"openai:{model}"

    def chat(self, messages, *, temperature, seed, meta=None) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if self.send_seed:
            payload["seed"] = seed
        data = _post_json(f"{self.base_url}/chat/completions", payload,
                          {"Authorization": f"Bearer {self.api_key}"}, self.timeout)
        return data["choices"][0]["message"]["content"]


_MOCK_PROGRAM = '''\
from cpsr.families import get_family


def _corrupt(obj):
    """Damage a solution: copy the second entry of the first list onto the first entry."""
    if isinstance(obj, dict):
        key = next(iter(obj))
        obj[key] = _corrupt(obj[key])
    elif isinstance(obj, list) and obj:
        if isinstance(obj[0], list):
            obj[0] = _corrupt(obj[0])
        elif len(obj) > 1:
            obj[0] = obj[1]
    return obj


def solve(instance):
    solution, _ = get_family("{family}").reference(instance)
    return _corrupt(solution) if {corrupt} else solution
'''


class MockBackend(Backend):
    """Deterministic stand-in for a model, for testing the plumbing only.

    oracle: always returns a correct program.
    flaky : some samples are wrong at round 0 and get fixed after one or two counterexamples.
    bad   : always wrong.
    The numbers produced with it say nothing about any language model.
    """

    def __init__(self, mode: str = "flaky"):
        assert mode in ("oracle", "flaky", "bad")
        self.mode = mode
        self.id = f"mock:{mode}"

    def chat(self, messages, *, temperature, seed, meta=None) -> str:
        meta = meta or {}
        sample, rnd, kind = meta.get("sample", 0), meta.get("round", 0), meta.get("kind", "cpsat")
        if self.mode == "oracle":
            corrupt = False
        elif self.mode == "bad":
            corrupt = True
        elif kind == "python":
            corrupt = sample % 3 != 0
        else:
            fix_round = 1 if sample % 4 == 1 else 2
            corrupt = sample % 2 == 1 and rnd < fix_round
        code = _MOCK_PROGRAM.format(family=meta["family"], corrupt=corrupt)
        return f"```python\n{code}```"


class LLMCache:
    """Append-only JSONL cache so an interrupted run can resume without repeating model calls."""

    def __init__(self, path: Path):
        self.path = path
        self.store: dict[str, str] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.store[row["key"]] = row["reply"]

    @staticmethod
    def key(backend_id: str, messages: list[dict], temperature: float, seed: int) -> str:
        blob = json.dumps([backend_id, messages, temperature, seed], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self.store.get(key)

    def put(self, key: str, reply: str) -> None:
        self.store[key] = reply
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": key, "reply": reply}) + "\n")


def make_backend(name: str, model: Optional[str], base_url: Optional[str], api_key_env: str, mock_mode: str) -> Backend:
    if name == "mock":
        return MockBackend(mock_mode)
    if not model:
        raise SystemExit("--model is required for this backend")
    if name == "ollama":
        return OllamaBackend(model, base_url or "http://localhost:11434")
    if name == "openai":
        key = os.environ.get(api_key_env, "")
        if not key:
            raise SystemExit(f"set the environment variable {api_key_env} with your API key")
        if not base_url:
            raise SystemExit("--base-url is required for the openai backend (e.g. https://api.groq.com/openai/v1)")
        return OpenAICompatBackend(model, base_url, key)
    raise SystemExit(f"unknown backend {name}")
