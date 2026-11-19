"""LLM backends. All of them take chat messages and return the reply text.

  ollama  - local models served by Ollama (http://localhost:11434)
  openai  - any OpenAI-compatible endpoint (Groq, Mistral, OpenRouter, Gemini's compat API, vLLM, ...)
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

USER_AGENT = "cpsr/0.1 (research script)"     # some providers sit behind Cloudflare and block urllib's default


class BackendError(RuntimeError):
    """The provider could not be reached or kept refusing (rate limit, quota, bad key, bad model name)."""


class Backend:
    id = "base"

    def chat(self, messages: list[dict], *, temperature: float, seed: int, meta: Optional[dict] = None) -> str:
        raise NotImplementedError


def _retry_after(headers) -> Optional[float]:
    value = headers.get("Retry-After") if headers else None
    try:
        return min(float(value), 120.0) if value is not None else None
    except ValueError:
        return None


def _request_json(req: urllib.request.Request, timeout: float, retries: int = 6) -> dict:
    delay = 5.0
    for attempt in range(retries):
        wait = delay
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:400]
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                hint = " (rate limit or daily quota reached: re-run the same command later, it resumes)" if e.code == 429 else ""
                raise BackendError(f"HTTP {e.code} from {req.full_url}{hint}: {body}") from e
            wait = _retry_after(e.headers) or delay
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise BackendError(f"cannot reach {req.full_url}: {e}") from e
        time.sleep(wait)
        delay = min(delay * 2, 90)
    raise BackendError("unreachable")


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers},
    )
    return _request_json(req, timeout)


def _get_json(url: str, headers: dict, timeout: float = 60) -> dict:
    return _request_json(urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers}), timeout, retries=2)


class OllamaBackend(Backend):
    def __init__(self, model: str, host: str = "http://localhost:11434", num_ctx: int = 8192,
                 max_tokens: int = 4096, extra_body: Optional[dict] = None, timeout: float = 900):
        self.model, self.host, self.num_ctx = model, host.rstrip("/"), num_ctx
        self.max_tokens, self.extra_body, self.timeout = max_tokens, extra_body or {}, timeout
        self.id = f"ollama:{model}"

    def chat(self, messages, *, temperature, seed, meta=None) -> str:
        options = {"temperature": temperature, "seed": seed, "num_ctx": self.num_ctx}
        if self.max_tokens:
            options["num_predict"] = self.max_tokens
        payload = {"model": self.model, "messages": messages, "stream": False, "options": options, **self.extra_body}
        return _post_json(f"{self.host}/api/chat", payload, {}, self.timeout)["message"]["content"]

    def list_models(self) -> list[str]:
        return [m["name"] for m in _get_json(f"{self.host}/api/tags", {}).get("models", [])]


class OpenAICompatBackend(Backend):
    def __init__(self, model: str, base_url: str, api_key: str, max_tokens: int = 4096,
                 send_seed: bool = True, extra_body: Optional[dict] = None, timeout: float = 300):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.max_tokens, self.send_seed, self.extra_body, self.timeout = max_tokens, send_seed, extra_body or {}, timeout
        self.id = f"openai:{model}"

    @property
    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def chat(self, messages, *, temperature, seed, meta=None) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": temperature, **self.extra_body}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.send_seed:
            payload["seed"] = seed
        data = _post_json(f"{self.base_url}/chat/completions", payload, self._auth, self.timeout)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError(f"unexpected response format: {str(data)[:300]}") from e

    def list_models(self) -> list[str]:
        return sorted(m["id"] for m in _get_json(f"{self.base_url}/models", self._auth).get("data", []))


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


def make_backend(name: str, model: Optional[str], base_url: Optional[str], api_key_env: str, mock_mode: str,
                 max_tokens: int = 4096, send_seed: bool = True, extra_body: Optional[dict] = None) -> Backend:
    if name == "mock":
        return MockBackend(mock_mode)
    if not model:
        raise SystemExit("--model is required for this backend")
    if name == "ollama":
        return OllamaBackend(model, base_url or "http://localhost:11434", max_tokens=max_tokens, extra_body=extra_body)
    if name == "openai":
        key = os.environ.get(api_key_env, "")
        if not key:
            raise SystemExit(f"set the environment variable {api_key_env} with your API key")
        if not base_url:
            raise SystemExit("--base-url is required for the openai backend (e.g. https://api.groq.com/openai/v1)")
        return OpenAICompatBackend(model, base_url, key, max_tokens=max_tokens, send_seed=send_seed, extra_body=extra_body)
    raise SystemExit(f"unknown backend {name}")
