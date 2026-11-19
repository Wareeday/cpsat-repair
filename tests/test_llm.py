"""The HTTP layer, tested against a tiny local server that imitates the two API styles."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cpsr.llm import BackendError, LLMCache, OllamaBackend, OpenAICompatBackend

MESSAGES = [{"role": "user", "content": "hi"}]


class Handler(BaseHTTPRequestHandler):
    seen: list = []
    fail_first = 0            # number of 429 answers before the first success
    always = None             # force this status for every request

    def log_message(self, *args):
        pass

    def _send(self, code, body, headers=None):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        Handler.seen.append({"path": self.path, "headers": dict(self.headers)})
        if self.path.endswith("/models"):
            self._send(200, {"data": [{"id": "b-model"}, {"id": "a-model"}]})
        else:
            self._send(200, {"models": [{"name": "qwen:7b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if Handler.always:
            return self._send(Handler.always, {"error": "forced"})
        if Handler.fail_first > 0:
            Handler.fail_first -= 1
            return self._send(429, {"error": "slow down"}, {"Retry-After": "1"})
        if self.path.endswith("/chat/completions"):
            self._send(200, {"choices": [{"message": {"content": "hello from openai-style"}}]})
        else:
            self._send(200, {"message": {"content": "hello from ollama-style"}})


@pytest.fixture()
def server():
    Handler.seen, Handler.fail_first, Handler.always = [], 0, None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_openai_style_request_and_reply(server):
    b = OpenAICompatBackend("m", server + "/v1", "KEY", max_tokens=123, extra_body={"reasoning_effort": "low"})
    assert b.chat(MESSAGES, temperature=0.7, seed=5) == "hello from openai-style"
    req = Handler.seen[-1]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer KEY"
    assert req["headers"]["User-Agent"].startswith("cpsr")            # not Python-urllib
    assert req["body"] == {"model": "m", "messages": MESSAGES, "temperature": 0.7,
                           "reasoning_effort": "low", "max_tokens": 123, "seed": 5}


def test_seed_and_max_tokens_can_be_left_out(server):
    OpenAICompatBackend("m", server, "K", max_tokens=0, send_seed=False).chat(MESSAGES, temperature=0, seed=1)
    assert "seed" not in Handler.seen[-1]["body"] and "max_tokens" not in Handler.seen[-1]["body"]


def test_retry_after_is_honoured(server):
    Handler.fail_first = 2
    b = OpenAICompatBackend("m", server, "K")
    assert b.chat(MESSAGES, temperature=0, seed=0) == "hello from openai-style"
    assert len(Handler.seen) == 3                                     # two 429s, then success


def test_persistent_error_becomes_backend_error(server):
    Handler.always = 401
    with pytest.raises(BackendError, match="401"):
        OpenAICompatBackend("m", server, "bad").chat(MESSAGES, temperature=0, seed=0)
    assert len(Handler.seen) == 1                                     # 401 is not retried


def test_ollama_request_and_reply(server):
    b = OllamaBackend("qwen:7b", server, max_tokens=77)
    assert b.chat(MESSAGES, temperature=0.3, seed=9) == "hello from ollama-style"
    body = Handler.seen[-1]["body"]
    assert Handler.seen[-1]["path"] == "/api/chat" and body["stream"] is False
    assert body["options"] == {"temperature": 0.3, "seed": 9, "num_ctx": 8192, "num_predict": 77}


def test_list_models(server):
    assert OpenAICompatBackend("m", server, "K").list_models() == ["a-model", "b-model"]
    assert OllamaBackend("m", server).list_models() == ["qwen:7b"]


def test_cache_roundtrip(tmp_path):
    cache = LLMCache(tmp_path / "c.jsonl")
    k = LLMCache.key("x", MESSAGES, 0.7, 1)
    assert cache.get(k) is None
    cache.put(k, "reply")
    assert LLMCache(tmp_path / "c.jsonl").get(k) == "reply"
    assert LLMCache.key("x", MESSAGES, 0.7, 2) != k                   # a different seed is a different sample
