"""Run the real command line against a fake OpenAI-style server that answers like a model would."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

CPSAT_TSP = '''```python
from ortools.sat.python import cp_model

def solve(instance):
    n, d = instance["n"], instance["dist"]
    model = cp_model.CpModel()
    lits = {(i, j): model.NewBoolVar(f"a{i}_{j}") for i in range(n) for j in range(n) if i != j}
    model.AddCircuit([(i, j, v) for (i, j), v in lits.items()])
    model.Minimize(sum(d[i][j] * v for (i, j), v in lits.items()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20
    solver.Solve(model)
    nxt = {i: j for (i, j), v in lits.items() if solver.Value(v)}
    tour, cur = [0], nxt[0]
    while cur != 0:
        tour.append(cur)
        cur = nxt[cur]
    return {"tour": tour}
```'''

BRUTE_TSP = '''Here is a brute-force solution.
```python
import itertools

def solve(instance):
    n, d = instance["n"], instance["dist"]
    best, best_tour = None, None
    for p in itertools.permutations(range(1, n)):
        t = [0, *p]
        c = sum(d[t[i]][t[(i + 1) % n]] for i in range(n))
        if best is None or c < best:
            best, best_tour = c, t
    return {"tour": best_tour}
```'''

state = {"status": 200, "calls": 0}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        state["calls"] += 1
        if state["status"] != 200:
            payload, code = {"error": "nope"}, state["status"]
        else:
            prompt = body["messages"][-1]["content"]
            reply = CPSAT_TSP if "cp_model" in prompt else BRUTE_TSP
            payload, code = {"choices": [{"message": {"content": reply}}]}, 200
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def url():
    state.update(status=200, calls=0)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def cli(*args, env_key="secret"):
    import os
    env = dict(os.environ, OPENAI_API_KEY=env_key)
    return subprocess.run([sys.executable, "-m", "cpsr.run", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=240)


def test_full_run_with_openai_style_backend_and_resume(url, tmp_path):
    out = tmp_path / "run"
    args = ["--backend", "openai", "--base-url", url, "--model", "fake", "--families", "tsp",
            "--max-size", "2", "--samples", "1", "--out", str(out)]
    first = cli(*args)
    assert first.returncode == 0, first.stderr
    rows = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 4 * 3                                          # 2 sizes x 2 instances x 3 conditions
    assert all(r["status"] == "ok" for r in rows), [(r["condition"], r["status"], r["message"]) for r in rows]
    assert state["calls"] == 2                                         # one python + one cpsat program only
    assert (out / "programs" / "tsp" / "s0_cpsat.py").exists()
    assert json.loads((out / "config.json").read_text())["backend"] == "openai:fake"
    # a second invocation must not call the model or add rows
    calls = state["calls"]
    assert cli(*args).returncode == 0
    assert state["calls"] == calls
    assert len((out / "runs.jsonl").read_text().splitlines()) == len(rows)
    # the report works on this run
    rep = subprocess.run([sys.executable, "-m", "cpsr.report", "--results", str(out)], cwd=ROOT,
                         capture_output=True, text=True, timeout=120)
    assert rep.returncode == 0, rep.stderr
    assert (out / "summary.md").exists() and (out / "repair_curve.png").exists()


def test_provider_error_stops_cleanly(url, tmp_path):
    state["status"] = 401
    res = cli("--backend", "openai", "--base-url", url, "--model", "fake", "--families", "tsp",
              "--max-size", "1", "--samples", "1", "--out", str(tmp_path / "run"))
    assert res.returncode != 0
    assert "stopped:" in res.stderr and "401" in res.stderr
    assert "Traceback" not in res.stderr


def test_ping(url):
    import os
    env = dict(os.environ, OPENAI_API_KEY="k")
    res = subprocess.run([sys.executable, "-m", "cpsr.ping", "--backend", "openai", "--base-url", url,
                          "--model", "fake"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0 and "reply:" in res.stdout
