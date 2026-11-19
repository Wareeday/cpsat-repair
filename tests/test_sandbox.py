from cpsr.prompts import extract_code
from cpsr.sandbox import run_program


def test_ok_and_numpy_free_json():
    r = run_program("def solve(i):\n    return {'x': i['a'] + 1}", {"a": 1})
    assert r.status == "ok" and r.output == {"x": 2}


def test_timeout_is_enforced():
    r = run_program("def solve(i):\n    while True:\n        pass", {}, timeout=1.5)
    assert r.status == "timeout" and r.seconds < 5


def test_errors_are_reported_without_runner_frames():
    r = run_program("def solve(i):\n    return 1 / 0", {})
    assert r.status == "runtime_error" and "ZeroDivisionError" in r.error and "_runner" not in r.error
    assert run_program("def solve(", {}).status == "runtime_error"
    assert "solve" in run_program("x = 1", {}).error


def test_ortools_runs_inside_memory_cap():
    code = (
        "from ortools.sat.python import cp_model\n"
        "def solve(i):\n"
        "    m = cp_model.CpModel(); x = m.NewIntVar(0, 9, 'x'); m.Maximize(x)\n"
        "    s = cp_model.CpSolver(); s.Solve(m)\n"
        "    return {'x': s.Value(x)}\n"
    )
    assert run_program(code, {}, timeout=30, mem_mb=4096).output == {"x": 9}


def test_extract_code():
    reply = "Here you go\n```python\nimport os\ndef solve(instance):\n    return {}\n```\nbye"
    assert "def solve" in extract_code(reply)
    assert extract_code("no code here") is None
    assert extract_code("```python\nprint(1)\n```") is None
    assert extract_code("def solve(instance):\n    return {}") is not None


def test_extract_code_takes_last_block_and_ignores_reasoning():
    draft = "```python\ndef solve(instance):\n    return {'v': 'draft, and it is the longer block'}\n```"
    final = "```python\ndef solve(instance):\n    return {'v': 1}\n```"
    assert "'v': 1" in extract_code(f"{draft}\nOn second thought:\n{final}")
    thinking = f"<think>\n{draft}\n</think>\n{final}"
    assert "'v': 1" in extract_code(thinking)
    assert extract_code(f"<think>\n{draft}\n(cut off by the token limit)") is None


def test_extract_code_accepts_other_fence_labels():
    assert extract_code("```python3\ndef solve(instance):\n    return {}\n```") is not None
    assert extract_code("```\ndef solve(instance):\n    return {}\n```") is not None
