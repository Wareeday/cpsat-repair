import json

import pytest

from cpsr.generate import build_instances
from cpsr.llm import MockBackend
from cpsr.pipeline import CONDITIONS, Experiment, Settings


@pytest.fixture(scope="module")
def instances():
    rows = build_instances(1, ["knapsack_conflicts", "tsp"], 20, verbose=False)
    return [r for r in rows if r["size_idx"] <= 2]


def run(tmp_path, instances, mode, samples=4):
    exp = Experiment(MockBackend(mode), tmp_path, Settings(samples=samples, timeout=30))
    exp.run(instances)
    return [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]


def test_oracle_is_always_correct(tmp_path, instances):
    rows = run(tmp_path, instances, "oracle", samples=1)
    assert len(rows) == len(instances) * len(CONDITIONS)
    assert all(r["status"] == "ok" for r in rows)
    assert all(r["rounds"] == 0 for r in rows)


def test_repair_only_helps_and_is_paired(tmp_path, instances):
    rows = run(tmp_path, instances, "flaky")
    key = lambda r: (r["id"], r["sample"])  # noqa: E731
    one = {key(r): r for r in rows if r["condition"] == "cpsat"}
    rep = {key(r): r for r in rows if r["condition"] == "cpsat_repair"}
    assert one.keys() == rep.keys()
    repaired = 0
    for k in one:
        assert rep[k]["trace"][0] == one[k]["status"]            # same round-0 program
        if one[k]["status"] == "ok":
            assert rep[k]["rounds"] == 0
        elif rep[k]["status"] == "ok":
            repaired += 1
            assert 1 <= rep[k]["rounds"] <= 3
    assert repaired > 0, "the mock should have produced at least one successful repair"
    assert any(r["status"] != "ok" for r in one.values())


def test_bad_model_stops_after_three_rounds(tmp_path, instances):
    rows = run(tmp_path, instances, "bad", samples=1)
    rep = [r for r in rows if r["condition"] == "cpsat_repair"]
    assert all(r["rounds"] == 3 and len(r["trace"]) == 4 for r in rep if r["status"] != "ok")


def test_resume_does_not_duplicate(tmp_path, instances):
    first = run(tmp_path, instances, "flaky", samples=2)
    second = run(tmp_path, instances, "flaky", samples=2)
    assert len(first) == len(second)
