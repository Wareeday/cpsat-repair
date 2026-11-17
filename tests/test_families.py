import itertools
import random

import pytest

from cpsr.families import all_families, get_family
from cpsr.verdict import FORMAT_ERROR, INVALID, OK, SUBOPTIMAL

FAMILIES = [f.name for f in all_families()]


def make(name, size_idx=0, seed=0):
    fam = get_family(name)
    data = fam.generate(fam.sizes[size_idx], random.Random(seed))
    sol, obj = fam.reference(data)
    return fam, data, sol, obj


@pytest.mark.parametrize("name", FAMILIES)
def test_five_sizes_and_valid_example(name):
    fam = get_family(name)
    assert len(fam.sizes) == 5
    # the example in the prompt must itself be a valid answer
    ref = None
    if fam.sense:
        ref = fam.evaluate(fam.example_input, fam.example_output)
    assert fam.check(fam.example_input, fam.example_output, ref).ok


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("size_idx", [0, 1])
def test_reference_passes_checker(name, size_idx):
    fam, data, sol, obj = make(name, size_idx)
    verdict = fam.check(data, sol, obj)
    assert verdict.status == OK, verdict.message


@pytest.mark.parametrize("name", FAMILIES)
def test_checker_is_independent_of_ortools(name):
    import importlib
    import inspect
    src = inspect.getsource(importlib.import_module(type(get_family(name)).__module__))
    # the checker lives in `evaluate`; make sure it does not call into the solver
    body = src.split("def evaluate")[1]
    assert "cp_model" not in body and "solve_reference" not in body


@pytest.mark.parametrize("name", FAMILIES)
def test_garbage_output_is_format_error(name):
    fam, data, sol, obj = make(name)
    assert fam.check(data, "nope", obj).status == FORMAT_ERROR
    assert fam.check(data, {}, obj).status == FORMAT_ERROR
    key = next(iter(sol))
    assert fam.check(data, {key: "x"}, obj).status == FORMAT_ERROR


def test_nqueens_counterexamples():
    fam, data, sol, _ = make("nqueens_blocked")
    n = data["n"]
    same_col = {"queens": [0] * n}
    v = fam.check(data, same_col, None)
    assert v.status == INVALID and "column" in v.message or "blocked" in v.message
    # a queen on a blocked square is reported as such
    r, c = data["blocked"][0]
    q = list(sol["queens"])
    q[r] = c
    assert "blocked" in fam.check(data, {"queens": q}, None).message


def test_latin_prefilled_and_duplicates():
    fam, data, sol, _ = make("latin_completion")
    g = [row[:] for row in sol["grid"]]
    r, c, v = data["prefilled"][0]
    g[r][c] = v % data["n"] + 1
    assert fam.check(data, {"grid": g}, None).status == INVALID
    g2 = [row[:] for row in sol["grid"]]
    g2[0][0] = g2[0][1]
    assert "twice" in fam.check(data, {"grid": g2}, None).message or "pre-filled" in fam.check(data, {"grid": g2}, None).message


def test_suboptimal_detected_for_every_optimisation_family():
    fam, data, sol, obj = make("knapsack_conflicts")
    assert fam.check(data, {"selected": []}, obj).status == SUBOPTIMAL
    fam, data, sol, obj = make("graph_coloring")
    n = data["n"]
    assert fam.check(data, {"colors": list(range(n))}, obj).status in (SUBOPTIMAL, OK)
    fam, data, sol, obj = make("bin_packing")
    assert fam.check(data, {"assignment": list(range(len(data["sizes"])))}, obj).status == SUBOPTIMAL
    fam, data, sol, obj = make("tsp")
    worst = max(itertools.islice(itertools.permutations(range(data["n"])), 200),
                key=lambda t: fam.evaluate(data, {"tour": list(t)}))
    assert fam.check(data, {"tour": list(worst)}, obj).status == SUBOPTIMAL
    fam, data, sol, obj = make("job_shop")
    seq, t = [], 0
    for job in data["jobs"]:       # run jobs one after another: valid, but slow
        row = []
        for _, d in job:
            row.append(t)
            t += d
        seq.append(row)
    assert fam.check(data, {"starts": seq}, obj).status == SUBOPTIMAL


def test_job_shop_violations():
    fam, data, sol, obj = make("job_shop")
    zeros = [[0] * len(j) for j in data["jobs"]]
    v = fam.check(data, {"starts": zeros}, obj)
    assert v.status == INVALID and ("overlap" in v.message or "finishes" in v.message)


def test_shift_violations():
    fam, data, sol, obj = make("shift_scheduling", 1)
    E, D = data["n_employees"], data["n_days"]
    empty = {"roster": [[-1] * D for _ in range(E)]}
    assert fam.check(data, empty, obj).status == INVALID
    bad = [row[:] for row in sol["roster"]]
    bad[0][0], bad[0][1] = 2, 0
    assert fam.check(data, {"roster": bad}, obj).status == INVALID


# ---- reference optima against brute force on tiny instances ---------------------------------------

def test_tsp_reference_matches_brute_force():
    fam = get_family("tsp")
    for seed in range(3):
        data = fam.generate({"n": 7}, random.Random(seed))
        best = min(fam.evaluate(data, {"tour": [0, *p]}) for p in itertools.permutations(range(1, 7)))
        assert fam.reference(data)[1] == best


def test_knapsack_reference_matches_brute_force():
    fam = get_family("knapsack_conflicts")
    for seed in range(3):
        data = fam.generate({"n": 12}, random.Random(seed))
        best = 0
        for mask in range(1 << 12):
            sel = [i for i in range(12) if mask >> i & 1]
            try:
                best = max(best, fam.evaluate(data, {"selected": sel}))
            except Exception:
                pass
        assert fam.reference(data)[1] == best


def test_graph_coloring_reference_matches_brute_force():
    fam = get_family("graph_coloring")
    for seed in range(3):
        data = fam.generate({"n": 7, "p": 0.5}, random.Random(seed))
        n = data["n"]
        best = next(
            k for k in range(1, n + 1)
            if any(all(c[u] != c[v] for u, v in data["edges"]) for c in itertools.product(range(k), repeat=n))
        )
        assert fam.reference(data)[1] == best


def test_bin_packing_reference_matches_brute_force():
    fam = get_family("bin_packing")
    for seed in range(3):
        data = fam.generate({"n": 7}, random.Random(seed))
        n, best = 7, 7
        for assign in itertools.product(range(n), repeat=n):
            if assign[0] != 0:
                continue
            try:
                best = min(best, fam.evaluate(data, {"assignment": list(assign)}))
            except Exception:
                pass
        assert fam.reference(data)[1] == best
