"""Turn results/<run>/runs.jsonl into tables, figures and (optionally) the Results section of the README.

    python -m cpsr.report --results results/qwen7b
    python -m cpsr.report --results results/qwen7b --update-readme README.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .pipeline import CONDITIONS  # noqa: E402
from .verdict import ALL_STATUSES  # noqa: E402

LABELS = {"python": "Direct Python", "cpsat": "CP-SAT (one shot)", "cpsat_repair": "CP-SAT + repair"}
COLORS = {"python": "#c0504d", "cpsat": "#4f81bd", "cpsat_repair": "#2e8b57"}
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def pct(k: int, n: int, ci: bool = True) -> str:
    if n == 0:
        return "-"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]" if ci else f"{100 * k / n:.0f}%"


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load(results: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in (results / "runs.jsonl").read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["correct"] = df["status"] == "ok"
    df["valid"] = df["status"].isin(["ok", "suboptimal"])
    return df


def cond_df(df: pd.DataFrame, c: str) -> pd.DataFrame:
    return df[df["condition"] == c]


def overall_table(df: pd.DataFrame) -> str:
    rows = []
    for c in CONDITIONS:
        d = cond_df(df, c)
        rows.append([LABELS[c], len(d), pct(int(d["valid"].sum()), len(d), False), pct(int(d["correct"].sum()), len(d))])
    return md_table(["Condition", "Runs", "Valid answer", "Correct (valid and optimal) [95% CI]"], rows)


def by_family_table(df: pd.DataFrame) -> str:
    rows = []
    for fam, g in df.groupby("family", sort=False):
        row = [fam]
        for c in CONDITIONS:
            d = cond_df(g, c)
            row.append(pct(int(d["correct"].sum()), len(d), False))
        rows.append(row)
    return md_table(["Family"] + [LABELS[c] for c in CONDITIONS], rows)


def by_size_table(df: pd.DataFrame) -> str:
    rows = []
    for s, g in df.groupby("size_idx"):
        row = [f"size {s}"]
        for c in CONDITIONS:
            d = cond_df(g, c)
            row.append(pct(int(d["correct"].sum()), len(d), False))
        rows.append(row)
    return md_table(["Size level (1 = smallest, 5 = largest)"] + [LABELS[c] for c in CONDITIONS], rows)


def failure_table(df: pd.DataFrame) -> str:
    rows = []
    for st in ALL_STATUSES:
        row = [st]
        for c in CONDITIONS:
            d = cond_df(df, c)
            row.append(f"{100 * (d['status'] == st).mean():.0f}%" if len(d) else "-")
        rows.append(row)
    return md_table(["Outcome"] + [LABELS[c] for c in CONDITIONS], rows)


def repair_curve(df: pd.DataFrame) -> tuple[list[int], list[int], int]:
    d = cond_df(df, "cpsat_repair")
    max_r = int(d["rounds"].max()) if len(d) else 0
    max_r = max(max_r, 3)
    ks = [int(sum("ok" in t[: r + 1] for t in d["trace"])) for r in range(max_r + 1)]
    return list(range(max_r + 1)), ks, len(d)


def repair_table(df: pd.DataFrame) -> str:
    rounds, ks, n = repair_curve(df)
    rows = [[f"after {r} repair round(s)" if r else "round 0 (no repair)", pct(k, n)] for r, k in zip(rounds, ks)]
    return md_table(["CP-SAT + repair", "Correct (cumulative) [95% CI]"], rows)


def make_figures(df: pd.DataFrame, out: Path) -> None:
    # 1. correctness by size level
    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = sorted(df["size_idx"].unique())
    for c in CONDITIONS:
        d = cond_df(df, c)
        ys, lo, hi = [], [], []
        for s in sizes:
            g = d[d["size_idx"] == s]
            k, n = int(g["correct"].sum()), len(g)
            ys.append(100 * k / max(n, 1)), lo.append(100 * wilson(k, n)[0]), hi.append(100 * wilson(k, n)[1])
        ax.plot(sizes, ys, marker="o", label=LABELS[c], color=COLORS[c])
        ax.fill_between(sizes, lo, hi, alpha=0.12, color=COLORS[c])
    ax.set_xlabel("size level (1 = smallest)"), ax.set_ylabel("correct answers (%)")
    ax.set_xticks(sizes), ax.set_ylim(-3, 103), ax.grid(alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(), fig.savefig(out / "correct_by_size.png", dpi=150), plt.close(fig)

    # 2. per family
    fams = list(dict.fromkeys(df["family"]))
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.27
    for i, c in enumerate(CONDITIONS):
        d = cond_df(df, c)
        vals = [100 * d[d["family"] == f]["correct"].mean() if len(d[d["family"] == f]) else 0 for f in fams]
        ax.bar([x + (i - 1) * w for x in range(len(fams))], vals, w, label=LABELS[c], color=COLORS[c])
    ax.set_xticks(range(len(fams))), ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.set_ylabel("correct answers (%)"), ax.set_ylim(0, 105), ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(), fig.savefig(out / "correct_by_family.png", dpi=150), plt.close(fig)

    # 3. repair curve
    rounds, ks, n = repair_curve(df)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(rounds, [100 * k / max(n, 1) for k in ks], marker="o", color=COLORS["cpsat_repair"])
    ax.set_xlabel("counterexample repair rounds"), ax.set_ylabel("correct answers (%)")
    ax.set_xticks(rounds), ax.set_ylim(0, 103), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(out / "repair_curve.png", dpi=150), plt.close(fig)

    # 4. failure modes
    fig, ax = plt.subplots(figsize=(7, 3.8))
    left = [0.0] * len(CONDITIONS)
    cmap = plt.get_cmap("tab10")
    for j, st in enumerate(ALL_STATUSES):
        vals = [100 * (cond_df(df, c)["status"] == st).mean() if len(cond_df(df, c)) else 0 for c in CONDITIONS]
        ax.barh([LABELS[c] for c in CONDITIONS], vals, left=left, label=st, color=cmap(j))
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel("share of runs (%)"), ax.set_xlim(0, 100)
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(), fig.savefig(out / "failure_modes.png", dpi=150), plt.close(fig)


def export_failures(df: pd.DataFrame, out: Path) -> int:
    """First failing instance per (family, sample, status) of the one-shot CP-SAT programs, to annotate by hand."""
    d = cond_df(df, "cpsat")
    d = d[~d["correct"]].drop_duplicates(["family", "sample", "status"])
    table = pd.DataFrame({
        "family": d["family"], "sample": d["sample"], "instance": d["id"], "status": d["status"],
        "message": d["message"].str.replace("\n", " | "),
        "program": [f"programs/{f}/s{s}_cpsat.py" for f, s in zip(d["family"], d["sample"])],
        "my_category": "",     # e.g. missing constraint / wrong objective / indexing / API misuse / format
    })
    table.to_csv(out / "failures_for_annotation.csv", index=False)
    return len(table)


def build_summary(df: pd.DataFrame, cfg: dict) -> str:
    n_inst, n_fam, n_sizes = df["id"].nunique(), df["family"].nunique(), df["size_idx"].nunique()
    lines = [
        f"Model: `{cfg.get('backend', 'unknown')}`. {n_inst} instances ({n_fam} families, {n_sizes} size levels), "
        f"{cfg.get('settings', {}).get('samples', '?')} independent programs per family and condition, "
        f"up to {cfg.get('settings', {}).get('repair_rounds', '?')} repair rounds, "
        f"{cfg.get('settings', {}).get('timeout', '?')} s limit per run. Total: {len(df)} runs.",
        "", "**Overall**", "", overall_table(df),
        "", "**By repair round**", "", repair_table(df),
        "", "**By problem family** (correct answers)", "", by_family_table(df),
        "", "**By size level** (correct answers)", "", by_size_table(df),
        "", "**Outcome of every run**", "", failure_table(df),
        "", "![correct by size](results/correct_by_size.png)",
        "", "![correct by family](results/correct_by_family.png)",
        "", "![repair curve](results/repair_curve.png)",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--update-readme", type=Path, help="replace the block between the RESULTS markers")
    args = ap.parse_args()

    df = load(args.results)
    cfg = json.loads((args.results / "config.json").read_text()) if (args.results / "config.json").exists() else {}
    make_figures(df, args.results)
    n_fail = export_failures(df, args.results)
    summary = build_summary(df, cfg)
    (args.results / "summary.md").write_text(summary + "\n")
    print(summary)
    print(f"\nfigures and summary.md written to {args.results}; {n_fail} failures exported for annotation")

    if args.update_readme:
        text = args.update_readme.read_text()
        if START not in text or END not in text:
            raise SystemExit("README has no RESULTS markers")
        rel = args.results.as_posix().rstrip("/")
        block = summary.replace("](results/", f"]({rel}/")
        head, rest = text.split(START)
        _, tail = rest.split(END)
        args.update_readme.write_text(f"{head}{START}\n{block}\n{END}{tail}")
        print(f"README updated: {args.update_readme}")


if __name__ == "__main__":
    main()
