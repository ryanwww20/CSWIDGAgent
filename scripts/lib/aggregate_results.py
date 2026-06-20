#!/usr/bin/env python3
"""Aggregate ablation runs into a comparison table + plots.

Scans ``runs/<tag>/`` directories (each one run = condition x topic x seed),
reads ``meta.json`` + ``execution_result.json`` + ``judge_result.json``, and
writes:

- ``results/summary.csv``      — one row per run (machine-readable)
- ``results/summary.md``       — per-condition rollup (human-readable)
- ``results/overall_by_condition.png``  — mean overall judge score per condition
- ``results/criteria_by_condition.png`` — the 8 criteria per condition

Two executability rates are reported because environment/timeout failures are
NOT agent-quality failures (these notebooks target Colab's preinstalled stack):

- ``exec_rate_raw``     = ran_to_completion / all runs
- ``exec_rate_quality`` = ran_to_completion / (all runs excluding env+timeout)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CRITERIA = [
    "executability",
    "concept_correctness",
    "interactivity",
    "visualization_quality",
    "pedagogical_value",
    "alignment_with_source",
    "robustness",
    "simplicity_maintainability",
]
ENV_OR_TIMEOUT = {"environment_dependency", "timeout"}


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def collect_rows(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        meta = _load(run_dir / "meta.json")
        execr = _load(run_dir / "execution_result.json")
        judge = _load(run_dir / "judge_result.json")
        if not (meta or execr or judge):
            continue
        row: dict[str, Any] = {
            "run_tag": meta.get("run_tag", run_dir.name),
            "condition": meta.get("condition", "?"),
            "topic": meta.get("topic", "?"),
            "seed": meta.get("seed", "?"),
            "ran_to_completion": execr.get("ran_to_completion"),
            "error_type": execr.get("error_type"),
            "first_failed_code_cell": execr.get("first_failed_code_cell"),
            "mean_overall": judge.get("mean_overall"),
        }
        for c in CRITERIA:
            row[c] = (judge.get("mean_scores") or {}).get(c)
        rows.append(row)
    return rows


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.fmean(vals), 2) if vals else None


def rollup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)

    summary: dict[str, dict[str, Any]] = {}
    for cond, group in by_cond.items():
        n = len(group)
        ran = sum(1 for r in group if r["ran_to_completion"] is True)
        env_to = sum(1 for r in group if r["error_type"] in ENV_OR_TIMEOUT)
        quality_denom = n - env_to
        summary[cond] = {
            "n": n,
            "exec_rate_raw": round(ran / n, 2) if n else None,
            "exec_rate_quality": round(ran / quality_denom, 2) if quality_denom else None,
            "env_or_timeout": env_to,
            "mean_overall": _mean([r["mean_overall"] for r in group]),
            **{c: _mean([r[c] for r in group]) for c in CRITERIA},
        }
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_tag", "condition", "topic", "seed", "ran_to_completion",
              "error_type", "first_failed_code_cell", "mean_overall", *CRITERIA]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})


def write_md(summary: dict[str, dict[str, Any]], path: Path, n_rows: int) -> None:
    order = ["S0", "ablate-concept-extractor", "ablate-notebook-architect",
             "ablate-cell-analyzer", "B", "B+bug_solver"]
    conds = [c for c in order if c in summary] + [c for c in summary if c not in order]

    lines = [f"# Ablation summary ({n_rows} runs)", ""]
    lines.append("## Executability (objective harness)")
    lines.append("")
    lines.append("| Condition | n | exec_raw | exec_quality | env/timeout |")
    lines.append("|---|--:|--:|--:|--:|")
    for c in conds:
        s = summary[c]
        lines.append(f"| {c} | {s['n']} | {s['exec_rate_raw']} | {s['exec_rate_quality']} | {s['env_or_timeout']} |")
    lines += ["", "## Judge scores (1-5, mean across topics x seeds x judges)", ""]
    header = "| Condition | overall | " + " | ".join(c.replace("_", " ") for c in CRITERIA) + " |"
    lines.append(header)
    lines.append("|---|--:|" + "--:|" * len(CRITERIA))
    for c in conds:
        s = summary[c]
        cells = " | ".join(str(s[c2]) for c2 in CRITERIA)
        lines.append(f"| {c} | {s['mean_overall']} | {cells} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_overall(summary: dict[str, dict[str, Any]], path: Path) -> None:
    conds = list(summary.keys())
    vals = [summary[c]["mean_overall"] or 0 for c in conds]
    plt.figure(figsize=(max(6, len(conds) * 1.4), 4))
    plt.bar(conds, vals, color="#4C72B0")
    plt.ylabel("mean overall judge score (1-5)")
    plt.title("Notebook quality by ablation condition")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_criteria(summary: dict[str, dict[str, Any]], path: Path) -> None:
    conds = list(summary.keys())
    x = range(len(CRITERIA))
    plt.figure(figsize=(12, 5))
    width = 0.8 / max(1, len(conds))
    for i, c in enumerate(conds):
        offsets = [xi + i * width for xi in x]
        plt.bar(offsets, [summary[c][crit] or 0 for crit in CRITERIA], width=width, label=c)
    plt.xticks([xi + 0.4 for xi in x], [c.replace("_", "\n") for c in CRITERIA], fontsize=8)
    plt.ylabel("mean score (1-5)")
    plt.ylim(0, 5)
    plt.title("Per-criterion scores by ablation condition")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ablation runs")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not args.runs_dir.exists():
        raise SystemExit(f"runs dir not found: {args.runs_dir}")

    rows = collect_rows(args.runs_dir)
    if not rows:
        raise SystemExit(f"no runs with results found under {args.runs_dir}")
    summary = rollup(rows)

    write_csv(rows, args.out_dir / "summary.csv")
    write_md(summary, args.out_dir / "summary.md", len(rows))
    try:
        plot_overall(summary, args.out_dir / "overall_by_condition.png")
        plot_criteria(summary, args.out_dir / "criteria_by_condition.png")
    except Exception as exc:  # noqa: BLE001 — plots are a nice-to-have
        print(f"[aggregate] plotting skipped: {exc}")

    print(f"[aggregate] {len(rows)} runs across {len(summary)} conditions")
    print(f"[aggregate] wrote {args.out_dir}/summary.csv, summary.md, *.png")


if __name__ == "__main__":
    main()
