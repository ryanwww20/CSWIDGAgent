#!/usr/bin/env python3
"""Aggregate evalkit per-notebook scores into a condition-level ablation table.

Reads every ``evalkit/results/<condition>/<topic>__s<seed>/summary.json`` (the
7-metric vector written by ``evalkit/run_eval.py``), groups by ablation condition,
and emits:

  results/evalkit_runs.csv          one row per (condition, topic, seed)
  results/evalkit_by_condition.md   per-condition mean of each metric, plus the
                                    PAIRED delta vs the baseline condition (default B)

Paired delta is the right comparison for an ablation: topic difficulty dominates
raw variance, so for each metric we average (condition - baseline) over the
(topic, seed) pairs present in BOTH, instead of differencing the raw means.

Stdlib only. evalkit's run-id convention is ``<topic>__s<seed>``.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

# (key in summary["metrics"], short column label). 1_run_success is a bool ->0/1.
METRICS = [
    ("1_run_success", "run_success"),
    ("2_faithfulness", "faithfulness"),
    ("3_pedagogy", "pedagogy"),
    ("4_topic", "topic_worth"),
    ("5_interactivity", "interactivity"),
    ("7_clarity", "clarity"),
    ("6_duration_seconds", "duration_s"),
]


def _num(v):
    """Coerce a metric to float; bools -> 0/1; None/NA -> None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_run_id(run_id: str) -> tuple[str, str]:
    """`<topic>__s<seed>` -> (topic, seed). Falls back to (run_id, '')."""
    if "__s" in run_id:
        topic, _, seed = run_id.rpartition("__s")
        return topic, seed
    return run_id, ""


def load_rows(results_dir: Path, conditions: list[str] | None) -> list[dict]:
    rows: list[dict] = []
    for summary_path in sorted(results_dir.glob("*/*/summary.json")):
        cond = summary_path.parent.parent.name
        if conditions and cond not in conditions:
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = data.get("metrics", {}) or {}
        topic, seed = _parse_run_id(data.get("run_id", summary_path.parent.name))
        row = {
            "condition": cond,
            "topic": topic,
            "seed": seed,
            "no_llm": data.get("no_llm"),
        }
        for key, label in METRICS:
            row[label] = _num(metrics.get(key))
        rows.append(row)
    return rows


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def write_runs_csv(rows: list[dict], out: Path) -> None:
    cols = ["condition", "topic", "seed", "no_llm"] + [label for _, label in METRICS]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["condition"], r["topic"], r["seed"])):
            w.writerow({c: r.get(c) for c in cols})


def write_condition_md(rows: list[dict], out: Path, order: list[str], baseline: str) -> None:
    labels = [label for _, label in METRICS]
    by_cond: dict[str, list[dict]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)

    present = [c for c in order if c in by_cond] + [
        c for c in sorted(by_cond) if c not in order
    ]

    def fmt(v):
        return "—" if v is None else f"{v:.2f}"

    lines = [f"# evalkit ablation summary ({len(rows)} runs)", ""]
    lines.append("## Per-condition mean (7-metric vector)")
    lines.append("")
    lines.append("| Condition | n | " + " | ".join(labels) + " |")
    lines.append("|---|--:|" + "--:|" * len(labels))
    cond_means: dict[str, dict[str, float | None]] = {}
    for cond in present:
        cr = by_cond[cond]
        means = {label: _mean([r[label] for r in cr]) for label in labels}
        cond_means[cond] = means
        lines.append(
            f"| {cond} | {len(cr)} | " + " | ".join(fmt(means[l]) for l in labels) + " |"
        )

    # Paired deltas vs baseline: per metric, mean over shared (topic,seed) of (cond-base).
    if baseline in by_cond:
        base_idx = {(r["topic"], r["seed"]): r for r in by_cond[baseline]}
        lines += ["", f"## Paired delta vs `{baseline}` (mean of per-(topic,seed) differences)", ""]
        lines.append("| Condition | n_paired | " + " | ".join(labels) + " |")
        lines.append("|---|--:|" + "--:|" * len(labels))
        for cond in present:
            if cond == baseline:
                continue
            deltas: dict[str, list[float]] = {l: [] for l in labels}
            n_paired = 0
            for r in by_cond[cond]:
                base = base_idx.get((r["topic"], r["seed"]))
                if not base:
                    continue
                n_paired += 1
                for label in labels:
                    if r[label] is not None and base[label] is not None:
                        deltas[label].append(r[label] - base[label])

            def fmt_d(vals):
                m = _mean(vals)
                if m is None:
                    return "—"
                return f"{m:+.2f}"

            lines.append(
                f"| {cond} | {n_paired} | "
                + " | ".join(fmt_d(deltas[l]) for l in labels)
                + " |"
            )
    else:
        lines += ["", f"_(baseline `{baseline}` not found — paired deltas skipped)_"]

    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("evalkit/results"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--conditions", default="",
                    help="space-separated condition allowlist (default: all dirs found)")
    ap.add_argument("--baseline", default="B", help="condition to compute paired deltas against")
    args = ap.parse_args()

    conditions = args.conditions.split() if args.conditions.strip() else None
    rows = load_rows(args.results_dir, conditions)
    if not rows:
        print(f"no evalkit summaries under {args.results_dir}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_runs_csv(rows, args.out_dir / "evalkit_runs.csv")
    order = conditions or []
    write_condition_md(rows, args.out_dir / "evalkit_by_condition.md", order, args.baseline)
    print(f"wrote {args.out_dir / 'evalkit_runs.csv'} ({len(rows)} runs)")
    print(f"wrote {args.out_dir / 'evalkit_by_condition.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
