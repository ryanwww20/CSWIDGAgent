#!/usr/bin/env python3
"""One-command evaluation of a generated demo notebook on all 7 metrics.

Stages (each is also runnable standalone — see their own --help):
  A  execution    eval_notebook.py        -> #1 Run Success, #6 Efficiency
  B  interactivity run_interactivity.py   -> #5 Interactivity (+ frames)
  C  quality      run_quality_eval.py     -> #2 Faithfulness, #3 Pedagogy,
                                             #4 Topic, #7 Clarity

Everything a run produces — executed notebook, filmstrip, frames, judge
verdicts, quality report, and the RAW text of every LLM call (llm_calls.jsonl)
— is preserved under one run directory:

  results/<method>/<run_id>/
    ├── exec/            report.json, executed.ipynb          (stage A)
    ├── interactivity/   filmstrip.json, frames/, judge.json,
    │                    interactivity_score.json             (stage B)
    ├── quality/         quality_report.json, judge_images/   (stage C)
    ├── llm_calls.jsonl  every judge/planner/verifier exchange, verbatim
    └── summary.json     the 7-metric vector + provenance

Usage (explicit inputs):
  python evalkit/run_eval.py NB.ipynb --slides DECK.pdf --method my_method
      [--transcript T.txt] [--run-id ID] [--stages abc] [--no-llm]

Usage (instance from the manifest):
  python evalkit/run_eval.py NB.ipynb --instance HTLIN_ML__05_handout \
      --data-root /path/to/ml_colab --method my_method
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

KIT = Path(__file__).resolve().parent
PACK = KIT.parent

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except Exception:  # noqa: BLE001
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_instance(instance_id: str, data_root: Path) -> tuple[Path, Path | None, dict]:
    """Look up an instance in instances/manifest.jsonl -> (slides, transcript, meta)."""
    manifest = PACK / "instances" / "manifest.jsonl"
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("id") == instance_id:
                base = data_root / rec["rel_path"]
                slides = base / "slides_full.pdf"
                if not slides.exists():
                    raise FileNotFoundError(
                        f"instance data not found: {slides}\n"
                        f"(--data-root must point at the directory that contains "
                        f"'{rec['rel_path'].split('/')[0]}/' — see instances/README.md)")
                transcript = next(
                    (p for p in sorted(base.glob("transcript.*.txt")) if p.is_file()),
                    None)
                return slides, transcript, rec
    raise KeyError(f"instance id {instance_id!r} not found in {manifest}")


def _run(cmd: list[str], env: dict, label: str) -> int:
    print(f"\n=== {label} ===\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    return subprocess.run([str(c) for c in cmd], env=env).returncode


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path)
    src = ap.add_argument_group("source materials (give --slides OR --instance)")
    src.add_argument("--slides", type=Path, default=None)
    src.add_argument("--transcript", type=Path, default=None)
    src.add_argument("--instance", default=None,
                     help="instance id from instances/manifest.jsonl")
    src.add_argument("--data-root", type=Path, default=PACK.parent,
                     help="directory containing the testset data referenced by "
                          "the manifest (default: the pack's parent)")
    ap.add_argument("--method", default="unnamed_method",
                    help="who/what generated this notebook (results grouping)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", type=Path, default=PACK / "results")
    ap.add_argument("--stages", default="abc",
                    help="any of a (execution) b (interactivity) c (quality)")
    ap.add_argument("--config", type=Path, default=PACK / "configs" / "cpu-stable.yaml",
                    help="execution profile (official numbers: cpu-stable)")
    ap.add_argument("--no-llm", action="store_true",
                    help="deterministic parts only — no planner/judge/verifier")
    ap.add_argument("--max-points", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=6)
    ap.add_argument("--digest-slides", action="store_true",
                    help="build the once-per-deck vision slide digest first if "
                         "missing (image-heavy decks; needs an API key)")
    args = ap.parse_args()

    if not args.notebook.exists():
        print(f"notebook not found: {args.notebook}")
        return 2
    slides, transcript, inst_meta = args.slides, args.transcript, None
    if args.instance:
        slides, transcript, inst_meta = _resolve_instance(args.instance,
                                                          args.data_root)
        if args.transcript:
            transcript = args.transcript
    stages = set(args.stages.lower())

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.out / args.method / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")

    env = dict(os.environ)
    env["EVAL_LLM_LOG"] = str(run_dir / "llm_calls.jsonl")
    py = sys.executable

    rc: dict[str, int | None] = {"a": None, "b": None, "c": None}

    if args.digest_slides and slides and not args.no_llm:
        sidecar = slides.with_name(slides.name + ".digest.json")
        if not sidecar.exists():
            _run([py, KIT / "slide_digest.py", slides], env, "slide digest (once per deck)")

    # Stage A — execution (#1 Run Success, #6 Efficiency)
    executed = run_dir / "exec" / "executed.ipynb"
    if "a" in stages:
        rc["a"] = _run([py, KIT / "eval_notebook.py",
                        "--input-notebook", args.notebook,
                        "--config", args.config,
                        "--run-id", "exec",
                        "--output-dir", run_dir],
                       env, "stage A: execution")

    # Stage B — interactivity (#5) + frames
    if "b" in stages:
        cmd = [py, KIT / "run_interactivity.py", args.notebook,
               "--out", run_dir / "interactivity",
               "--max-points", str(args.max_points), "--source",
               "slides+transcript" if transcript else "slides"]
        if args.no_llm:
            cmd.append("--no-llm")
        rc["b"] = _run(cmd, env, "stage B: interactivity")

    # Stage C — quality judges (#2 #3 #4 #7)
    if "c" in stages:
        cmd = [py, KIT / "run_quality_eval.py", args.notebook,
               "--out", run_dir / "quality",
               "--max-images", str(args.max_images)]
        if executed.exists():
            cmd += ["--executed", executed]
        if slides:
            cmd += ["--slides", slides]
        if transcript:
            cmd += ["--transcript", transcript]
        frames = run_dir / "interactivity" / "frames"
        if frames.exists():
            cmd += ["--frames", frames]
        if args.no_llm:
            cmd.append("--no-llm")
        rc["c"] = _run(cmd, env, "stage C: quality judges")

    # ---- summary: the 7-metric vector + provenance -------------------------
    exec_report = _load(run_dir / "exec" / "report.json")
    inter = _load(run_dir / "interactivity" / "interactivity_score.json")
    quality = _load(run_dir / "quality" / "quality_report.json")

    def _q(section: str, field: str):
        if quality and isinstance(quality.get(section), dict):
            return quality[section].get(field)
        return None

    summary = {
        "method": args.method,
        "run_id": run_id,
        "notebook": str(args.notebook.resolve()),
        "notebook_sha256": _sha256(args.notebook),
        "instance": (inst_meta or {}).get("id"),
        "slides": str(slides) if slides else None,
        "transcript": str(transcript) if transcript else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage_exit_codes": rc,
        "no_llm": args.no_llm,
        "metrics": {
            "1_run_success": (exec_report or {}).get("run_success"),
            "2_faithfulness": _q("faithfulness", "score"),
            "2a_assertional": _q("faithfulness", "assertional"),
            "2b_computational": _q("faithfulness", "computational"),
            "2c_correctness": _q("faithfulness", "correctness"),
            "3_pedagogy": _q("pedagogy", "depth"),
            "4_topic": _q("topic", "worthiness"),
            "5_interactivity": (inter or {}).get("score"),
            "5_effectiveness": (inter or {}).get("effectiveness"),
            "5_robustness": (inter or {}).get("robustness"),
            "6_duration_seconds": (exec_report or {}).get("duration_seconds"),
            "7_clarity": _q("clarity", "score"),
            "7a_visual": _q("clarity", "visual"),
            "7b_textual": _q("clarity", "textual"),
            "7c_code_explanation": _q("clarity", "code_explanation"),
        },
        "models": (quality or {}).get("meta", {}),
        "artifacts": {
            "exec_report": "exec/report.json",
            "executed_notebook": "exec/executed.ipynb",
            "filmstrip": "interactivity/filmstrip.json",
            "interactivity_score": "interactivity/interactivity_score.json",
            "interactivity_judge": "interactivity/judge.json",
            "quality_report": "quality/quality_report.json",
            "llm_calls": "llm_calls.jsonl",
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== summary ===")
    for k, v in summary["metrics"].items():
        print(f"  {k:<22} {v if v is not None else 'NA'}")
    print(f"\nwrote {run_dir / 'summary.json'}")
    failed = [s for s, code in rc.items() if code not in (None, 0)]
    if failed:
        print(f"NOTE: stage(s) {','.join(failed)} exited non-zero — "
              f"check their output above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
