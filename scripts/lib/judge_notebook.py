#!/usr/bin/env python3
"""Blind LLM-judge scorer for the ablation study.

Wraps the existing rubric (``prompt/codex_evaluation.md``) into an automated
scorer: render a notebook to text, ask a judge model to score the 8 criteria on
a 1-5 scale, parse the JSON, optionally average across several independent
judges to cut single-sample variance.

Blinding: the judge never sees the condition / run tag — only the notebook
content (and, optionally, the source material for the alignment criterion). The
*order* in which notebooks are judged should be randomized by the orchestrator.

Output schema (judge_result.json)
----------------------------------
    {
      "notebook": "...",
      "n_judges": 3,
      "criteria": ["executability", ... 8 ...],
      "per_judge": [ {"scores": {...}, "overall_verdict": "...", "final_recommendation": "..."} ],
      "mean_scores": { "executability": 3.67, ... },
      "mean_overall": 3.5
    }

Usage
-----
    python judge_notebook.py --notebook nb.ipynb --output judge_result.json \
        [--source course_source/x.pdf] [--judges 3] [--claude-bin claude]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import nbformat

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

JSON_MARKER = "===JSON==="


def render_notebook(notebook_path: Path, max_chars: int = 120_000) -> str:
    nb = nbformat.read(str(notebook_path), as_version=4)
    parts: list[str] = []
    for i, cell in enumerate(nb.cells, 1):
        src = "".join(cell.get("source", "")) if isinstance(cell.get("source"), list) else cell.get("source", "")
        if cell.get("cell_type") == "markdown":
            parts.append(f"--- Cell {i} [markdown] ---\n{src}")
        else:
            parts.append(f"--- Cell {i} [code] ---\n```python\n{src}\n```")
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... notebook truncated for length ...]"
    return text


def build_prompt(rubric: str, rendered: str, source_path: str | None) -> str:
    source_note = (
        f"The original course source material is at '{source_path}'. Use your Read "
        f"tool to read it when scoring 'Alignment with Source Material'.\n\n"
        if source_path
        else "No source material is provided; score 'Alignment with Source Material' "
        "as 3 (neutral) and say so in its reason.\n\n"
    )
    schema_keys = ",\n    ".join(f'"{k}": {{"score": <1-5 int>, "reason": "<short>"}}' for k in CRITERIA)
    return (
        f"{rubric}\n\n"
        "=== INSTRUCTIONS FOR THIS AUTOMATED RUN ===\n"
        "Be strict and honest — this is used to compare notebook-generation methods.\n"
        f"{source_note}"
        "After your analysis, output your scores as a single JSON object. Put the "
        f"line '{JSON_MARKER}' on its own line, then ONLY the JSON object (no fences) "
        "as the very last thing in your reply, in exactly this shape:\n"
        "{\n"
        f"  \"scores\": {{\n    {schema_keys}\n  }},\n"
        "  \"overall_verdict\": \"Excellent|Good|Acceptable|Needs Major Revision|Not Suitable\",\n"
        "  \"final_recommendation\": \"<one of the 4 numbered recommendations>\"\n"
        "}\n\n"
        "=== NOTEBOOK UNDER EVALUATION ===\n"
        f"{rendered}\n"
    )


def call_claude(claude_bin: str, prompt: str, timeout: float = 900.0) -> str:
    result = subprocess.run(
        [claude_bin, "-p", prompt], capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude exited {result.returncode}: {(result.stderr or '').strip()[-500:]}"
        )
    return result.stdout


def parse_judgement(stdout: str) -> dict[str, Any]:
    # Prefer the text after our explicit marker; fall back to last JSON object.
    tail = stdout.rsplit(JSON_MARKER, 1)[-1].strip()
    candidates = [tail, stdout]
    decoder = json.JSONDecoder()
    for blob in candidates:
        for start in range(len(blob)):
            if blob[start] != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(blob[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "scores" in obj:
                return obj
    raise ValueError("no JSON object with a 'scores' field found in judge output")


def _score_of(judgement: dict[str, Any], key: str) -> float | None:
    entry = judgement.get("scores", {}).get(key)
    if isinstance(entry, dict):
        entry = entry.get("score")
    try:
        return float(entry)
    except (TypeError, ValueError):
        return None


def judge_notebook(
    notebook_path: Path,
    *,
    source_path: str | None,
    judges: int,
    claude_bin: str,
    rubric: str,
) -> dict[str, Any]:
    rendered = render_notebook(notebook_path)
    prompt = build_prompt(rubric, rendered, source_path)

    per_judge: list[dict[str, Any]] = []
    for j in range(1, judges + 1):
        try:
            stdout = call_claude(claude_bin, prompt)
            judgement = parse_judgement(stdout)
            per_judge.append(judgement)
        except Exception as exc:  # noqa: BLE001 — record the failure, keep going
            print(f"[judge] judge {j} failed: {exc}", file=sys.stderr)
            per_judge.append({"error": str(exc), "scores": {}})

    mean_scores: dict[str, float | None] = {}
    for key in CRITERIA:
        vals = [s for s in (_score_of(j, key) for j in per_judge) if s is not None]
        mean_scores[key] = round(sum(vals) / len(vals), 2) if vals else None
    numeric = [v for v in mean_scores.values() if v is not None]
    mean_overall = round(sum(numeric) / len(numeric), 2) if numeric else None

    return {
        "notebook": str(notebook_path),
        "source": source_path,
        "n_judges": judges,
        "criteria": CRITERIA,
        "per_judge": per_judge,
        "mean_scores": mean_scores,
        "mean_overall": mean_overall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a notebook with the codex rubric via an LLM judge")
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--source", default="", help="course source path (read by judge for alignment)")
    parser.add_argument("--judges", type=int, default=3)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--rubric", type=Path, default=None, help="rubric markdown (default: prompt/codex_evaluation.md)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rubric_path = args.rubric or (Path(__file__).resolve().parents[2] / "prompt" / "codex_evaluation.md")
    rubric = rubric_path.read_text(encoding="utf-8")

    result = judge_notebook(
        args.notebook,
        source_path=args.source or None,
        judges=args.judges,
        claude_bin=args.claude_bin,
        rubric=rubric,
    )

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
