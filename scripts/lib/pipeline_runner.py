#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log(message: str) -> None:
    print(f"[run_pipeline] {message}")


def die(message: str) -> None:
    print(f"[run_pipeline][ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


def extract_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()

    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no valid JSON payload found in model output")


def run_claude_prompt(claude_bin: str, prompt: str, dry_run: bool) -> str:
    cmd = [claude_bin, "-p", prompt]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return ""
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def write_json(path: Path, payload: Any, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] write {path}")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    if not path.exists():
        die(f"missing file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(f"invalid JSON in {path}: {exc}")


def make_prompt(agent_file: Path, task: str) -> str:
    return (
        f"You are agent following instruction file: {agent_file}. "
        f"{task} "
        "Return ONLY valid JSON. Do not include markdown fences or extra commentary."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4-stage Claude pipeline")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--claude-bin", default="claude")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    source = Path(args.source)
    source_path = source if source.is_absolute() else (root_dir / source).resolve()
    if not source_path.exists():
        die(f"source path does not exist: {source_path}")

    pipeline_dir = root_dir / "pipeline_outputs"
    notebook_dir = root_dir / "notebooks"
    agents_dir = root_dir / ".claude" / "agents"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    notebook_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    stages = [
        {
            "name": "concept-extractor",
            "agent_file": agents_dir / "concept-extractor.md",
            "output": pipeline_dir / "01_concepts.json",
            "task": (
                f"Analyze course source at '{source_path}' and produce the Stage 1 artifact "
                f"for this project. The output file target is '{pipeline_dir / '01_concepts.json'}'."
            ),
        },
        {
            "name": "notebook-architect",
            "agent_file": agents_dir / "notebook-architect.md",
            "output": pipeline_dir / "02_notebook_structure.json",
            "task": (
                f"Read input artifact '{pipeline_dir / '01_concepts.json'}' and produce Stage 2 output. "
                f"The output file target is '{pipeline_dir / '02_notebook_structure.json'}'."
            ),
        },
        {
            "name": "cell-analyzer",
            "agent_file": agents_dir / "cell-analyzer.md",
            "output": pipeline_dir / "03_cell_analysis.json",
            "task": (
                f"Read input artifact '{pipeline_dir / '02_notebook_structure.json'}' and produce Stage 3 output. "
                f"The output file target is '{pipeline_dir / '03_cell_analysis.json'}'."
            ),
        },
        {
            "name": "demo-coder",
            "agent_file": agents_dir / "demo-coder.md",
            "output": pipeline_dir / "04_generation_report.json",
            "task": (
                f"Read input artifact '{pipeline_dir / '03_cell_analysis.json'}', generate a notebook for topic "
                f"'{args.topic}', and produce Stage 4 report JSON. The output file target is "
                f"'{pipeline_dir / '04_generation_report.json'}'."
            ),
        },
    ]

    for stage in stages:
        if not stage["agent_file"].exists():
            die(f"agent instruction file missing: {stage['agent_file']}")
        log(f"Running stage: {stage['name']}")
        prompt = make_prompt(stage["agent_file"], stage["task"])
        stdout = run_claude_prompt(args.claude_bin, prompt, args.dry_run)
        if args.dry_run:
            continue
        try:
            payload = extract_json(stdout)
        except ValueError as exc:
            die(f"{stage['name']} did not return parseable JSON: {exc}")
        write_json(stage["output"], payload, dry_run=False)
        load_json(stage["output"])
        log(f"Wrote {stage['output']}")

    if args.dry_run:
        log("Dry-run complete.")
        return

    report = load_json(pipeline_dir / "04_generation_report.json")
    final_notebook = report.get("final_notebook_path", f"notebooks/{args.topic}_interactive_skill.ipynb")
    runnable = bool(report.get("execution_status", {}).get("top_to_bottom_runnable", False))

    run_log = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "course_source_path": str(source_path),
        "selected_concept": args.topic,
        "stage_status": {
            "concept_extractor": "completed",
            "notebook_architect": "completed",
            "cell_analyzer": "completed",
            "demo_coder": "completed",
        },
        "artifacts": {
            "concepts": "pipeline_outputs/01_concepts.json",
            "structure": "pipeline_outputs/02_notebook_structure.json",
            "analysis": "pipeline_outputs/03_cell_analysis.json",
            "generation_report": "pipeline_outputs/04_generation_report.json",
            "final_notebook": final_notebook,
        },
        "summary": {"top_to_bottom_runnable": runnable},
        "errors": [],
    }

    run_log_path = pipeline_dir / "run_log.json"
    write_json(run_log_path, run_log, dry_run=False)
    load_json(run_log_path)
    log("Pipeline completed successfully.")
    log(f"Run ID: {run_id}")
    log(f"Run log: {run_log_path}")


if __name__ == "__main__":
    main()
