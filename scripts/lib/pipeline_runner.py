#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_validator import ValidationError, validate_demo_coder_outputs, validate_stage_output


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


def load_agent_instructions(agent_file: Path) -> str:
    if not agent_file.exists():
        die(f"agent instruction file missing: {agent_file}")
    return agent_file.read_text(encoding="utf-8").strip()


def make_prompt(agent_instructions: str, task: str, *, json_only: bool = True) -> str:
    suffix = (
        "Return ONLY valid JSON. Do not include markdown fences or extra commentary."
        if json_only
        else (
            "Write the notebook file to disk as instructed, then return ONLY the "
            "04_generation_report.json content as valid JSON. "
            "Do not include markdown fences or extra commentary."
        )
    )
    return (
        "Follow these agent instructions exactly:\n\n"
        f"{agent_instructions}\n\n"
        f"Task: {task}\n\n"
        f"{suffix}"
    )


def expected_notebook_path(topic: str) -> str:
    return f"notebooks/{topic}_interactive_skill.ipynb"


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
    notebook_target = expected_notebook_path(args.topic)

    concepts_path = pipeline_dir / "01_concepts.json"
    structure_path = pipeline_dir / "02_notebook_structure.json"
    analysis_path = pipeline_dir / "03_cell_analysis.json"
    report_path = pipeline_dir / "04_generation_report.json"

    stages = [
        {
            "name": "concept-extractor",
            "agent_file": agents_dir / "concept-extractor.md",
            "output": concepts_path,
            "json_only": True,
            "task": (
                f"Analyze course source at '{source_path}' and produce the Stage 1 artifact. "
                f"Write the result to '{concepts_path}'."
            ),
        },
        {
            "name": "notebook-architect",
            "agent_file": agents_dir / "notebook-architect.md",
            "output": structure_path,
            "json_only": True,
            "task": (
                f"Read input artifact '{concepts_path}' and produce Stage 2 output. "
                f"Write the result to '{structure_path}'."
            ),
        },
        {
            "name": "cell-analyzer",
            "agent_file": agents_dir / "cell-analyzer.md",
            "output": analysis_path,
            "json_only": True,
            "task": (
                f"Read input artifact '{structure_path}' and produce Stage 3 output. "
                f"Write the result to '{analysis_path}'."
            ),
        },
        {
            "name": "demo-coder",
            "agent_file": agents_dir / "demo-coder.md",
            "output": report_path,
            "json_only": False,
            "task": (
                f"Read '{structure_path}' and '{analysis_path}'. "
                f"Generate topic '{args.topic}' and write the notebook to '{root_dir / notebook_target}'. "
                f"Then produce Stage 4 report JSON and write it to '{report_path}'."
            ),
        },
    ]

    structure_payload: dict[str, Any] | None = None
    errors: list[str] = []

    for stage in stages:
        agent_instructions = load_agent_instructions(stage["agent_file"])
        log(f"Running stage: {stage['name']}")
        prompt = make_prompt(
            agent_instructions,
            stage["task"],
            json_only=stage["json_only"],
        )
        stdout = run_claude_prompt(args.claude_bin, prompt, args.dry_run)
        if args.dry_run:
            continue

        try:
            payload = extract_json(stdout)
        except ValueError as exc:
            die(f"{stage['name']} did not return parseable JSON: {exc}")

        if not isinstance(payload, dict):
            die(f"{stage['name']} must return a JSON object")

        try:
            if stage["name"] == "cell-analyzer":
                if structure_payload is None:
                    structure_payload = load_json(structure_path)
                validate_stage_output(stage["name"], payload, structure=structure_payload)
            elif stage["name"] == "demo-coder":
                if structure_payload is None:
                    structure_payload = load_json(structure_path)
                validate_stage_output(stage["name"], payload)
                cell_counts = validate_demo_coder_outputs(payload, structure_payload, root_dir)
                log(
                    "Stage 4 notebook verified: "
                    f"{cell_counts['total_cells']} cells "
                    f"({cell_counts['code_cells']} code, {cell_counts['markdown_cells']} markdown)"
                )
            else:
                validate_stage_output(stage["name"], payload)
        except ValidationError as exc:
            die(f"{stage['name']} output failed validation: {exc}")

        write_json(stage["output"], payload, dry_run=False)
        load_json(stage["output"])

        if stage["name"] == "notebook-architect":
            structure_payload = payload

        log(f"Wrote {stage['output']}")

    if args.dry_run:
        log("Dry-run complete.")
        return

    report = load_json(report_path)
    final_notebook = report.get("final_notebook_path", notebook_target)
    runnable = bool(report.get("execution_status", {}).get("top_to_bottom_runnable", False))
    structure = structure_payload or load_json(structure_path)
    structure_cells = structure.get("cells", [])
    code_cells = sum(1 for cell in structure_cells if cell.get("cell_type") == "code")
    markdown_cells = len(structure_cells) - code_cells

    run_log = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "topic": args.topic,
        "course_source_path": str(source_path),
        "generation_mode": "artifact_driven",
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
        "summary": {
            "top_to_bottom_runnable": runnable,
            "total_cells": len(structure_cells),
            "code_cells": code_cells,
            "markdown_cells": markdown_cells,
        },
        "legacy_notes": None,
        "errors": errors,
    }

    run_log_path = pipeline_dir / "run_log.json"
    write_json(run_log_path, run_log, dry_run=False)
    load_json(run_log_path)
    log("Pipeline completed successfully.")
    log(f"Run ID: {run_id}")
    log(f"Generation mode: artifact_driven")
    log(f"Run log: {run_log_path}")


if __name__ == "__main__":
    main()
