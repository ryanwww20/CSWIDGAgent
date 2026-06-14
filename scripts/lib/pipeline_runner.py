#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from notebook_assembler import AssemblyError, assemble_notebook, expected_notebook_path as assembler_notebook_path
from pipeline_validator import (
    ValidationError,
    validate_cell_sources,
    validate_demo_coder_outputs,
    validate_stage_output,
)


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
    cmd = [
        claude_bin, "-p", prompt,
        "--allowedTools", "Read,Write,Bash",
        "--dangerously-skip-permissions",
    ]
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


def make_prompt(agent_instructions: str, task: str, *, response_mode: str = "json_only") -> str:
    if response_mode == "json_only":
        suffix = "Return ONLY valid JSON. Do not include markdown fences or extra commentary."
    elif response_mode == "stage4_report":
        suffix = (
            "Write pipeline_outputs/04_cell_sources.json to disk as instructed. "
            "Do NOT write the .ipynb file — the pipeline assembler builds it from cell sources. "
            "Return ONLY the 04_generation_report.json content as valid JSON. "
            "Do not include markdown fences or extra commentary."
        )
    else:
        die(f"unknown response_mode: {response_mode}")

    return (
        "Follow these agent instructions exactly:\n\n"
        f"{agent_instructions}\n\n"
        f"Task: {task}\n\n"
        f"{suffix}"
    )


def run_stage4_assembly(
    structure: dict[str, Any],
    cell_sources_path: Path,
    notebook_path: Path,
    report: dict[str, Any],
    root_dir: Path,
) -> dict[str, int]:
    cell_sources = load_json(cell_sources_path)
    try:
        validate_cell_sources(cell_sources, structure)
    except ValidationError as exc:
        die(f"demo-coder cell sources failed validation: {exc}")

    try:
        assemble_result = assemble_notebook(structure, cell_sources, notebook_path)
    except AssemblyError as exc:
        die(f"notebook assembly failed: {exc}")

    log(
        "Assembled notebook: "
        f"{assemble_result.output_path} "
        f"({assemble_result.total_cells} cells, {assemble_result.notebook_bytes} bytes)"
    )

    try:
        validate_stage_output("demo-coder", report)
        return validate_demo_coder_outputs(report, structure, root_dir)
    except ValidationError as exc:
        die(f"demo-coder report/notebook gate failed: {exc}")


def run_assemble_only(root_dir: Path, topic: str, source_path: Path | None) -> None:
    pipeline_dir = root_dir / "pipeline_outputs"
    structure_path = pipeline_dir / "02_notebook_structure.json"
    cell_sources_path = pipeline_dir / "04_cell_sources.json"
    report_path = pipeline_dir / "04_generation_report.json"
    notebook_target = assembler_notebook_path(topic)
    notebook_path = root_dir / notebook_target

    structure = load_json(structure_path)
    if not cell_sources_path.exists():
        die(
            f"missing {cell_sources_path}. "
            "Run demo-coder or bootstrap_cell_sources.py first."
        )

    if report_path.exists():
        report = load_json(report_path)
    else:
        log(f"{report_path} not found; using minimal report for assembly check")
        report = {
            "final_notebook_path": notebook_target,
            "generated_cells": [
                {"cell_id": cell["cell_id"], "status": "generated", "notes": "assemble-only"}
                for cell in structure.get("cells", [])
            ],
            "execution_status": {"top_to_bottom_runnable": False, "failed_cell_ids": []},
            "dependency_notes": ["assemble-only run: execution not verified"],
            "assumptions": ["Notebook assembled from existing 04_cell_sources.json"],
        }

    cell_counts = run_stage4_assembly(structure, cell_sources_path, notebook_path, report, root_dir)
    log(
        "Assemble-only complete: "
        f"{cell_counts['total_cells']} cells "
        f"({cell_counts['code_cells']} code, {cell_counts['markdown_cells']} markdown)"
    )

    if source_path is None and (pipeline_dir / "01_concepts.json").exists():
        concepts = load_json(pipeline_dir / "01_concepts.json")
        source_path = Path(concepts.get("source_path", ""))
        if not source_path.is_absolute():
            source_path = (root_dir / source_path).resolve()

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    structure_cells = structure.get("cells", [])
    code_cells = sum(1 for cell in structure_cells if cell.get("cell_type") == "code")
    runnable = bool(report.get("execution_status", {}).get("top_to_bottom_runnable", False))

    run_log = {
        "run_id": f"assemble-only-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp_utc": timestamp_utc,
        "topic": topic,
        "course_source_path": str(source_path) if source_path else None,
        "generation_mode": "artifact_driven",
        "stage_status": {
            "concept_extractor": "skipped",
            "notebook_architect": "skipped",
            "cell_analyzer": "skipped",
            "demo_coder": "assemble_only",
        },
        "artifacts": {
            "concepts": "pipeline_outputs/01_concepts.json",
            "structure": "pipeline_outputs/02_notebook_structure.json",
            "analysis": "pipeline_outputs/03_cell_analysis.json",
            "cell_sources": "pipeline_outputs/04_cell_sources.json",
            "generation_report": "pipeline_outputs/04_generation_report.json",
            "final_notebook": notebook_target,
        },
        "summary": {
            "top_to_bottom_runnable": runnable,
            "total_cells": len(structure_cells),
            "code_cells": code_cells,
            "markdown_cells": len(structure_cells) - code_cells,
        },
        "legacy_notes": None,
        "errors": [],
    }
    write_json(pipeline_dir / "run_log.json", run_log, dry_run=False)
    log(f"Run log: {pipeline_dir / 'run_log.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4-stage Claude pipeline")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    source_path: Path | None = None
    if args.source:
        source = Path(args.source)
        source_path = source if source.is_absolute() else (root_dir / source).resolve()
        if not source_path.exists():
            die(f"source path does not exist: {source_path}")

    if args.assemble_only:
        run_assemble_only(root_dir, args.topic, source_path)
        return

    if not args.source:
        die("--source is required unless --assemble-only is set")

    pipeline_dir = root_dir / "pipeline_outputs"
    notebook_dir = root_dir / "notebooks"
    agents_dir = root_dir / ".claude" / "agents"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    notebook_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    notebook_target = assembler_notebook_path(args.topic)
    notebook_path = root_dir / notebook_target

    concepts_path = pipeline_dir / "01_concepts.json"
    structure_path = pipeline_dir / "02_notebook_structure.json"
    analysis_path = pipeline_dir / "03_cell_analysis.json"
    cell_sources_path = pipeline_dir / "04_cell_sources.json"
    report_path = pipeline_dir / "04_generation_report.json"

    stages = [
        {
            "name": "concept-extractor",
            "agent_file": agents_dir / "concept-extractor.md",
            "output": concepts_path,
            "response_mode": "json_only",
            "task": (
                f"Analyze course source at '{source_path}' and produce the Stage 1 artifact. "
                f"Write the result to '{concepts_path}'."
            ),
        },
        {
            "name": "notebook-architect",
            "agent_file": agents_dir / "notebook-architect.md",
            "output": structure_path,
            "response_mode": "json_only",
            "task": (
                f"Read input artifact '{concepts_path}' and produce Stage 2 output. "
                f"Write the result to '{structure_path}'."
            ),
        },
        {
            "name": "cell-analyzer",
            "agent_file": agents_dir / "cell-analyzer.md",
            "output": analysis_path,
            "response_mode": "json_only",
            "task": (
                f"Read input artifact '{structure_path}' and produce Stage 3 output. "
                f"Write the result to '{analysis_path}'."
            ),
        },
        {
            "name": "demo-coder",
            "agent_file": agents_dir / "demo-coder.md",
            "output": report_path,
            "response_mode": "stage4_report",
            "task": (
                f"Read '{structure_path}' and '{analysis_path}'. "
                f"Generate topic '{args.topic}'. "
                f"Write cell sources to '{cell_sources_path}'. "
                f"Set final_notebook_path to '{notebook_target}' in the report."
            ),
        },
    ]

    structure_payload: dict[str, Any] | None = None

    # --- skip stages before --from-stage; ensure their outputs already exist ---
    if args.from_stage > 1:
        for skipped in stages[: args.from_stage - 1]:
            out = Path(skipped["output"])
            if not out.exists():
                die(
                    f"--from-stage {args.from_stage} requires {skipped['name']} output "
                    f"at {out}, but it does not exist. Run from an earlier stage first."
                )
            log(f"Skipping {skipped['name']} (using existing {out.name})")
        # demo-coder needs structure in memory; preload it when we skip past stage 2
        if structure_path.exists():
            structure_payload = load_json(structure_path)
        stages = stages[args.from_stage - 1:]

    for stage in stages:
        agent_instructions = load_agent_instructions(stage["agent_file"])
        log(f"Running stage: {stage['name']}")
        prompt = make_prompt(
            agent_instructions,
            stage["task"],
            response_mode=stage["response_mode"],
        )
        stdout = run_claude_prompt(args.claude_bin, prompt, args.dry_run)
        print(f"[DEBUG] raw output:\n{stdout[:500]}")
        if args.dry_run:
            if stage["name"] == "demo-coder":
                log(f"[dry-run] would assemble notebook at {notebook_path}")
            continue

        try:
            payload = extract_json(stdout)
        except ValueError:
            output_path = stage["output"]
            if Path(output_path).exists():
                log(f"{stage['name']} stdout not JSON; reading from {output_path}")
                payload = load_json(Path(output_path))
            else:
                die(f"{stage['name']} did not return parseable JSON and no output file found")

        if not isinstance(payload, dict):
            die(f"{stage['name']} must return a JSON object")

        if stage["name"] == "demo-coder":
            if structure_payload is None:
                structure_payload = load_json(structure_path)
            if not cell_sources_path.exists():
                die(
                    f"demo-coder did not write {cell_sources_path}. "
                    "Stage 4 must write cell sources before returning the report."
                )
            cell_counts = run_stage4_assembly(
                structure_payload,
                cell_sources_path,
                notebook_path,
                payload,
                root_dir,
            )
            log(
                "Stage 4 complete: "
                f"{cell_counts['total_cells']} cells "
                f"({cell_counts['code_cells']} code, {cell_counts['markdown_cells']} markdown)"
            )
            write_json(report_path, payload, dry_run=False)
            load_json(report_path)
            log(f"Wrote {report_path}")
            continue

        try:
            if stage["name"] == "cell-analyzer":
                if structure_payload is None:
                    structure_payload = load_json(structure_path)
                validate_stage_output(stage["name"], payload, structure=structure_payload)
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
            "cell_sources": "pipeline_outputs/04_cell_sources.json",
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
        "errors": [],
    }

    run_log_path = pipeline_dir / "run_log.json"
    write_json(run_log_path, run_log, dry_run=False)
    load_json(run_log_path)
    log("Pipeline completed successfully.")
    log(f"Run ID: {run_id}")
    log("Generation mode: artifact_driven")
    log(f"Run log: {run_log_path}")


if __name__ == "__main__":
    main()