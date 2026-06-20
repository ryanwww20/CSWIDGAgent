#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from notebook_assembler import AssemblyError, assemble_notebook, expected_notebook_path as assembler_notebook_path
from notebook_verifier import COLAB_KERNEL, VerificationError, build_execution_report, verify_notebook
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


def parse_stage_payload(stage_name: str, stdout: str, output_path: Path) -> dict[str, Any]:
    """Resolve a stage's JSON payload, tolerant of a prose final message.

    Stages 1-3 are instructed to BOTH write their artifact to ``output_path`` and
    return it on stdout. Models sometimes end with a human-style summary instead of
    raw JSON; when that happens we fall back to the artifact the agent already wrote
    to disk rather than failing the whole pipeline.
    """
    parse_err: str | None = None
    payload: Any = None
    try:
        payload = extract_json(stdout)
    except ValueError as exc:
        parse_err = str(exc)

    if isinstance(payload, dict):
        return payload

    # stdout was unparseable or not a JSON object — try the on-disk artifact.
    if output_path.exists():
        disk = load_json(output_path)
        if isinstance(disk, dict):
            reason = parse_err or f"top-level {type(payload).__name__}"
            log(
                f"{stage_name}: stdout was not a JSON object ({reason}); "
                f"using the on-disk artifact {output_path.name}"
            )
            return disk

    if parse_err is not None:
        die(
            f"{stage_name} did not return parseable JSON and no usable "
            f"artifact at {output_path}: {parse_err}"
        )
    die(
        f"{stage_name} must return a JSON object on stdout or write one to {output_path}"
    )


# Substrings that mark a transient (retryable) failure of the `claude` CLI:
# network/socket drops and upstream API hiccups. Matched case-insensitively against
# the combined stdout+stderr of a failed invocation.
TRANSIENT_ERROR_MARKERS = (
    "socket connection was closed",
    "socket hang up",
    "api error",
    "overloaded",
    "rate limit",
    "econnreset",
    "etimedout",
    "connection error",
    "connection reset",
    "fetch failed",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway time-out",
    "timed out",
    " 429",
    " 500",
    " 502",
    " 503",
    " 529",
)


def _looks_transient(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in TRANSIENT_ERROR_MARKERS)


def run_claude_prompt(
    claude_bin: str,
    prompt: str,
    dry_run: bool,
    *,
    max_attempts: int = 3,
    backoff_seconds: float = 10.0,
    timeout_seconds: float = 1800.0,
) -> str:
    cmd = [
        claude_bin, "-p", prompt,
        "--allowedTools", "Read,Write,Bash",
        "--dangerously-skip-permissions",
    ]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return ""

    last_error = "<none>"
    returncode: int | None = 1
    for attempt in range(1, max_attempts + 1):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired:
            # A hung claude (e.g. a half-open connection during a network outage)
            # would otherwise block forever — bound each attempt and treat the
            # timeout as transient so we retry instead of stalling the batch.
            returncode = None
            last_error = f"timed out after {timeout_seconds:.0f}s"
            print(
                f"[run_pipeline][ERROR] claude {last_error} "
                f"(attempt {attempt}/{max_attempts})",
                file=sys.stderr,
            )
            if attempt < max_attempts:
                wait = backoff_seconds * attempt
                print(
                    f"[run_pipeline] timeout treated as transient; retrying in {wait:.0f}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            break

        if result.returncode == 0:
            return result.stdout

        returncode = result.returncode
        last_error = (result.stderr or "").strip() or "<empty stderr>"
        combined = f"{result.stdout or ''}\n{result.stderr or ''}"

        # Surface the real failure instead of swallowing it behind a traceback.
        print(
            f"[run_pipeline][ERROR] claude exited {returncode} "
            f"(attempt {attempt}/{max_attempts})",
            file=sys.stderr,
        )
        if result.stderr and result.stderr.strip():
            print(
                f"[run_pipeline][ERROR] stderr: {result.stderr.strip()[-2000:]}",
                file=sys.stderr,
            )
        stdout_tail = (result.stdout or "").strip()[-1000:]
        if stdout_tail:
            print(f"[run_pipeline][ERROR] stdout tail: {stdout_tail}", file=sys.stderr)

        if attempt < max_attempts and _looks_transient(combined):
            wait = backoff_seconds * attempt
            print(
                f"[run_pipeline] transient error detected; retrying in {wait:.0f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue
        break

    die(
        f"claude invocation failed after {max_attempts} attempt(s) "
        f"(exit {returncode}). Last error: {last_error[-500:]}"
    )


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
    elif response_mode == "stage5_fix":
        suffix = (
            "Edit pipeline_outputs/04_cell_sources.json in place to fix the failing cells. "
            "Do NOT write the .ipynb file — the pipeline re-assembles and re-verifies it. "
            "Return ONLY the fix report content as valid JSON. "
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


def run_stage5_verification(
    structure: dict[str, Any],
    notebook_path: Path,
    cell_sources_path: Path,
    structure_path: Path,
    analysis_path: Path,
    execution_report_path: Path,
    *,
    claude_bin: str,
    autofix: bool,
    max_fix_attempts: int,
    cell_timeout: int,
    startup_timeout: int,
    kernel_name: str = COLAB_KERNEL,
) -> dict[str, Any]:
    """Stage 5: verify the assembled notebook, then optionally repair + re-verify."""
    fixer_agent = structure_path.parent.parent / ".claude" / "agents" / "notebook-fixer.md"
    fix_attempts: list[dict[str, Any]] = []

    def _verify():
        try:
            return verify_notebook(
                notebook_path,
                structure,
                execute=True,
                cell_timeout=cell_timeout,
                startup_timeout=startup_timeout,
                kernel_name=kernel_name,
            )
        except VerificationError as exc:
            die(f"notebook verification could not run: {exc}")

    def _write_report(result) -> dict[str, Any]:
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        report = build_execution_report(
            result,
            timestamp_utc=timestamp_utc,
            fix_attempts=fix_attempts,
        )
        try:
            validate_stage_output("notebook-verifier", report)
        except ValidationError as exc:
            die(f"05_execution_report.json failed validation: {exc}")
        write_json(execution_report_path, report, dry_run=False)
        return report

    def _log_result(result) -> None:
        syntax_n = len(result.syntax_failures)
        exec_n = len(result.execution_failures)
        colab = "colab-match" if result.kernel_matches_colab else "NON-colab"
        log(
            f"Stage 5 verify [kernel={result.kernel_used} ({colab})]: "
            f"syntax_ok={result.syntax_ok} runnable={result.runnable} "
            f"(syntax_failures={syntax_n}, execution_failures={exec_n}, "
            f"{result.duration_seconds:.1f}s)"
        )

    result = _verify()
    _log_result(result)
    report = _write_report(result)

    attempt = 0
    while not result.ok and autofix and attempt < max_fix_attempts:
        attempt += 1
        if not cell_sources_path.exists():
            log(f"Stage 5: cannot autofix, {cell_sources_path.name} missing; reporting failures only")
            break
        log(f"Stage 5: notebook not runnable; running notebook-fixer (attempt {attempt}/{max_fix_attempts})")

        agent_instructions = load_agent_instructions(fixer_agent)
        task = (
            f"Read '{execution_report_path}' for the failing cells. "
            f"Read '{cell_sources_path}', '{structure_path}', and '{analysis_path}'. "
            f"Fix the failing cells in '{cell_sources_path}' in place, preserving cell count, "
            "ids, order, and types."
        )
        prompt = make_prompt(agent_instructions, task, response_mode="stage5_fix")
        stdout = run_claude_prompt(claude_bin, prompt, dry_run=False)

        fixed_ids: list[str] = []
        notes = ""
        try:
            fix_report = extract_json(stdout)
            if isinstance(fix_report, dict):
                fixed_ids = fix_report.get("fixed_cell_ids", []) or []
                notes = fix_report.get("notes", "") or ""
        except ValueError:
            notes = "notebook-fixer returned no parseable report"

        cell_sources = load_json(cell_sources_path)
        try:
            validate_cell_sources(cell_sources, structure)
            assemble_notebook(structure, cell_sources, notebook_path)
        except (ValidationError, AssemblyError) as exc:
            die(f"notebook-fixer attempt {attempt} produced invalid cell sources: {exc}")

        result = _verify()
        fix_attempts.append({
            "attempt": attempt,
            "fixed_cell_ids": fixed_ids,
            "notes": notes,
            "syntax_ok_after": result.syntax_ok,
            "runnable_after": result.runnable,
        })
        _log_result(result)
        report = _write_report(result)

    status = "completed" if result.ok else ("repaired" if fix_attempts else "failed")
    log(
        f"Stage 5 complete ({status}): syntax_ok={result.syntax_ok}, "
        f"runnable={result.runnable}, fix_attempts={len(fix_attempts)}"
    )
    log(f"Wrote {execution_report_path}")
    return report


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

    # Report-only verification (assemble-only is the no-agent path, so no autofix).
    execution_report_path = pipeline_dir / "05_execution_report.json"
    syntax_ok: bool | None = None
    verified = False
    try:
        verify_result = verify_notebook(notebook_path, structure, execute=True)
        exec_report = build_execution_report(
            verify_result, timestamp_utc=timestamp_utc,
        )
        write_json(execution_report_path, exec_report, dry_run=False)
        runnable = bool(verify_result.runnable)
        syntax_ok = bool(verify_result.syntax_ok)
        verified = True
        log(
            f"Assemble-only verify: syntax_ok={syntax_ok}, runnable={runnable} "
            f"({verify_result.duration_seconds:.1f}s)"
        )
    except VerificationError as exc:
        log(f"Assemble-only verification skipped: {exc}")

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
            "notebook_verifier": "completed" if verified else "skipped",
        },
        "artifacts": {
            "concepts": "pipeline_outputs/01_concepts.json",
            "structure": "pipeline_outputs/02_notebook_structure.json",
            "analysis": "pipeline_outputs/03_cell_analysis.json",
            "cell_sources": "pipeline_outputs/04_cell_sources.json",
            "generation_report": "pipeline_outputs/04_generation_report.json",
            "execution_report": "pipeline_outputs/05_execution_report.json",
            "final_notebook": notebook_target,
        },
        "summary": {
            "top_to_bottom_runnable": runnable,
            "syntax_ok": syntax_ok,
            "verified_by_execution": verified,
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
    parser.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip Stage 5 (notebook execution verification)")
    parser.add_argument("--no-autofix", action="store_true",
                        help="Stage 5 reports failures but does not run notebook-fixer")
    parser.add_argument("--max-fix-attempts", type=int, default=2,
                        help="max notebook-fixer rounds in Stage 5 (default 2)")
    parser.add_argument("--cell-timeout", type=int, default=120,
                        help="per-cell execution timeout in seconds (Stage 5)")
    parser.add_argument("--startup-timeout", type=int, default=60,
                        help="kernel startup timeout in seconds (Stage 5)")
    parser.add_argument("--kernel-name", default=COLAB_KERNEL,
                        help=f"Stage 5 execution kernel (default '{COLAB_KERNEL}', the "
                             "Colab-matching runtime; falls back to python3 if not installed)")
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

        if stage["name"] == "demo-coder":
            # The report must come from the model's stdout — the on-disk
            # 04_generation_report.json may be stale from a prior topic, so no
            # disk fallback here.
            try:
                payload = extract_json(stdout)
            except ValueError as exc:
                die(f"{stage['name']} did not return parseable JSON: {exc}")
            if not isinstance(payload, dict):
                die(f"{stage['name']} must return a JSON object")
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

        # Stages 1-3: the agent also writes its artifact to disk, so tolerate a
        # prose final message by falling back to that file.
        payload = parse_stage_payload(stage["name"], stdout, stage["output"])

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
        if not args.skip_verify:
            log(
                f"[dry-run] would verify {notebook_path} "
                f"(autofix={not args.no_autofix}, max_attempts={args.max_fix_attempts})"
            )
        log("Dry-run complete.")
        return

    report = load_json(report_path)
    final_notebook = report.get("final_notebook_path", notebook_target)
    runnable = bool(report.get("execution_status", {}).get("top_to_bottom_runnable", False))
    structure = structure_payload or load_json(structure_path)
    structure_cells = structure.get("cells", [])
    code_cells = sum(1 for cell in structure_cells if cell.get("cell_type") == "code")
    markdown_cells = len(structure_cells) - code_cells

    execution_report_path = pipeline_dir / "05_execution_report.json"
    verify_report: dict[str, Any] | None = None
    if args.skip_verify:
        log("Stage 5 (notebook-verifier) skipped via --skip-verify")
        verifier_status = "skipped"
        syntax_ok: bool | None = None
        colab_match: bool | None = None
    else:
        log("Running stage: notebook-verifier")
        verify_report = run_stage5_verification(
            structure,
            notebook_path,
            cell_sources_path,
            structure_path,
            analysis_path,
            execution_report_path,
            claude_bin=args.claude_bin,
            autofix=not args.no_autofix,
            max_fix_attempts=args.max_fix_attempts,
            cell_timeout=args.cell_timeout,
            startup_timeout=args.startup_timeout,
            kernel_name=args.kernel_name,
        )
        final_status = verify_report["final_status"]
        runnable = bool(final_status["runnable"])  # authoritative: real kernel execution
        syntax_ok = bool(final_status["syntax_ok"])
        colab_match = bool(verify_report.get("colab_runtime_match", False))
        verifier_status = "completed" if (runnable and syntax_ok) else "failed"

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
            "notebook_verifier": verifier_status,
        },
        "artifacts": {
            "concepts": "pipeline_outputs/01_concepts.json",
            "structure": "pipeline_outputs/02_notebook_structure.json",
            "analysis": "pipeline_outputs/03_cell_analysis.json",
            "cell_sources": "pipeline_outputs/04_cell_sources.json",
            "generation_report": "pipeline_outputs/04_generation_report.json",
            "execution_report": "pipeline_outputs/05_execution_report.json",
            "final_notebook": final_notebook,
        },
        "summary": {
            "top_to_bottom_runnable": runnable,
            "syntax_ok": syntax_ok,
            "verified_by_execution": verify_report is not None,
            "colab_runtime_match": colab_match,
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