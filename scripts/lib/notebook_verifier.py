#!/usr/bin/env python3
"""Verify an assembled notebook: per-cell syntax check + top-to-bottom execution.

Stage 5 component (deterministic core). The pipeline runner pairs this with the
`notebook-fixer` LLM agent to repair failing cells. See docs/pipeline.md for the
Stage 5 contract and 05_execution_report.json schema.

Design mirrors scripts/lib/notebook_assembler.py: pure-Python, no LLM, importable
helpers plus a thin CLI.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mK]")


class VerificationError(Exception):
    """Raised when verification cannot run (bad inputs, missing notebook)."""


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


@dataclass
class CellFailure:
    cell_index: int
    cell_id: str | None
    phase: str  # "syntax" | "execution"
    ename: str
    evalue: str
    traceback_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_index": self.cell_index,
            "cell_id": self.cell_id,
            "phase": self.phase,
            "ename": self.ename,
            "evalue": self.evalue,
            "traceback_excerpt": self.traceback_excerpt,
        }


@dataclass
class VerifyResult:
    notebook_path: Path
    syntax_ok: bool
    executed: bool
    runnable: bool
    code_cell_count: int
    duration_seconds: float
    syntax_failures: list[CellFailure] = field(default_factory=list)
    execution_failures: list[CellFailure] = field(default_factory=list)
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.syntax_ok and self.runnable


def _cell_source_text(cell: Any) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source or ""


def _id_by_index(structure: dict[str, Any] | None) -> list[str | None]:
    """Map notebook cell positions to structure cell_ids (1:1 order)."""
    if not structure:
        return []
    cells = structure.get("cells")
    if not isinstance(cells, list):
        return []
    return [c.get("cell_id") if isinstance(c, dict) else None for c in cells]


def _transform_magics(source: str) -> str:
    """Convert IPython magics / shell escapes to valid Python before compiling.

    Colab cells frequently use ``%matplotlib inline`` or ``!pip install ...`` which
    are not valid Python and would raise spurious SyntaxErrors under raw compile().
    """
    try:
        from IPython.core.inputtransformer2 import TransformerManager

        return TransformerManager().transform_cell(source)
    except Exception:
        # IPython unavailable or transform failed; fall back to raw source.
        return source


def check_syntax(
    notebook: nbformat.NotebookNode,
    id_by_index: list[str | None],
) -> list[CellFailure]:
    failures: list[CellFailure] = []
    for cell_index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source_text(cell)
        if not source.strip():
            continue
        cell_id = id_by_index[cell_index] if cell_index < len(id_by_index) else None
        transformed = _transform_magics(source)
        try:
            compile(transformed, cell_id or f"cell[{cell_index}]", "exec")
        except SyntaxError as exc:
            failures.append(
                CellFailure(
                    cell_index=cell_index,
                    cell_id=cell_id,
                    phase="syntax",
                    ename="SyntaxError",
                    evalue=f"{exc.msg} (line {exc.lineno})",
                    traceback_excerpt=str(exc),
                )
            )
    return failures


def collect_execution_failures(
    notebook: nbformat.NotebookNode,
    id_by_index: list[str | None],
) -> list[CellFailure]:
    failures: list[CellFailure] = []
    for cell_index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                cell_id = id_by_index[cell_index] if cell_index < len(id_by_index) else None
                tb_raw = "\n".join(out.get("traceback", []))
                failures.append(
                    CellFailure(
                        cell_index=cell_index,
                        cell_id=cell_id,
                        phase="execution",
                        ename=out.get("ename", ""),
                        evalue=out.get("evalue", ""),
                        traceback_excerpt=_strip_ansi(tb_raw)[:2000],
                    )
                )
                break  # one error entry per cell
    return failures


def verify_notebook(
    notebook_path: Path,
    structure: dict[str, Any] | None = None,
    *,
    execute: bool = True,
    cell_timeout: int = 120,
    startup_timeout: int = 60,
    kernel_name: str = "python3",
) -> VerifyResult:
    """Run syntax check and (optionally) top-to-bottom execution on a notebook."""
    if not notebook_path.exists():
        raise VerificationError(f"notebook does not exist: {notebook_path}")

    import warnings

    with notebook_path.open("r", encoding="utf-8") as fh:
        with warnings.catch_warnings():
            # Assembler-built cells omit nbformat ids; harmless for execution.
            warnings.simplefilter("ignore")
            notebook = nbformat.read(fh, as_version=4)

    id_by_index = _id_by_index(structure)
    code_cell_count = sum(1 for c in notebook.cells if c.get("cell_type") == "code")

    syntax_failures = check_syntax(notebook, id_by_index)
    syntax_ok = not syntax_failures

    # Skip execution when syntax is broken: the kernel would just fail the first
    # bad cell and the syntax report is already the actionable signal.
    if not syntax_ok or not execute:
        return VerifyResult(
            notebook_path=notebook_path,
            syntax_ok=syntax_ok,
            executed=False,
            runnable=False,
            code_cell_count=code_cell_count,
            duration_seconds=0.0,
            syntax_failures=syntax_failures,
            execution_failures=[],
            error_message=None if syntax_ok else "syntax errors present; execution skipped",
        )

    start = time.perf_counter()
    error_message: str | None = None
    client = NotebookClient(
        notebook,
        timeout=cell_timeout,
        startup_timeout=startup_timeout,
        kernel_name=kernel_name,
        allow_errors=True,  # run all cells so we can report every failure at once
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        error_message = str(exc)
    except Exception as exc:  # broad catch for kernel/infrastructure failures
        error_message = f"{type(exc).__name__}: {exc}"
    duration_seconds = time.perf_counter() - start

    execution_failures = collect_execution_failures(notebook, id_by_index)
    runnable = not execution_failures and error_message is None

    return VerifyResult(
        notebook_path=notebook_path,
        syntax_ok=True,
        executed=True,
        runnable=runnable,
        code_cell_count=code_cell_count,
        duration_seconds=round(duration_seconds, 4),
        syntax_failures=[],
        execution_failures=execution_failures,
        error_message=error_message,
    )


def build_execution_report(
    result: VerifyResult,
    *,
    kernel_name: str,
    timestamp_utc: str,
    fix_attempts: list[dict[str, Any]] | None = None,
    assumptions: list[str] | None = None,
) -> dict[str, Any]:
    """Build the 05_execution_report.json payload from a VerifyResult."""
    fix_attempts = fix_attempts or []
    return {
        "notebook_path": str(result.notebook_path),
        "verified_at_utc": timestamp_utc,
        "kernel_name": kernel_name,
        "code_cell_count": result.code_cell_count,
        "syntax_check": {
            "passed": result.syntax_ok,
            "failures": [f.to_dict() for f in result.syntax_failures],
        },
        "execution": {
            "attempted": result.executed,
            "runnable": result.runnable,
            "duration_seconds": result.duration_seconds,
            "error_message": result.error_message,
            "failures": [f.to_dict() for f in result.execution_failures],
        },
        "fix_attempts": fix_attempts,
        "final_status": {
            "syntax_ok": result.syntax_ok,
            "runnable": result.runnable,
            "fix_attempts_used": len(fix_attempts),
        },
        "assumptions": assumptions or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a notebook: syntax + execution")
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--structure", type=Path, default=None,
                        help="02_notebook_structure.json, for cell_id mapping")
    parser.add_argument("--output", type=Path, default=None,
                        help="where to write 05_execution_report.json")
    parser.add_argument("--no-execute", action="store_true",
                        help="static syntax check only; do not run the kernel")
    parser.add_argument("--cell-timeout", type=int, default=120)
    parser.add_argument("--startup-timeout", type=int, default=60)
    parser.add_argument("--kernel-name", default="python3")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if the notebook is not runnable")
    args = parser.parse_args()

    structure = None
    if args.structure and args.structure.exists():
        structure = json.loads(args.structure.read_text(encoding="utf-8"))

    result = verify_notebook(
        args.notebook,
        structure,
        execute=not args.no_execute,
        cell_timeout=args.cell_timeout,
        startup_timeout=args.startup_timeout,
        kernel_name=args.kernel_name,
    )

    from datetime import datetime, timezone

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = build_execution_report(result, kernel_name=args.kernel_name, timestamp_utc=timestamp_utc)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.strict and not result.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
