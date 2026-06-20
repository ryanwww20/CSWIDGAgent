#!/usr/bin/env python3
"""Objective executability harness for the ablation study.

Runs a generated notebook top-to-bottom in a fresh kernel and reports, as JSON,
whether it ran to completion and — if not — *where* and *why* it failed.

Why this exists
---------------
The pipeline's only existing executability signal is ``top_to_bottom_runnable``
in ``04_generation_report.json``, which the demo-coder agent *reports about its
own output*. That is unusable for an ablation (the model grades itself). This
harness actually executes the notebook and produces an independent verdict.

Failure classification (so local vs Colab runs stay comparable)
---------------------------------------------------------------
These notebooks target Google Colab, where heavy deps (torch, torchvision, ...)
are preinstalled. Run locally without them, an ``import torch`` fails for an
*environment* reason, not because the agent wrote a bad notebook. We therefore
classify the first failure:

- ``environment_dependency`` — ModuleNotFoundError / ImportError. In a Colab-like
  env these should not occur; locally they flag "needs the Colab stack", NOT an
  agent-quality failure. Aggregation can exclude these from the quality verdict.
- ``timeout``               — a cell exceeded the per-cell timeout (often a
  training loop that needs a GPU; environment, not correctness).
- ``runtime_error``         — any other exception raised by a cell. This is the
  real agent-quality signal.
- ``kernel_error``          — the kernel failed to start / died.

Usage
-----
    python execute_notebook.py --notebook nb.ipynb --output execution_result.json \
        [--per-cell-timeout 300] [--kernel python3] [--save-executed out.ipynb]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

ENV_ERROR_NAMES = {"ModuleNotFoundError", "ImportError"}


def _classify(error_name: str | None) -> str:
    if error_name in ENV_ERROR_NAMES:
        return "environment_dependency"
    return "runtime_error"


def _first_error_cell(nb: Any) -> tuple[int | None, int | None, str | None, str | None]:
    """Return (cell_index, code_cell_ordinal, ename, evalue) of the first cell
    whose outputs contain an error, or (None, None, None, None)."""
    code_ordinal = 0
    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        code_ordinal += 1
        for out in cell.get("outputs", []) or []:
            if out.get("output_type") == "error":
                return idx, code_ordinal, out.get("ename"), _shorten(out.get("evalue", ""))
    return None, None, None, None


def _shorten(text: str, limit: int = 500) -> str:
    text = (text or "").strip().replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def execute_notebook(
    notebook_path: Path,
    *,
    per_cell_timeout: int = 300,
    kernel_name: str = "python3",
    save_executed: Path | None = None,
) -> dict[str, Any]:
    nb = nbformat.read(str(notebook_path), as_version=4)
    code_cells = [c for c in nb.cells if c.get("cell_type") == "code"]
    n_code = len(code_cells)

    # Start from a clean slate so prior saved outputs never leak into the verdict.
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    client = NotebookClient(
        nb,
        timeout=per_cell_timeout,
        kernel_name=kernel_name,
        allow_errors=True,  # run every cell; we scan outputs for the first error
        record_timing=True,
    )

    status = "completed"
    fatal: str | None = None
    started = time.perf_counter()
    try:
        client.execute()
    except CellTimeoutError as exc:
        status = "timeout"
        fatal = _shorten(str(exc))
    except CellExecutionError as exc:  # only if a cell sets allow_errors=False via tag
        status = "runtime_error"
        fatal = _shorten(str(exc))
    except Exception as exc:  # kernel failed to start / died mid-run
        status = "kernel_error"
        fatal = f"{type(exc).__name__}: {_shorten(str(exc))}"
    wall = round(time.perf_counter() - started, 2)

    if save_executed is not None:
        save_executed.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, str(save_executed))

    fail_idx, fail_ordinal, ename, evalue = _first_error_cell(nb)

    if status == "completed" and fail_idx is None:
        error_type = None
        ran = True
    elif status == "timeout":
        error_type = "timeout"
        ran = False
    elif status == "kernel_error":
        error_type = "kernel_error"
        ran = False
        evalue = evalue or fatal
    else:
        error_type = _classify(ename)
        ran = False

    executed = sum(
        1
        for c in nb.cells
        if c.get("cell_type") == "code" and c.get("execution_count") is not None
    )

    return {
        "notebook": str(notebook_path),
        "ran_to_completion": ran,
        "error_type": error_type,
        "error_name": ename,
        "error_value": evalue,
        "first_failed_cell_index": fail_idx,
        "first_failed_code_cell": fail_ordinal,
        "n_code_cells": n_code,
        "executed_code_cells": executed,
        "wall_seconds": wall,
        "per_cell_timeout": per_cell_timeout,
        "kernel": kernel_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a notebook and emit a JSON verdict")
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="write the JSON verdict here")
    parser.add_argument("--per-cell-timeout", type=int, default=300)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--save-executed", type=Path, help="save the executed notebook here")
    args = parser.parse_args()

    if not args.notebook.exists():
        result = {
            "notebook": str(args.notebook),
            "ran_to_completion": False,
            "error_type": "missing_notebook",
            "error_name": "FileNotFoundError",
            "error_value": f"notebook not found: {args.notebook}",
            "first_failed_cell_index": None,
            "first_failed_code_cell": None,
            "n_code_cells": 0,
            "executed_code_cells": 0,
            "wall_seconds": 0.0,
            "per_cell_timeout": args.per_cell_timeout,
            "kernel": args.kernel,
        }
    else:
        result = execute_notebook(
            args.notebook,
            per_cell_timeout=args.per_cell_timeout,
            kernel_name=args.kernel,
            save_executed=args.save_executed,
        )

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)

    # Exit non-zero on real failure so shell orchestration can branch, but treat
    # environment/timeout as "not a quality failure" → still exit 0.
    if result["error_type"] in {"runtime_error", "kernel_error", "missing_notebook"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
