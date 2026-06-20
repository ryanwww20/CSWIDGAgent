#!/usr/bin/env python3
"""Headless notebook evaluator for fair, reproducible runs."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mK]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)

import nbformat
import yaml
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def _run_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def detect_hardware() -> dict[str, Any]:
    gpu_name = _run_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    gpu_mem = _run_command(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"]
    )
    cuda_version = _run_command(["nvidia-smi"])
    cuda_hint = None
    if cuda_version:
        for line in cuda_version.splitlines():
            if "CUDA Version" in line:
                cuda_hint = line.strip()
                break

    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "gpu_name": gpu_name,
        "gpu_memory_total": gpu_mem,
        "cuda_hint": cuda_hint,
    }


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping.")
    return cfg


def summarize_notebook(notebook: nbformat.NotebookNode) -> dict[str, Any]:
    code_cell_count = 0
    error_cells = 0
    errors: list[dict[str, Any]] = []

    for cell_index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        code_cell_count += 1
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                error_cells += 1
                tb_raw = "\n".join(out.get("traceback", []))
                errors.append({
                    "cell_index": cell_index,
                    "ename": out.get("ename", ""),
                    "evalue": out.get("evalue", ""),
                    "traceback_excerpt": _strip_ansi(tb_raw)[:2000],
                })
                break  # one error entry per cell

    return {
        "code_cell_count": code_cell_count,
        "error_cell_count": error_cells,
        "errors": errors,
    }


def execute_notebook(
    input_path: Path,
    output_path: Path,
    timeout: int,
    startup_timeout: int,
    kernel_name: str,
    allow_errors: bool,
) -> tuple[bool, str | None, dict[str, Any], float]:
    with input_path.open("r", encoding="utf-8") as fh:
        notebook = nbformat.read(fh, as_version=4)

    start = time.perf_counter()
    error_message = None
    success = True

    client = NotebookClient(
        notebook,
        timeout=timeout,
        startup_timeout=startup_timeout,
        kernel_name=kernel_name,
        allow_errors=allow_errors,
    )

    try:
        client.execute()
    except CellExecutionError as exc:
        success = False
        error_message = str(exc)
    except Exception as exc:  # broad catch for infrastructure failures
        success = False
        error_message = f"{type(exc).__name__}: {exc}"

    duration_seconds = time.perf_counter() - start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        nbformat.write(notebook, fh)

    stats = summarize_notebook(notebook)
    if not allow_errors and stats["error_cell_count"] > 0:
        success = False
        if error_message is None:
            error_message = "Notebook executed but contains error outputs."

    return success, error_message, stats, duration_seconds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-notebook", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if run_success is false.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = read_config(args.config)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    executed_notebook_path = run_dir / "executed.ipynb"
    report_path = run_dir / "report.json"

    execution_cfg = config.get("execution", {})
    timeout = int(execution_cfg.get("timeout_seconds", 180))
    startup_timeout = int(execution_cfg.get("startup_timeout_seconds", 90))
    kernel_name = str(execution_cfg.get("kernel_name", "python3"))
    allow_errors = bool(execution_cfg.get("allow_errors", False))

    success, error_message, stats, duration_seconds = execute_notebook(
        input_path=args.input_notebook,
        output_path=executed_notebook_path,
        timeout=timeout,
        startup_timeout=startup_timeout,
        kernel_name=kernel_name,
        allow_errors=allow_errors,
    )

    report = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_notebook": str(args.input_notebook),
        "executed_notebook": str(executed_notebook_path),
        "profile_name": config.get("profile_name"),
        "lane": config.get("lane"),
        "official_for_reporting": bool(config.get("official_for_reporting", False)),
        "run_success": success,
        "duration_seconds": round(duration_seconds, 4),
        "error_message": error_message,
        "stats": stats,
        "hardware": detect_hardware(),
        "config": config,
    }

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.strict and not success:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
