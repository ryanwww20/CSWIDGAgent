#!/usr/bin/env python3
"""Assemble a Jupyter notebook from structure + cell-source artifacts.

Phase 2 component. See docs/phase2_cell_sources_and_assembler.md for schema
and interface contract.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


CellType = Literal["markdown", "code"]


class AssemblyError(Exception):
    """Raised when cell sources cannot be assembled into a notebook."""


CELL_SOURCE_REQUIRED = {"cell_id", "cell_type", "source", "generation_notes"}
CELL_SOURCES_TOP_LEVEL = {"topic", "notebook_title", "source_artifacts", "cells", "assumptions"}


@dataclass(frozen=True)
class AssembleConfig:
    python_version: str = "3.10.0"
    indent: int = 1
    ensure_parent_dir: bool = True


@dataclass(frozen=True)
class AssembleResult:
    output_path: Path
    total_cells: int
    code_cells: int
    markdown_cells: int
    notebook_bytes: int


def expected_notebook_path(topic: str) -> str:
    return f"notebooks/{topic}_interactive_skill.ipynb"


def _require_keys(obj: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - set(obj.keys())
    if missing:
        raise AssemblyError(f"{label} missing required keys: {sorted(missing)}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssemblyError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssemblyError(f"{path} must contain a JSON object")
    return payload


def validate_cell_sources_payload(
    structure: dict[str, Any],
    cell_sources: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate 04_cell_sources.json against 02_notebook_structure.json.

    Returns the normalized list of cell source entries in structure order.
    """
    _require_keys(cell_sources, CELL_SOURCES_TOP_LEVEL, "04_cell_sources.json")

    structure_cells = structure.get("cells")
    if not isinstance(structure_cells, list) or not structure_cells:
        raise AssemblyError("02_notebook_structure.json must contain a non-empty cells array")

    source_cells = cell_sources.get("cells")
    if not isinstance(source_cells, list) or not source_cells:
        raise AssemblyError("04_cell_sources.json must contain a non-empty cells array")

    if len(source_cells) != len(structure_cells):
        raise AssemblyError(
            f"cell count mismatch: structure has {len(structure_cells)}, "
            f"cell_sources has {len(source_cells)}"
        )

    by_id: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(source_cells):
        if not isinstance(item, dict):
            raise AssemblyError(f"cell_sources.cells[{idx}] must be an object")
        _require_keys(item, CELL_SOURCE_REQUIRED, f"cell_sources.cells[{idx}]")
        cell_id = item["cell_id"]
        if cell_id in by_id:
            raise AssemblyError(f"duplicate cell_id in cell_sources: {cell_id}")
        by_id[cell_id] = item

    ordered: list[dict[str, Any]] = []
    for idx, struct_cell in enumerate(structure_cells):
        if not isinstance(struct_cell, dict):
            raise AssemblyError(f"structure.cells[{idx}] must be an object")

        cell_id = struct_cell.get("cell_id")
        cell_type = struct_cell.get("cell_type")
        if not isinstance(cell_id, str) or not isinstance(cell_type, str):
            raise AssemblyError(f"structure.cells[{idx}] missing cell_id or cell_type")

        if cell_id not in by_id:
            raise AssemblyError(f"cell_sources missing entry for {cell_id}")

        source_entry = by_id[cell_id]
        if source_entry["cell_type"] != cell_type:
            raise AssemblyError(
                f"{cell_id} cell_type mismatch: structure={cell_type}, "
                f"cell_sources={source_entry['cell_type']}"
            )
        if cell_type not in {"markdown", "code"}:
            raise AssemblyError(f"{cell_id} has unsupported cell_type: {cell_type}")

        source_text = source_entry["source"]
        if not isinstance(source_text, str) or not source_text.strip():
            raise AssemblyError(f"{cell_id} source must be a non-empty string")

        ordered.append(source_entry)

    return ordered


def source_to_nbformat_lines(source: str) -> list[str]:
    """Convert a plain-text cell source to nbformat line array."""
    if not source:
        return []
    lines = source.splitlines(keepends=True)
    if not lines:
        return [source]
    if not source.endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return lines


def build_nbformat_cell(cell_type: CellType, source: str) -> dict[str, Any]:
    nb_source = source_to_nbformat_lines(source)
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": nb_source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": nb_source,
    }


def build_notebook_document(
    structure: dict[str, Any],
    ordered_sources: list[dict[str, Any]],
    *,
    config: AssembleConfig,
) -> dict[str, Any]:
    cells = [
        build_nbformat_cell(entry["cell_type"], entry["source"])
        for entry in ordered_sources
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": config.python_version,
            },
        },
        "cells": cells,
    }


def assemble_notebook(
    structure: dict[str, Any],
    cell_sources: dict[str, Any],
    output_path: Path,
    *,
    config: AssembleConfig | None = None,
) -> AssembleResult:
    """Merge structure + cell sources into a valid .ipynb file."""
    cfg = config or AssembleConfig()
    ordered_sources = validate_cell_sources_payload(structure, cell_sources)
    notebook = build_notebook_document(structure, ordered_sources, config=cfg)

    if cfg.ensure_parent_dir:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    serialized = json.dumps(notebook, ensure_ascii=False, indent=cfg.indent) + "\n"
    output_path.write_text(serialized, encoding="utf-8")

    code_cells = sum(1 for entry in ordered_sources if entry["cell_type"] == "code")
    total_cells = len(ordered_sources)
    return AssembleResult(
        output_path=output_path.resolve(),
        total_cells=total_cells,
        code_cells=code_cells,
        markdown_cells=total_cells - code_cells,
        notebook_bytes=output_path.stat().st_size,
    )


def assemble_from_files(
    structure_path: Path,
    cell_sources_path: Path,
    output_path: Path,
    *,
    config: AssembleConfig | None = None,
) -> AssembleResult:
    """Load artifacts from disk and assemble a notebook."""
    structure = _load_json(structure_path)
    cell_sources = _load_json(cell_sources_path)
    return assemble_notebook(structure, cell_sources, output_path, config=config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble notebook from pipeline artifacts")
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--cell-sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python-version", default="3.10.0")
    parser.add_argument("--indent", type=int, default=1)
    args = parser.parse_args()

    result = assemble_from_files(
        args.structure,
        args.cell_sources,
        args.output,
        config=AssembleConfig(python_version=args.python_version, indent=args.indent),
    )
    print(f"Notebook written to: {result.output_path}")
    print(
        f"Cells: {result.total_cells} total "
        f"({result.code_cells} code, {result.markdown_cells} markdown), "
        f"{result.notebook_bytes} bytes"
    )


if __name__ == "__main__":
    main()
