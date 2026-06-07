#!/usr/bin/env python3
"""Validate pipeline stage outputs against agent contracts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ValidationError(Exception):
    """Raised when a stage output fails contract checks."""


STAGE_REQUIRED_TOP_LEVEL: dict[str, set[str]] = {
    "concept-extractor": {
        "course",
        "source_path",
        "candidates",
        "selected_concepts",
        "selection_rationale",
        "assumptions",
    },
    "notebook-architect": {
        "notebook_title",
        "learning_objectives",
        "cells",
        "global_flow_notes",
        "assumptions",
    },
    "cell-analyzer": {
        "cell_specs",
        "cross_cell_invariants",
        "assumptions",
    },
    "demo-coder": {
        "final_notebook_path",
        "generated_cells",
        "execution_status",
        "dependency_notes",
        "assumptions",
    },
}

CANDIDATE_REQUIRED = {
    "concept",
    "importance_score",
    "demo_feasibility_score",
    "prerequisites",
    "why_it_matters",
    "transcript_summary",
}

CELL_REQUIRED = {
    "cell_id",
    "cell_type",
    "goal",
    "inputs",
    "outputs",
    "widget_plan",
    "estimated_lines",
    "depends_on",
}

CELL_SPEC_REQUIRED = {
    "cell_id",
    "implementation_plan",
    "function_signatures",
    "state_variables",
    "error_handling",
    "test_checks",
}

GENERATED_CELL_REQUIRED = {"cell_id", "status", "notes"}
EXECUTION_STATUS_REQUIRED = {"top_to_bottom_runnable", "failed_cell_ids"}


def _require_keys(obj: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - set(obj.keys())
    if missing:
        raise ValidationError(f"{label} missing required keys: {sorted(missing)}")


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{label} must be a list")
    return value


def validate_concept_extractor(payload: dict[str, Any]) -> None:
    _require_keys(payload, STAGE_REQUIRED_TOP_LEVEL["concept-extractor"], "01_concepts.json")
    candidates = _require_list(payload["candidates"], "candidates")
    if not candidates:
        raise ValidationError("candidates must not be empty")
    for idx, item in enumerate(candidates):
        if not isinstance(item, dict):
            raise ValidationError(f"candidates[{idx}] must be an object")
        _require_keys(item, CANDIDATE_REQUIRED, f"candidates[{idx}]")
    selected = _require_list(payload["selected_concepts"], "selected_concepts")
    if not selected:
        raise ValidationError("selected_concepts must contain at least one concept")


def validate_notebook_architect(payload: dict[str, Any]) -> None:
    _require_keys(payload, STAGE_REQUIRED_TOP_LEVEL["notebook-architect"], "02_notebook_structure.json")
    cells = _require_list(payload["cells"], "cells")
    if not cells:
        raise ValidationError("cells must not be empty")
    for idx, item in enumerate(cells):
        if not isinstance(item, dict):
            raise ValidationError(f"cells[{idx}] must be an object")
        _require_keys(item, CELL_REQUIRED, f"cells[{idx}]")
        if item["cell_type"] not in {"markdown", "code"}:
            raise ValidationError(f"cells[{idx}].cell_type must be 'markdown' or 'code'")


def validate_cell_analyzer(payload: dict[str, Any], structure: dict[str, Any] | None = None) -> None:
    _require_keys(payload, STAGE_REQUIRED_TOP_LEVEL["cell-analyzer"], "03_cell_analysis.json")
    specs = _require_list(payload["cell_specs"], "cell_specs")
    if not specs:
        raise ValidationError("cell_specs must not be empty")
    for idx, item in enumerate(specs):
        if not isinstance(item, dict):
            raise ValidationError(f"cell_specs[{idx}] must be an object")
        _require_keys(item, CELL_SPEC_REQUIRED, f"cell_specs[{idx}]")

    if structure is not None:
        code_ids = {
            cell["cell_id"]
            for cell in structure.get("cells", [])
            if cell.get("cell_type") == "code"
        }
        spec_ids = {spec["cell_id"] for spec in specs}
        missing = code_ids - spec_ids
        extra = spec_ids - code_ids
        if missing:
            raise ValidationError(
                f"cell_specs missing implementation plans for code cells: {sorted(missing)}"
            )
        if extra:
            raise ValidationError(
                f"cell_specs contain ids not present as code cells in structure: {sorted(extra)}"
            )


def validate_generation_report(payload: dict[str, Any]) -> None:
    _require_keys(payload, STAGE_REQUIRED_TOP_LEVEL["demo-coder"], "04_generation_report.json")
    generated = _require_list(payload["generated_cells"], "generated_cells")
    if not generated:
        raise ValidationError("generated_cells must not be empty")
    for idx, item in enumerate(generated):
        if not isinstance(item, dict):
            raise ValidationError(f"generated_cells[{idx}] must be an object")
        _require_keys(item, GENERATED_CELL_REQUIRED, f"generated_cells[{idx}]")
    status = payload["execution_status"]
    if not isinstance(status, dict):
        raise ValidationError("execution_status must be an object")
    _require_keys(status, EXECUTION_STATUS_REQUIRED, "execution_status")


def validate_stage_output(stage_name: str, payload: dict[str, Any], structure: dict[str, Any] | None = None) -> None:
    if stage_name == "concept-extractor":
        validate_concept_extractor(payload)
    elif stage_name == "notebook-architect":
        validate_notebook_architect(payload)
    elif stage_name == "cell-analyzer":
        validate_cell_analyzer(payload, structure=structure)
    elif stage_name == "demo-coder":
        validate_generation_report(payload)
    else:
        raise ValidationError(f"unknown stage: {stage_name}")


def validate_notebook_file(notebook_path: Path) -> dict[str, Any]:
    if not notebook_path.exists():
        raise ValidationError(f"notebook file does not exist: {notebook_path}")
    if not notebook_path.is_file():
        raise ValidationError(f"notebook path is not a file: {notebook_path}")
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"notebook is not valid JSON: {exc}") from exc

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValidationError("notebook must contain a non-empty cells array")

    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValidationError(f"notebook.cells[{idx}] must be an object")
        cell_type = cell.get("cell_type")
        if cell_type not in {"markdown", "code"}:
            raise ValidationError(f"notebook.cells[{idx}].cell_type must be 'markdown' or 'code'")
        source = cell.get("source")
        if source is None:
            raise ValidationError(f"notebook.cells[{idx}] missing source")

    return notebook


def validate_demo_coder_outputs(
    report: dict[str, Any],
    structure: dict[str, Any],
    root_dir: Path,
) -> dict[str, int]:
    """Validate stage 4 report + on-disk notebook against stage 2 structure."""
    validate_generation_report(report)

    structure_cells = _require_list(structure.get("cells"), "02_notebook_structure.json cells")
    expected_total = len(structure_cells)
    expected_code = sum(1 for cell in structure_cells if cell.get("cell_type") == "code")
    expected_markdown = expected_total - expected_code

    generated = _require_list(report["generated_cells"], "generated_cells")
    if len(generated) != expected_total:
        raise ValidationError(
            f"generated_cells count ({len(generated)}) does not match structure cells ({expected_total})"
        )

    structure_ids = [cell["cell_id"] for cell in structure_cells]
    generated_ids = [item["cell_id"] for item in generated]
    if structure_ids != generated_ids:
        raise ValidationError(
            "generated_cells cell_id order does not match 02_notebook_structure.json: "
            f"structure={structure_ids}, report={generated_ids}"
        )

    rel_path = report["final_notebook_path"]
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValidationError("final_notebook_path must be a non-empty string")

    notebook_path = Path(rel_path)
    if not notebook_path.is_absolute():
        notebook_path = (root_dir / notebook_path).resolve()

    notebook = validate_notebook_file(notebook_path)
    actual_total = len(notebook["cells"])
    if actual_total != expected_total:
        raise ValidationError(
            f"notebook cell count ({actual_total}) does not match structure ({expected_total})"
        )

    actual_code = sum(1 for cell in notebook["cells"] if cell.get("cell_type") == "code")
    actual_markdown = actual_total - actual_code
    if actual_code != expected_code or actual_markdown != expected_markdown:
        raise ValidationError(
            "notebook cell type counts do not match structure: "
            f"notebook(code={actual_code}, markdown={actual_markdown}) vs "
            f"structure(code={expected_code}, markdown={expected_markdown})"
        )

    return {
        "total_cells": actual_total,
        "code_cells": actual_code,
        "markdown_cells": actual_markdown,
    }
