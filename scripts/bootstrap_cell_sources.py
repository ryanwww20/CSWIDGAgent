#!/usr/bin/env python3
"""Bootstrap 04_cell_sources.json from an existing notebook + structure artifact.

Useful for regression testing the Phase 2 assembler path without re-running
demo-coder. See docs/phase2_cell_sources_and_assembler.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from pipeline_validator import ValidationError, validate_cell_sources


def cell_source_from_nbcell(nb_cell: dict) -> str:
    source = nb_cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def bootstrap(
    structure_path: Path,
    notebook_path: Path,
    topic: str,
    output_path: Path,
) -> dict:
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    structure_cells = structure.get("cells", [])
    nb_cells = notebook.get("cells", [])

    if len(structure_cells) != len(nb_cells):
        raise SystemExit(
            f"cell count mismatch: structure={len(structure_cells)}, notebook={len(nb_cells)}"
        )

    cells = []
    for struct_cell, nb_cell in zip(structure_cells, nb_cells):
        if struct_cell["cell_type"] != nb_cell.get("cell_type"):
            raise SystemExit(
                f"{struct_cell['cell_id']} type mismatch: "
                f"structure={struct_cell['cell_type']}, notebook={nb_cell.get('cell_type')}"
            )
        cells.append(
            {
                "cell_id": struct_cell["cell_id"],
                "cell_type": struct_cell["cell_type"],
                "source": cell_source_from_nbcell(nb_cell),
                "generation_notes": "Bootstrapped from existing notebook for assembler regression.",
            }
        )

    payload = {
        "topic": topic,
        "notebook_title": structure.get("notebook_title", topic),
        "source_artifacts": {
            "structure": "pipeline_outputs/02_notebook_structure.json",
            "analysis": "pipeline_outputs/03_cell_analysis.json",
        },
        "cells": cells,
        "assumptions": [
            "Cell sources extracted mechanically from an existing .ipynb; not produced by demo-coder.",
        ],
    }
    validate_cell_sources(payload, structure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap 04_cell_sources.json from notebook")
    parser.add_argument("--structure", type=Path, default=ROOT / "pipeline_outputs/02_notebook_structure.json")
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "pipeline_outputs/04_cell_sources.json")
    args = parser.parse_args()

    try:
        payload = bootstrap(args.structure, args.notebook, args.topic, args.output)
    except ValidationError as exc:
        raise SystemExit(f"validation failed: {exc}") from exc

    print(f"Wrote {args.output} ({len(payload['cells'])} cells)")


if __name__ == "__main__":
    main()
