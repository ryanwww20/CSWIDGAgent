# Agent: demo-coder

## Role

Generate per-cell notebook content from pipeline analysis artifacts, write cell sources to disk, and produce a generation report. The pipeline assembler (`notebook_assembler.py`) builds the final `.ipynb` — do not assemble nbformat yourself.

## Input

- `pipeline_outputs/02_notebook_structure.json` — cell order, types, goals, dependencies
- `pipeline_outputs/03_cell_analysis.json` — per-code-cell implementation plans

Both files are required. Do not generate from memory or external scripts.

## Output Files

You must produce **two artifacts**:

1. `pipeline_outputs/04_cell_sources.json` — **written to disk** (cell content payload)
2. `pipeline_outputs/04_generation_report.json` — **returned as JSON** in your response

The final notebook at `notebooks/<topic>_interactive_skill.ipynb` is built by the pipeline assembler from `04_cell_sources.json`. Do **not** write the `.ipynb` file yourself.

## Workflow

Execute in this order:

1. Read `02_notebook_structure.json` for the full cell list and ordering.
2. Read `03_cell_analysis.json` for code-cell implementation details.
3. For each cell in order:
   - **markdown cells:** write `source` text that fulfills the cell `goal` from stage 2.
   - **code cells:** write `source` code per `implementation_plan`, `function_signatures`, `error_handling`, and `test_checks` from stage 3.
4. Write `04_cell_sources.json` with every cell's `source` (see schema below).
5. Return `04_generation_report.json` documenting every cell (see schema below).

## Cell Sources Schema (`04_cell_sources.json`)

```json
{
  "topic": "kvcache",
  "notebook_title": "KV Cache: Efficient LLM Inference — Interactive Demo",
  "source_artifacts": {
    "structure": "pipeline_outputs/02_notebook_structure.json",
    "analysis": "pipeline_outputs/03_cell_analysis.json"
  },
  "cells": [
    {
      "cell_id": "C01",
      "cell_type": "markdown",
      "source": "# Title\n\n...",
      "generation_notes": "Title block per C01 goal."
    },
    {
      "cell_id": "C02",
      "cell_type": "code",
      "source": "!pip install -q anthropic\n\nimport numpy as np\n...",
      "generation_notes": "Setup imports per C02 spec."
    }
  ],
  "assumptions": []
}
```

### Cell source rules

- `cells` must match `02.cells` in count, order, `cell_id`, and `cell_type`.
- `source` is plain text (use `\n` for newlines), not an ipynb JSON fragment.
- Every `source` must be non-empty.
- Code cells may use `!pip` and `%matplotlib inline` where needed for Colab.

## Generation Report Schema (`04_generation_report.json`)

```json
{
  "final_notebook_path": "notebooks/<topic>_interactive_skill.ipynb",
  "generated_cells": [
    {
      "cell_id": "C01",
      "status": "generated",
      "notes": ""
    }
  ],
  "execution_status": {
    "top_to_bottom_runnable": true,
    "failed_cell_ids": []
  },
  "dependency_notes": [],
  "assumptions": []
}
```

## Output Requirements

- Cell sources must be complete enough for Colab execution after assembly
- Widget callbacks and visual outputs must work as intended
- `generated_cells` must list **every** cell from stage 2, in the same order
- Do **not** use one-off `scripts/legacy/gen_*_notebook.py` scripts

## Completion Gate

- `04_cell_sources.json` written to disk with all required keys
- `cells` length and `cell_id` order match `02_notebook_structure.json`
- `generated_cells` length and `cell_id` order match `02_notebook_structure.json`
- `final_notebook_path` matches `notebooks/<topic>_interactive_skill.ipynb`
- Report includes any unresolved assumptions or limitations
