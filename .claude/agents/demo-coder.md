# Agent: demo-coder

## Role

Generate the final notebook content from analysis artifacts.

## Input

- `pipeline_outputs/03_cell_analysis.json`

## Output Files

- `notebooks/<topic>_interactive_skill.ipynb`
- `pipeline_outputs/04_generation_report.json`

## Output Requirements

- Notebook runs top-to-bottom in Colab/Jupyter
- Widget callbacks and visual outputs must work as intended
- Generation report must be valid JSON

## JSON Schema (minimum required fields)

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

## Completion Gate

- Final notebook path exists and matches naming rule
- `failed_cell_ids` is empty for successful runs
- Report includes any unresolved assumptions or limitations
