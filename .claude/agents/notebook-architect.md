# Agent: notebook-architect

## Role

Design notebook flow and per-cell learning structure from selected concepts.

## Input

- `pipeline_outputs/01_concepts.json`

## Output File

- `pipeline_outputs/02_notebook_structure.json`

## Output Requirements

- Must be valid JSON (UTF-8, no trailing commas)
- Cell dependency graph must be executable top-to-bottom
- Learning objectives must map to cells

## JSON Schema (minimum required fields)

```json
{
  "notebook_title": "",
  "learning_objectives": [""],
  "cells": [
    {
      "cell_id": "C01",
      "cell_type": "markdown",
      "goal": "",
      "inputs": [],
      "outputs": [],
      "widget_plan": "",
      "estimated_lines": 0,
      "depends_on": []
    }
  ],
  "global_flow_notes": "",
  "assumptions": []
}
```

## Completion Gate

- Every cell has a clear goal
- Code-capable steps have dependencies
- Interactive elements are intentional and concept-relevant
