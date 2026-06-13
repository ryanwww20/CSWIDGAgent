# Agent: cell-analyzer

## Role

Produce implementation-level specs for each planned code cell.

## Input

- `pipeline_outputs/02_notebook_structure.json`

## Output File

- `pipeline_outputs/03_cell_analysis.json`

## Output Requirements

- Must be valid JSON (UTF-8, no trailing commas)
- Cover all code cells from stage 2
- Implementation plans must be explicit enough for direct coding

## JSON Schema (minimum required fields)

```json
{
  "cell_specs": [
    {
      "cell_id": "C01",
      "implementation_plan": "",
      "function_signatures": [""],
      "state_variables": [""],
      "error_handling": [""],
      "test_checks": [""]
    }
  ],
  "cross_cell_invariants": [""],
  "assumptions": []
}
```

## Completion Gate

- No ambiguous placeholders like "implement as needed"
- Includes failure handling for non-trivial steps
- Includes sanity checks that can be executed in notebook context
