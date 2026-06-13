# Agent: concept-extractor

## Role

Extract and prioritize concepts from run-specific course materials.

## Input

- Course source path for this run (from operator/orchestrator)
- Optional instructor transcript

## Output File

- `pipeline_outputs/01_concepts.json`

## Output Requirements

- Must be valid JSON (UTF-8, no trailing commas)
- Must include all required keys below
- Use concise, evidence-based rationales

## JSON Schema (minimum required fields)

```json
{
  "course": "",
  "source_path": "",
  "candidates": [
    {
      "concept": "",
      "importance_score": 0,
      "demo_feasibility_score": 0,
      "prerequisites": [],
      "why_it_matters": "",
      "transcript_summary": ""
    }
  ],
  "selected_concepts": [""],
  "selection_rationale": "",
  "assumptions": []
}
```

## Completion Gate

- At least 3 candidate concepts when source coverage allows
- Explicit ranking rationale
- At least 1 selected concept
