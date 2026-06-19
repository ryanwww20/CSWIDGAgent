# Project: Colab Demo Generator Pipeline

## Goal

Transform course materials into classroom-ready, interactive Colab notebooks that are executable and pedagogically useful.

## Main Pipeline (5-stage)

1. Concept Extracting Agent
2. Notebook Structure Agent
3. Cell-level Analysis Agent
4. Demo Code Agent
5. Notebook Verification (syntax + execution, with LLM auto-fix)

Each stage must read the previous stage artifact and write its own output under `pipeline_outputs/`.

## Required Output Files

- `pipeline_outputs/01_concepts.json`
- `pipeline_outputs/02_notebook_structure.json`
- `pipeline_outputs/03_cell_analysis.json`
- `pipeline_outputs/04_cell_sources.json`
- `pipeline_outputs/04_generation_report.json`
- `pipeline_outputs/05_execution_report.json`
- `pipeline_outputs/run_log.json`
- final notebook under `notebooks/` (assembled by `notebook_assembler.py`, verified by `notebook_verifier.py`)

## Workflow Rule

Always execute in order:

1. `concept-extractor`
2. `notebook-architect`
3. `cell-analyzer`
4. `demo-coder`
5. `notebook-verifier` (deterministic verify + `notebook-fixer` repair loop)

Do not skip stages unless explicitly requested.

## Quality Priorities

1. Executability
2. Concept correctness
3. Interactivity
4. Visualization quality
5. Pedagogical value
6. Alignment with source material
7. Robustness
8. Simplicity and maintainability

## Run Configuration

- Course source is run-specific.
- Default root: `course_source/`
- Official entry point: `./scripts/run_pipeline.sh --source <path> --topic <topic>`
- Record per-run source path and outputs in `pipeline_outputs/run_log.json`.
- Final notebook naming convention: `<topic>_interactive_skill.ipynb`.
- See `docs/pipeline.md` for architecture, `generation_mode`, and legacy script policy.

## Stage 4 Contract (demo-coder + assembler)

Stage 4 is **artifact-driven** in two steps:

1. **demo-coder** reads `02` + `03`, writes `pipeline_outputs/04_cell_sources.json`, returns `04_generation_report.json`.
2. **notebook_assembler.py** reads `02` + `04_cell_sources.json`, writes `notebooks/<topic>_interactive_skill.ipynb`.

Do **not** use one-off `scripts/legacy/gen_*_notebook.py` scripts for new runs. Legacy scripts are reference-only.

See `docs/phase2_cell_sources_and_assembler.md` for schema and interface details.

## Stage 5 Contract (notebook-verifier + notebook-fixer)

Stage 5 is **deterministic-first, repair-second**:

1. **notebook_verifier.py** (pure Python, no LLM) reads the assembled `.ipynb` and `02`,
   runs a per-code-cell syntax check (IPython-magic aware) then a top-to-bottom
   `nbclient` execution, and writes `pipeline_outputs/05_execution_report.json`.
2. If the notebook fails (syntax or execution) and auto-fix is enabled, **notebook-fixer**
   (LLM agent) reads `05_execution_report.json` + `04_cell_sources.json` and repairs the
   failing cells in place. The runner then re-assembles via `notebook_assembler.py` and
   re-verifies. This loops up to `--max-fix-attempts` (default 2).

`05_execution_report.json` is the authoritative source of `top_to_bottom_runnable` in
`run_log.json` (real kernel execution, not the demo-coder self-report).

Stage 5 flags: `--skip-verify`, `--no-autofix`, `--max-fix-attempts N`, `--cell-timeout`,
`--startup-timeout`. Use `--from-stage 5` to re-verify an already-assembled notebook.

`run_log.json` must include `generation_mode`:

- `artifact_driven` — notebook produced from pipeline artifacts (required for new runs).
- `legacy_script` — historical one-off script (deprecated).

## Agent Instruction Files

JSON schemas and stage-specific output contracts are maintained in:

- `.claude/agents/concept-extractor.md`
- `.claude/agents/notebook-architect.md`
- `.claude/agents/cell-analyzer.md`
- `.claude/agents/demo-coder.md`
- `.claude/agents/notebook-fixer.md`