# Legacy Scripts

Scripts in this directory are **deprecated** and not part of the official pipeline.

## `gen_kvcache_notebook.py`

- **Purpose:** One-off generator for `notebooks/kvcache_interactive_skill.ipynb`.
- **Problem:** Cell content is hard-coded (~1200 lines). It does not read `pipeline_outputs/03_cell_analysis.json`.
- **Status:** Kept as a golden reference for the initial KV Cache delivery.
- **Do not use** for new runs. Use `./scripts/run_pipeline.sh` with artifact-driven stage 4 instead.

To regenerate the legacy notebook manually (not recommended):

```bash
python3 scripts/legacy/gen_kvcache_notebook.py
```
