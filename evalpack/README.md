# Demo Evaluation Pack

Everything needed to evaluate generated interactive teaching notebooks on the
7-metric vector — the task definition, the evaluation pipeline, and the test
instances — in one place.

```
evalpack/
├── TASK_DEFINITION.md     what each method must produce (the task contract)
├── PROMPT_TEMPLATE.txt    the input prompt to give every method, verbatim
├── evalkit/               the evaluation pipeline (run_eval.py = entry point)
├── configs/               execution profiles (cpu-stable = official numbers)
├── environment/           conda env definition
├── instances/             eval instances: starter decks + full-testset manifest
└── results/               every run's artifacts land here (see its README)
```

## Setup (once)

```bash
conda env create -f environment/conda-env.yml
conda activate ml-colab-eval
cp .env.template .env        # then fill in your API keys
```

The judges/planner/verifier are auto-picked from whichever keys you provide and
kept in **different model families** (anti-bias). Override with
`EVAL_JUDGE_MODEL` / `EVAL_PLANNER_MODEL` / `EVAL_VERIFIER_MODEL` in `.env`.

## Evaluate a notebook (one command)

```bash
python evalkit/run_eval.py path/to/demo.ipynb \
    --slides instances/starter/KVcahce.pdf \
    --method my_method
```

Or pick a test-set instance by id (see `instances/README.md` for the data root):

```bash
python evalkit/run_eval.py path/to/demo.ipynb \
    --instance HTLIN_ML__05_handout --data-root /path/to/ml_colab \
    --method my_method
```

Useful flags: `--no-llm` (deterministic parts only, zero API cost),
`--stages ab` (skip the quality judges), `--run-id NAME`,
`--digest-slides` (auto-build the once-per-deck vision digest for image-heavy
decks), `--transcript T.txt`.

## What a run produces

`results/<method>/<run_id>/` (see `results/README.md`):

| artifact | content |
|---|---|
| `summary.json` | the 7-metric vector + provenance (hashes, models, exit codes) |
| `exec/report.json`, `exec/executed.ipynb` | #1 Run Success, #6 Efficiency |
| `interactivity/` | filmstrip, PNG frames, blind-judge verdict, #5 score |
| `quality/quality_report.json` | #2/#3/#4/#7 with sub-scores, verified errors, citations |
| `quality/judge_images/` | the labeled evidence images the judges actually saw |
| `llm_calls.jsonl` | **every LLM exchange, verbatim** (prompts, replies, usage, retries) — a scored run is fully auditable without re-running |

## The 7 metrics (report the vector, not a composite)

| # | Metric | How |
|---|--------|-----|
| 1 | Run Success | headless execution, canonical cpu-stable profile |
| 2 | Faithfulness & Correctness | judge (2a/2b/2c) + independent verifier; harmonic |
| 3 | Pedagogical Depth | slide-aware judge: marginal value over the slide |
| 4 | Topic Significance | slide-aware judge: was the chosen concept worth demoing |
| 5 | Interactivity | deterministic widget actuation + blind judge; harmonic(effectiveness, robustness) |
| 6 | Efficiency | runtime (cpu-stable numbers are the official ones) |
| 7 | Clarity | slides-blind judge (7a visual / 7b textual / 7c code); harmonic |

Design principles baked into the pipeline:

- **Actuation never judges; judges never actuate.** Controls are driven by
  deterministic code; LLMs only read the recorded evidence.
- **Evidence is labeled.** Every image a judge sees carries a burned-in
  `IMAGE k | origin` banner + manifest, so citations are checkable.
- **Burden of proof.** Sub-5 faithfulness deductions must cite concrete errors,
  each independently confirmed by a different-family verifier model.
- **NA, never fake zeros.** A failed/unparseable judge call is retried with
  corrective feedback, then reported as NA — it never contaminates scores.
- **No silent truncation.** Anything cut from judge input carries a visible
  `[TRUNCATED ...]` marker, and judges are instructed not to score what they
  can't see.
- **Image-heavy decks**: build a once-per-deck vision digest
  (`evalkit/slide_digest.py`) so judges see full slide content at ~zero
  marginal cost per evaluation.

## Scoring multiple outputs (pass@k)

Run each of the k outputs through `run_eval.py` (distinct `--run-id`s).
Run Success is the rate over k; score #2–#7 on the first successful output
(NA if none pass). Compare methods on aggregate per-instance vectors —
comparison is unpaired by design (see TASK_DEFINITION.md → Scoring).
