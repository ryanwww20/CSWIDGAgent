# Ablation Run Plan (MLBasic-only, OpenAI-only quality)

Operational runbook for the agent-ablation study. Pick up from here at any time —
the orchestrator is resumable (it skips any `(condition, topic, seed)` whose
notebook already parses).

## Locked decisions

- **Ablation mode = merge/absorb (M):** a dropped agent's job folds into its
  downstream neighbor. demo-coder is non-ablatable (it's the only cell-content
  generator) — that's itself a finding.
- **Quality track = evalkit** (the senior's eval pack), NOT the built-in LLM judge.
  `evalkit/run_eval.py` → 7-metric vector at
  `evalkit/results/<condition>/<topic>__s<seed>/summary.json`.
- **Scope right now = MLBasic only** (torch-free → executes cleanly on CPU). The
  other default topics (gan/autoencoder/adversarial/xformer) import torch, which
  is NOT installed in the `ml-colab-eval` env, so they'd fail `#1 run_success` as
  an *environment* artifact. Add them once torch is available (see "Unblock ML topics").
- **Quality judging = OpenAI-only (degraded anti-bias).** `ANTHROPIC_API_KEY` is
  empty in `evalkit/.env`, so planner/judge/verifier all fall back to
  `gpt-5.4-mini`. Fill the Anthropic key later to restore cross-family anti-bias.

## Conditions (agent count low → high)

| Condition | What it is |
|---|---|
| `S0` | single-shot baseline, 0 agents (`prompt/claude_1shot.md`) |
| `ablate-concept-extractor` | drop CE, merge into notebook-architect |
| `ablate-notebook-architect` | drop NA, merge into cell-analyzer |
| `ablate-cell-analyzer` | drop CA, merge into demo-coder |
| `B` | full 4-stage pipeline (baseline for paired deltas) |
| `B+bug_solver` | full + Stage-5 verify/autofix (proxy for the teammate's 5th agent) |

## Harness state (fixed 2026-06-21)

- `scripts/run_ablation.sh` — `EVALKIT=1` path scores each notebook via evalkit
  under the `ml-colab-eval` conda env; runs the eval from `evalkit/` so
  `find_dotenv` picks up `evalkit/.env`. Missing-PDF guard skips bad topics loudly.
- `scripts/lib/aggregate_evalkit.py` — `evalkit/results/*` →
  `results/evalkit_runs.csv` + `results/evalkit_by_condition.md`
  (per-condition mean + **paired delta vs B**).
- Contaminated pilot archived under `_pilot_archive/` (reversible).

## The sequence

Run from repo root. `EVALKIT_PY` defaults to
`/Users/ryan/miniconda3/envs/ml-colab-eval/bin/python`.

### Step 1 — free exec smoke (0 API for evalkit; validates the clean pipeline)
```bash
EVALKIT_NO_LLM=1 EVALKIT_STAGES=a \
TOPICS="MLBasic:MLBasic.pdf" CONDITIONS="S0 ablate-cell-analyzer B" SEEDS=1 \
./scripts/run_ablation.sh
```

### Step 2 — quality pilot, seed 1, all conditions (PAID: claude gen + OpenAI judges)
```bash
EVALKIT_STAGES=abc EVALKIT_JUDGE_SAMPLES=1 \
TOPICS="MLBasic:MLBasic.pdf" \
CONDITIONS="S0 ablate-concept-extractor ablate-notebook-architect ablate-cell-analyzer B B+bug_solver" \
SEEDS=1 ./scripts/run_ablation.sh
```
Confirm `results/evalkit_by_condition.md` has non-dash quality columns before scaling.

### Step 3 — full, 3 seeds (paired deltas get statistical weight)
```bash
EVALKIT_STAGES=abc EVALKIT_JUDGE_SAMPLES=3 \
TOPICS="MLBasic:MLBasic.pdf" \
CONDITIONS="S0 ablate-concept-extractor ablate-notebook-architect ablate-cell-analyzer B B+bug_solver" \
SEEDS=3 ./scripts/run_ablation.sh
```

## Reading results

- `results/evalkit_by_condition.md` — headline table. The **paired-delta-vs-B**
  block is the ablation result: a metric that drops when an agent is removed =
  that agent adds value; flat/positive = collapsing it is fine.
- `results/evalkit_runs.csv` — one row per run for re-analysis.
- Per-notebook detail: `evalkit/results/<cond>/MLBasic__s<seed>/summary.json`.

## Resume / re-run

- Re-running the same command **skips** finished `(cond, topic, seed)` notebooks
  (resume). Force regeneration with `RESUME=0`.
- For a fully clean re-measure: `rm -rf runs results` first (both git-ignored).

## Unblock ML topics later (torch)

Either install CPU torch into the eval env:
```bash
/Users/ryan/miniconda3/envs/ml-colab-eval/bin/pip install torch
```
…or point evalkit at a torch-capable python and widen the matrix:
```bash
EVALKIT_PY=/path/to/gpu/python \
TOPICS="MLBasic:MLBasic.pdf autoencoder:autoencoder.pdf gan:gan.pdf adversarial_attack:Adversarial_attack.pdf xformer:xformer.pdf" \
... ./scripts/run_ablation.sh
```

## Restore full anti-bias judging later

Fill `ANTHROPIC_API_KEY` in `evalkit/.env`. evalkit then keeps judge vs verifier
in different model families automatically (claude-sonnet-4-6 vs gpt-5.4-mini).
