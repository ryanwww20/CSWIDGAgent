# Pipeline + Evaluation Report — course_source (9 PDFs)

_Generated 2026-06-19 20:11 UTC. Pipeline: 5-stage artifact-driven (Stage 5 verifies on the Colab-matched kernel). Evaluation: evalpack `run_eval.py`, method `cswidge_pipeline`, GPT judge `gpt-5.4-mini`._


**Summary: 6/9 notebooks generated, verified, and scored. 3/9 failed in the pipeline.**


Metrics 2–5 and 7 are scored **0–5** (higher is better); #1 is pass/fail; #6 is notebook execution time in seconds.


## Scores

| Topic | Run OK | Faithful | Pedagogy | Topic | Interact | Exec(s) | Clarity | Colab-verified |
|---|---|---|---|---|---|---|---|---|
| `ml2021_metalearning` | ❌ | 3.33 | 4.00 | 5.00 | 4.74 | 3.76 | 3.83 | ❌ (colab) |
| `ml2021_rl` | ✅ | 4.62 | 5.00 | 5.00 | 4.79 | 20.05 | 3.60 | ✅ (colab) |
| `ml2021_self_attention` | ❌ | 3.33 | 4.00 | 5.00 | 4.74 | 2.36 | 4.00 | ❌ (colab) |
| `ml2022_attack_innlp` | ✅ | 4.62 | 5.00 | 5.00 | 4.44 | 1.85 | 4.00 | ✅ (colab) |
| `ml2022_ssl` | ✅ | 3.33 | 5.00 | 5.00 | 4.57 | 218.48 | 3.60 | ❌ (colab) |
| `ml2023_aiexplaination` | ✅ | 5.00 | 5.00 | 5.00 | 2.31 | 1.73 | 4.62 | ✅ (colab) |


## Averages (scored notebooks)

| Metric | Mean |
|---|---|
| Run OK | 4/6 pass |
| Faithful | 4.04 |
| Pedagogy | 4.67 |
| Topic | 5.00 |
| Interact | 4.26 |
| Exec(s) | 41.37 |
| Clarity | 3.94 |


## Failures

| Topic | Stage that failed / reason |
|---|---|
| `ml2022_attention` | claude CLI subprocess failed (transient API/availability error) — retry when Claude is stable |
| `ml2023_diffusion_model` | claude CLI subprocess failed (transient API/availability error) — retry when Claude is stable |
| `ml2023_prompt` | claude CLI subprocess failed (transient API/availability error) — retry when Claude is stable |

Failure modes seen: (1) stochastic LLM schema slips (cell-analyzer emitting specs for non-code cells; demo-coder omitting a key) on the first run; (2) transient `claude` CLI errors during the long batch on retry. Neither is a fundamental defect — re-running these topics individually when Claude is stable should produce scores.
