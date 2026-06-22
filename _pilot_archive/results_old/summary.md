# Ablation summary (10 runs)

## Executability (objective harness)

| Condition | n | exec_raw | exec_quality | env/timeout |
|---|--:|--:|--:|--:|
| S0 | 2 | 1.0 | 1.0 | 0 |
| ablate-concept-extractor | 2 | 0.0 | 0.0 | 0 |
| ablate-notebook-architect | 2 | 0.5 | 0.5 | 0 |
| ablate-cell-analyzer | 2 | 0.0 | 0.0 | 1 |
| B | 2 | 0.0 | 0.0 | 1 |

## Judge scores (1-5, mean across topics x seeds x judges)

| Condition | overall | executability | concept correctness | interactivity | visualization quality | pedagogical value | alignment with source | robustness | simplicity maintainability |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| S0 | None | None | None | None | None | None | None | None | None |
| ablate-concept-extractor | None | None | None | None | None | None | None | None | None |
| ablate-notebook-architect | None | None | None | None | None | None | None | None | None |
| ablate-cell-analyzer | None | None | None | None | None | None | None | None | None |
| B | None | None | None | None | None | None | None | None | None |

