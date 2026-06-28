#!/usr/bin/env bash
# Stable ablation-study driver (tmux-launched). Mode = $1: gen3 | abc
set -uo pipefail
cd /work/b12901015/CSWIDGAgent

EVAL_ENV_PY="/work/b12901015/miniconda3/envs/ml-colab-eval/bin/python"
export PYTHON_BIN="$EVAL_ENV_PY"
export EVALKIT_PY="$EVAL_ENV_PY"
export KERNEL_NAME="python3"   # 32 GiB-capped kernel

TOPICS_FULL="MLBasic:MLBasic.pdf whydeeplearning:whydeeplearning.pdf \
ml2021_self_attention:ml2021_self_attention.pdf autoencoder:autoencoder.pdf \
gan:gan.pdf MetaLearning:MetaLearning.pdf DRL:DRL.pdf"
CONDS_FULL="S0 ablate-concept-extractor ablate-notebook-architect ablate-cell-analyzer B B+bug_solver"

case "${1:-gen3}" in
  gen3)  # all generation + exec score, 3 seeds, NO OpenAI (marked no_llm -> re-scored by abc)
    EVALKIT_NO_LLM=1 EVALKIT_STAGES=a SEEDS=3 \
    TOPICS="$TOPICS_FULL" CONDITIONS="$CONDS_FULL" ./scripts/run_ablation.sh ;;
  abc)   # full quality pass, 3 seeds, judge-samples=3 (resumes: skips gen, adds judging)
    EVALKIT_STAGES=abc EVALKIT_JUDGE_SAMPLES=3 SEEDS=3 \
    TOPICS="$TOPICS_FULL" CONDITIONS="$CONDS_FULL" ./scripts/run_ablation.sh ;;
  *) echo "usage: $0 {gen3|abc}"; exit 1 ;;
esac
echo "=== run_study $1 EXITED rc=$? at $(date -u) ==="
