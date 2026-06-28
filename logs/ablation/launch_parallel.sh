#!/usr/bin/env bash
# Launch the full abc ablation as 6 parallel shards (one condition each),
# each in its own tmux window of session 'ablation'. Per-(cond,topic,seed)
# outputs are isolated, so shards never collide. Final global aggregate is
# done separately after all shards finish (per-shard aggregates are throwaway).
set -uo pipefail
cd /work/b12901015/CSWIDGAgent

EVAL_ENV_PY="/work/b12901015/miniconda3/envs/ml-colab-eval/bin/python"
export PYTHON_BIN="$EVAL_ENV_PY" EVALKIT_PY="$EVAL_ENV_PY" KERNEL_NAME="python3"

TOPICS_FULL="MLBasic:MLBasic.pdf whydeeplearning:whydeeplearning.pdf \
ml2021_self_attention:ml2021_self_attention.pdf autoencoder:autoencoder.pdf \
gan:gan.pdf MetaLearning:MetaLearning.pdf DRL:DRL.pdf"
CONDS=(S0 ablate-concept-extractor ablate-notebook-architect ablate-cell-analyzer B "B+bug_solver")
TS=$(date -u +%Y%m%dT%H%M%SZ)

tmux kill-session -t ablation 2>/dev/null || true
tmux new-session -d -s ablation -n "${CONDS[0]}"
first=1
for cond in "${CONDS[@]}"; do
  safe=$(echo "$cond" | tr '+' 'p')
  log="logs/ablation/shard_${safe}_${TS}.log"
  cmd="cd /work/b12901015/CSWIDGAgent; \
export PYTHON_BIN='$EVAL_ENV_PY' EVALKIT_PY='$EVAL_ENV_PY' KERNEL_NAME=python3; \
EVALKIT_STAGES=abc EVALKIT_JUDGE_SAMPLES=3 SEEDS=3 \
TOPICS=\"$TOPICS_FULL\" CONDITIONS=\"$cond\" \
./scripts/run_ablation.sh 2>&1 | tee '$log'; \
echo \"=== SHARD $cond EXITED rc=\$? $(date -u) ===\""
  if [ "$first" = 1 ]; then
    tmux send-keys -t ablation:0 "$cmd" C-m
    tmux rename-window -t ablation:0 "$safe"
    first=0
  else
    tmux new-window -t ablation -n "$safe"
    tmux send-keys -t "ablation:$safe" "$cmd" C-m
  fi
  echo "launched shard: $cond -> $log"
done
echo "all 6 shards launched in tmux session 'ablation' (windows: one per condition)"
