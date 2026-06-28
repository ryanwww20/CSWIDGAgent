#!/usr/bin/env bash
# Launch the full abc ablation as 6 detached shards (setsid nohup so they survive
# session/tmux teardown). One condition per shard; per-(cond,topic,seed) outputs
# are isolated. Resume skips already-done runs. Per-stage retry is in pipeline_runner.
set -uo pipefail
cd /work/b12901015/CSWIDGAgent

EVAL_ENV_PY="/work/b12901015/miniconda3/envs/ml-colab-eval/bin/python"
TOPICS_FULL="MLBasic:MLBasic.pdf whydeeplearning:whydeeplearning.pdf \
ml2021_self_attention:ml2021_self_attention.pdf autoencoder:autoencoder.pdf \
gan:gan.pdf MetaLearning:MetaLearning.pdf DRL:DRL.pdf"
CONDS=(S0 ablate-concept-extractor ablate-notebook-architect ablate-cell-analyzer B "B+bug_solver")
TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "$TS" > logs/ablation/.run_ts

for cond in "${CONDS[@]}"; do
  safe=$(echo "$cond" | tr '+' 'p')
  log="logs/ablation/shard_${safe}_${TS}.log"
  setsid nohup env \
    PYTHON_BIN="$EVAL_ENV_PY" EVALKIT_PY="$EVAL_ENV_PY" KERNEL_NAME=python3 \
    STAGE_VALIDATION_ATTEMPTS=8 \
    EVALKIT_STAGES=abc EVALKIT_JUDGE_SAMPLES=3 SEEDS=3 \
    TOPICS="$TOPICS_FULL" CONDITIONS="$cond" \
    bash -c './scripts/run_ablation.sh; echo "=== SHARD '"$cond"' EXITED rc=$? '"$(date -u)"' ==="' \
    > "$log" 2>&1 < /dev/null &
  echo "launched detached shard: $cond -> $log (pid $!)"
done
echo "all 6 detached shards launched (ts=$TS)"
