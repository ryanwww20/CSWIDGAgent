#!/usr/bin/env bash
# Resume the abc ablation on THIS (shared, old-libstdc++) node.
#
# Differences vs launch_detached.sh:
#   1. exports LD_LIBRARY_PATH=$CONDA_PREFIX/lib  -> the env's own libstdc++ wins
#      over the node's stale /lib64 (fixes "GLIBCXX_3.4.29 not found" that made
#      evalkit scoring + the capped python3 kernel fail). Inherited by claude
#      generation, evalkit run_eval.py, and the spawned jupyter kernel alike.
#   2. concurrency capped at MAXJOBS=3 (was 6 full-parallel) -> avoids the
#      simultaneous-claude-generation OOM that SIGKILLed the 20260627 shards.
# Resume skips (cond,topic,seed) whose notebook parses AND has an evalkit summary.
set -uo pipefail
cd /work/b12901015/CSWIDGAgent

PFX=/work/b12901015/miniconda3/envs/ml-colab-eval
export LD_LIBRARY_PATH="$PFX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHON_BIN="$PFX/bin/python" EVALKIT_PY="$PFX/bin/python" KERNEL_NAME=python3
export STAGE_VALIDATION_ATTEMPTS=8
export EVALKIT_STAGES=abc EVALKIT_JUDGE_SAMPLES=3 SEEDS=3
export TOPICS="MLBasic:MLBasic.pdf whydeeplearning:whydeeplearning.pdf \
ml2021_self_attention:ml2021_self_attention.pdf autoencoder:autoencoder.pdf \
gan:gan.pdf MetaLearning:MetaLearning.pdf DRL:DRL.pdf"

CONDS=(S0 ablate-concept-extractor ablate-notebook-architect ablate-cell-analyzer B "B+bug_solver")
MAXJOBS="${MAXJOBS:-3}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "$TS" > logs/ablation/.run_ts
echo "[resume] ts=$TS  maxjobs=$MAXJOBS  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

for cond in "${CONDS[@]}"; do
  # gate: keep at most MAXJOBS condition-shards running at once
  while (( $(jobs -rp | wc -l) >= MAXJOBS )); do wait -n; done
  safe=$(echo "$cond" | tr '+' 'p')
  log="logs/ablation/shard_${safe}_${TS}.log"
  ( CONDITIONS="$cond" bash ./scripts/run_ablation.sh
    echo "=== SHARD $cond EXITED rc=$? $(date -u) ===" ) > "$log" 2>&1 < /dev/null &
  echo "[resume] launched shard: $cond -> $log (pid $!)"
done
wait
echo "[resume] ALL shards finished (ts=$TS)"
