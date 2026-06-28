#!/usr/bin/env bash
# LOCAL (macOS Apple Silicon) resume launcher for the abc ablation.
#
# Differences vs the SSH launch_resume.sh:
#   * paths are DERIVED (repo root from script location, env python from conda) —
#     no hard-coded /work/... SSH paths.
#   * NO LD_LIBRARY_PATH export — that fixed a Linux-node stale-libstdc++ issue
#     that does not exist on macOS.
#   * MAXJOBS defaults to 2 (32 GB RAM). Lower to 1 if the machine swaps.
#   * KERNEL_NAME=python3 must resolve to a kernelspec pointing at THIS env — see
#     LOCAL_MIGRATION.md step "register kernel". macOS ulimit -v is unreliable so
#     there is no 32 GiB cap; PER_CELL_TIMEOUT guards runaway cells instead.
# Resume skips (cond,topic,seed) whose notebook parses AND has an evalkit summary.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-ml-colab-eval}"
PFX="${ENV_PREFIX:-$(conda info --base 2>/dev/null)/envs/$ENV_NAME}"
PYBIN="$PFX/bin/python"
[ -x "$PYBIN" ] || { echo "[local] env python not found: $PYBIN — create env from logs/ablation/environment.yml first"; exit 1; }

export PYTHON_BIN="$PYBIN" EVALKIT_PY="$PYBIN" KERNEL_NAME="${KERNEL_NAME:-python3}"
export STAGE_VALIDATION_ATTEMPTS="${STAGE_VALIDATION_ATTEMPTS:-8}"
export EVALKIT_STAGES="${EVALKIT_STAGES:-abc}" EVALKIT_JUDGE_SAMPLES="${EVALKIT_JUDGE_SAMPLES:-3}" SEEDS="${SEEDS:-3}"
export PER_CELL_TIMEOUT="${PER_CELL_TIMEOUT:-300}"
export TOPICS="${TOPICS:-MLBasic:MLBasic.pdf whydeeplearning:whydeeplearning.pdf \
ml2021_self_attention:ml2021_self_attention.pdf autoencoder:autoencoder.pdf \
gan:gan.pdf MetaLearning:MetaLearning.pdf DRL:DRL.pdf}"

CONDS=(S0 ablate-concept-extractor ablate-notebook-architect ablate-cell-analyzer B "B+bug_solver")
MAXJOBS="${MAXJOBS:-2}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "$TS" > logs/ablation/.run_ts
echo "[local] ts=$TS  maxjobs=$MAXJOBS  env=$PYBIN"

for cond in "${CONDS[@]}"; do
  while (( $(jobs -rp | wc -l) >= MAXJOBS )); do wait -n; done
  safe=$(echo "$cond" | tr '+' 'p')
  log="logs/ablation/shard_${safe}_${TS}.log"
  ( CONDITIONS="$cond" bash ./scripts/run_ablation.sh
    echo "=== SHARD $cond EXITED rc=$? $(date -u) ===" ) > "$log" 2>&1 < /dev/null &
  echo "[local] launched shard: $cond -> $log (pid $!)"
done
wait
echo "[local] ALL shards finished (ts=$TS)"
