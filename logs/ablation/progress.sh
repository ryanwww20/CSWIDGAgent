#!/usr/bin/env bash
# Quick ablation progress check. Run from repo root: bash logs/ablation/progress.sh
cd "$(dirname "$0")/../.." || exit 1
TOPICS="MLBasic whydeeplearning ml2021_self_attention autoencoder gan MetaLearning DRL"
CONDS=(S0 ablate-concept-extractor ablate-notebook-architect ablate-cell-analyzer B "B+bug_solver")
echo "cond                          gen  scored   (of 21 each)"
G=0; E=0
for c in "${CONDS[@]}"; do
  g=0; e=0
  for t in $TOPICS; do for s in 1 2 3; do
    ls "runs/${c}__${t}__s${s}/notebooks/"*.ipynb >/dev/null 2>&1 && { g=$((g+1)); G=$((G+1)); }
    [ -f "evalkit/results/$c/${t}__s${s}/summary.json" ] && { e=$((e+1)); E=$((E+1)); }
  done; done
  printf "%-28s %3d  %3d\n" "$c" "$g" "$e"
done
printf "%-28s %3d  %3d   /126\n" "TOTAL" "$G" "$E"
echo
echo "running shards: $(ps -eo cmd | grep -E 'scripts/run_ablation' | grep -v grep | wc -l)"
echo "recent kills:   $(grep -l Killed logs/ablation/shard_*_$(cat logs/ablation/.run_ts 2>/dev/null).log 2>/dev/null | wc -l) shard log(s) with a Killed line this run"