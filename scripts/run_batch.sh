#!/usr/bin/env bash
# Batch: run the full pipeline + evaluation for every PDF in course_source/.
# Resumable (skips a topic whose eval summary already exists) and fault-tolerant
# (one failing PDF does not abort the batch). Per-topic logs + artifact snapshots
# land in $OUT for the report builder to read.
set -uo pipefail

ROOT="/Users/justin/Desktop/NTU/CSWIDGAgent"
cd "$ROOT"

# torch (pip) and the conda-built eval env each link their own libomp on macOS,
# which aborts on `import torch` (OMP: Error #15). This documented workaround lets
# both runtimes coexist; safe for our execute-and-check use. Inherited by the
# Stage 5 colab kernel and the eval subprocess.
export KMP_DUPLICATE_LIB_OK=TRUE

OUT="${BATCH_OUT:-/tmp/batch_results}"
mkdir -p "$OUT"
PROGRESS="$OUT/progress.txt"

echo "batch started $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$PROGRESS"

for pdf in course_source/*.pdf; do
  stem="$(basename "$pdf" .pdf)"
  topic="$(echo "$stem" | tr '[:upper:]' '[:lower:]')"
  nb="notebooks/${topic}_interactive_skill.ipynb"
  summary="evalpack/results/cswidge_pipeline/${topic}-run1/summary.json"

  if [ -f "$summary" ]; then
    echo "[$topic] SKIP (eval summary already exists)" | tee -a "$PROGRESS"
    continue
  fi

  echo "[$topic] pipeline START $(date -u +%H:%M:%SZ)" | tee -a "$PROGRESS"
  ./scripts/run_pipeline.sh --source "$pdf" --topic "$topic" \
      --max-fix-attempts 1 --cell-timeout 240 --startup-timeout 90 \
      > "$OUT/${topic}.pipeline.log" 2>&1
  pipe_rc=$?
  # Snapshot per-topic pipeline artifacts before the next run overwrites them.
  cp pipeline_outputs/05_execution_report.json "$OUT/${topic}.05.json" 2>/dev/null
  cp pipeline_outputs/run_log.json            "$OUT/${topic}.runlog.json" 2>/dev/null
  echo "[$topic] pipeline DONE rc=$pipe_rc $(date -u +%H:%M:%SZ)" | tee -a "$PROGRESS"

  if [ -f "$nb" ]; then
    echo "[$topic] eval START $(date -u +%H:%M:%SZ)" | tee -a "$PROGRESS"
    ( cd evalpack && conda run -n ml-colab-eval python evalkit/run_eval.py \
        "../$nb" --slides "../$pdf" \
        --method cswidge_pipeline --run-id "${topic}-run1" \
        > "$OUT/${topic}.eval.log" 2>&1 )
    eval_rc=$?
    echo "[$topic] eval DONE rc=$eval_rc $(date -u +%H:%M:%SZ)" | tee -a "$PROGRESS"
  else
    echo "[$topic] NO NOTEBOOK — skipping eval" | tee -a "$PROGRESS"
  fi
done

echo "BATCH_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$PROGRESS"
