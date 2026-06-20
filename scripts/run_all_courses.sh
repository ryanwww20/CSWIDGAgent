#!/usr/bin/env bash
#
# Batch-run the 4-stage pipeline over every PDF in course_source/,
# excluding a fixed skip-list. Each run uses the PDF's filename (without
# the .pdf extension) as its --topic, so every final notebook is named
# <stem>_interactive_skill.ipynb and they never overwrite each other.
#
# Runs are SEQUENTIAL on purpose: every run reuses the same shared
# intermediate artifacts (pipeline_outputs/01..04_*.json), so running
# them in parallel would corrupt each other's state.
#
# Usage:
#   ./scripts/run_all_courses.sh            # process all eligible PDFs
#   ./scripts/run_all_courses.sh --dry-run  # forwarded to run_pipeline.sh
#
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$ROOT_DIR/course_source"
RUNNER="$ROOT_DIR/scripts/run_pipeline.sh"

# Case-insensitive filename stems to skip (note: KVCache file is "KVcahce.pdf").
SKIP=("mlbasic" "kvcache" "kvcahce" "dijkstra")

is_skipped() {
  local stem_lc="$1"
  for s in "${SKIP[@]}"; do
    [[ "$stem_lc" == "$s" ]] && return 0
  done
  return 1
}

EXTRA_ARGS=("$@")

processed=()
succeeded=()
failed=()

shopt -s nullglob
pdfs=("$SOURCE_DIR"/*.pdf)
shopt -u nullglob

if [[ ${#pdfs[@]} -eq 0 ]]; then
  echo "No PDFs found in $SOURCE_DIR" >&2
  exit 1
fi

for pdf in "${pdfs[@]}"; do
  base="$(basename "$pdf")"
  stem="${base%.pdf}"
  stem_lc="$(echo "$stem" | tr '[:upper:]' '[:lower:]')"

  if is_skipped "$stem_lc"; then
    echo "==> SKIP $base"
    continue
  fi

  echo ""
  echo "============================================================"
  echo "==> RUN  source=$base  topic=$stem"
  echo "============================================================"
  processed+=("$stem")

  if "$RUNNER" --source "course_source/$base" --topic "$stem" "${EXTRA_ARGS[@]}"; then
    succeeded+=("$stem")
    echo "==> DONE $stem"
  else
    rc=$?
    failed+=("$stem")
    echo "==> FAIL $stem (exit $rc)" >&2
  fi
done

echo ""
echo "============================================================"
echo "Batch summary: ${#processed[@]} processed, ${#succeeded[@]} succeeded, ${#failed[@]} failed"
echo "  succeeded: ${succeeded[*]:-(none)}"
echo "  failed:    ${failed[*]:-(none)}"
echo "============================================================"

[[ ${#failed[@]} -eq 0 ]]
