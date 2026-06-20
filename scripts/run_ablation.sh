#!/usr/bin/env bash
#
# Ablation-study orchestrator.
#
# For every (condition x topic x seed) it generates a notebook into an ISOLATED
# runs/<tag>/ dir, then runs the objective execution harness and the LLM judge,
# and finally aggregates everything into results/.
#
# Conditions (agent count low -> high):
#   S0                          single-shot baseline (0 agents, prompt/claude_1shot.md)
#   ablate-concept-extractor    drop CE  (merge into notebook-architect)
#   ablate-notebook-architect   drop NA  (merge into cell-analyzer)
#   ablate-cell-analyzer        drop CA  (merge into demo-coder)
#   B                           full 4-stage pipeline (baseline)
#   B+bug_solver                full + the teammate's 5th agent (needs a hook; see below)
#
# Override the matrix with env vars, e.g.:
#   TOPICS="dijkstra:Dijkstra.pdf gan:gan.pdf" CONDITIONS="B ablate-cell-analyzer" \
#   SEEDS=2 JUDGES=3 ./scripts/run_ablation.sh
#
# bug_solver: not in the repo yet. If scripts/lib/bug_solver_hook.sh exists it is
# invoked as `bug_solver_hook.sh <notebook> <run_dir>` after assembly; otherwise
# the B+bug_solver condition is SKIPPED with a notice.
#
# Pass --dry-run to forward to the pipeline runner (prints prompts, no API calls).
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
RUNNER="$ROOT_DIR/scripts/run_pipeline.sh"
RUNS_DIR="$ROOT_DIR/runs"
DRY_RUN="${DRY_RUN:-}"
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="--dry-run"

# topic:pdf pairs (pdf is relative to course_source/). Defaults span algorithm,
# basic ML, generative, robustness, and architecture topics — all API-free.
TOPICS="${TOPICS:-dijkstra:Dijkstra.pdf MLBasic:MLBasic.pdf autoencoder:autoencoder.pdf adversarial_attack:Adversarial_attack.pdf xformer:xformer.pdf}"
CONDITIONS="${CONDITIONS:-S0 ablate-concept-extractor ablate-notebook-architect ablate-cell-analyzer B B+bug_solver}"
SEEDS="${SEEDS:-3}"
# Judging is OFF by default (JUDGES=0): evaluation is handled downstream by a
# separate (senior's) scorer. Set JUDGES>0 to also run the built-in LLM judge.
JUDGES="${JUDGES:-0}"
PER_CELL_TIMEOUT="${PER_CELL_TIMEOUT:-300}"
BUG_SOLVER_HOOK="$ROOT_DIR/scripts/lib/bug_solver_hook.sh"

note() { echo "[ablation] $*"; }

# A run counts as already-done if its notebook parses and has a few cells.
# Lets the orchestrator resume after an interruption (RESUME=0 to force redo).
nb_ok() {
  [[ -f "$1" ]] || return 1
  "$PYTHON_BIN" -c "import json,sys
try:
    d=json.load(open(sys.argv[1]))
    sys.exit(0 if isinstance(d.get('cells'),list) and len(d['cells'])>=3 else 1)
except Exception:
    sys.exit(1)" "$1" 2>/dev/null
}

# Map a condition name to the pipeline runner's --ablate value ("" if none).
ablate_flag() {
  case "$1" in
    ablate-concept-extractor)  echo "concept-extractor" ;;
    ablate-notebook-architect) echo "notebook-architect" ;;
    ablate-cell-analyzer)      echo "cell-analyzer" ;;
    *)                         echo "" ;;
  esac
}

# S0: single-shot generation straight from the source PDF, no pipeline.
run_singleshot() {
  local topic="$1" pdf="$2" nb="$3"
  local base; base="$(cat "$ROOT_DIR/prompt/claude_1shot.md")"
  local prompt="${base}

The course material is at: course_source/${pdf}
Read it, then write the complete notebook as a valid .ipynb JSON file to:
  ${nb}
Create any parent directories. Do not print the notebook; just write the file."
  if [[ -n "$DRY_RUN" ]]; then
    echo "[dry-run] single-shot -> $nb"
    return 0
  fi
  ( cd "$ROOT_DIR" && "$CLAUDE_BIN" -p "$prompt" >/dev/null )
}

processed=0; generated=0; skipped=0
for cond in $CONDITIONS; do
  for pair in $TOPICS; do
    topic="${pair%%:*}"; pdf="${pair#*:}"
    for seed in $(seq 1 "$SEEDS"); do
      tag="${cond}__${topic}__s${seed}"
      run_dir="$RUNS_DIR/$tag"
      nb="$run_dir/notebooks/${topic}_interactive_skill.ipynb"
      processed=$((processed+1))
      note "=== $tag ==="

      # Resume: skip runs that already produced a valid notebook.
      if [[ -z "$DRY_RUN" && "${RESUME:-1}" == "1" ]] && nb_ok "$nb"; then
        note "SKIP $tag (notebook already present — resume)"
        skipped=$((skipped+1)); continue
      fi

      # bug_solver condition needs the teammate's hook; skip cleanly until it lands.
      if [[ "$cond" == "B+bug_solver" && ! -x "$BUG_SOLVER_HOOK" && -z "$DRY_RUN" ]]; then
        note "SKIP $tag (no bug_solver hook at $BUG_SOLVER_HOOK)"
        skipped=$((skipped+1))
        continue
      fi

      mkdir -p "$run_dir"
      ok=1
      if [[ "$cond" == "S0" ]]; then
        run_singleshot "$topic" "$pdf" "$nb" || ok=0
      else
        flag="$(ablate_flag "$cond")"
        args=(--source "course_source/${pdf}" --topic "$topic" --run-tag "$tag" --claude-bin "$CLAUDE_BIN")
        [[ -n "$flag" ]] && args+=(--ablate "$flag")
        [[ -n "$DRY_RUN" ]] && args+=(--dry-run)
        "$RUNNER" "${args[@]}" || ok=0
        # bug_solver runs on the assembled notebook, in place.
        if [[ "$cond" == "B+bug_solver" && -x "$BUG_SOLVER_HOOK" && -z "$DRY_RUN" ]]; then
          "$BUG_SOLVER_HOOK" "$nb" "$run_dir" || note "bug_solver hook failed for $tag"
        fi
      fi

      # meta.json (consumed by aggregate_results.py)
      "$PYTHON_BIN" - "$run_dir" "$tag" "$cond" "$topic" "$seed" "$pdf" <<'PY'
import json, sys
run_dir, tag, cond, topic, seed, pdf = sys.argv[1:7]
import pathlib
pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
(pathlib.Path(run_dir) / "meta.json").write_text(json.dumps({
    "run_tag": tag, "condition": cond, "topic": topic,
    "seed": int(seed), "source_pdf": pdf,
}, indent=2) + "\n", encoding="utf-8")
PY

      [[ -n "$DRY_RUN" ]] && continue
      if [[ "$ok" -ne 1 || ! -f "$nb" ]]; then
        note "generation failed/no notebook for $tag; recording empty exec result"
        printf '{"notebook":"%s","ran_to_completion":false,"error_type":"generation_failed","error_name":null,"error_value":null,"first_failed_cell_index":null,"first_failed_code_cell":null,"n_code_cells":0,"executed_code_cells":0,"wall_seconds":0.0}\n' "$nb" > "$run_dir/execution_result.json"
        continue
      fi
      generated=$((generated+1))

      # Objective executability (exit 2 on real failure — don't abort the batch).
      "$PYTHON_BIN" "$ROOT_DIR/scripts/lib/execute_notebook.py" \
        --notebook "$nb" --output "$run_dir/execution_result.json" \
        --per-cell-timeout "$PER_CELL_TIMEOUT" --save-executed "$run_dir/executed.ipynb" \
        >/dev/null || true

      # Blind LLM judge (skipped when JUDGES=0 — evaluation handled downstream).
      if [[ "$JUDGES" -gt 0 ]]; then
        "$PYTHON_BIN" "$ROOT_DIR/scripts/lib/judge_notebook.py" \
          --notebook "$nb" --source "course_source/${pdf}" --judges "$JUDGES" \
          --claude-bin "$CLAUDE_BIN" --output "$run_dir/judge_result.json" \
          >/dev/null || note "judge failed for $tag"
      fi
    done
  done
done

note "generated $generated / processed $processed (skipped $skipped)"

if [[ -z "$DRY_RUN" ]]; then
  note "aggregating..."
  ( cd "$ROOT_DIR" && "$PYTHON_BIN" scripts/lib/aggregate_results.py --runs-dir runs --out-dir results ) || true
  note "done. See results/summary.md"
fi
