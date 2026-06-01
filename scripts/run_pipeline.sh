#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="$ROOT_DIR/pipeline_outputs"
NOTEBOOK_DIR="$ROOT_DIR/notebooks"

SOURCE_PATH=""
TOPIC=""
RUN_ID=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_pipeline.sh --source <course_source_path> --topic <topic_name> [--run-id <id>] [--dry-run]

Examples:
  ./scripts/run_pipeline.sh --source course_source/Dijkstra.pdf --topic dijkstra
  ./scripts/run_pipeline.sh --source course_source/kvcache --topic kvcache --run-id demo-001

Environment overrides (optional):
  CONCEPT_CMD_TEMPLATE
  STRUCTURE_CMD_TEMPLATE
  ANALYSIS_CMD_TEMPLATE
  CODE_CMD_TEMPLATE

Template placeholders:
  {{SOURCE_PATH}}, {{TOPIC}}, {{ROOT_DIR}}
EOF
}

log() {
  printf '[run_pipeline] %s\n' "$1"
}

die() {
  printf '[run_pipeline][ERROR] %s\n' "$1" >&2
  exit 1
}

json_validate() {
  local file_path="$1"
  python3 - <<PY
import json
from pathlib import Path
path = Path(r"$file_path")
if not path.exists():
    raise SystemExit(f"missing file: {path}")
with path.open("r", encoding="utf-8") as f:
    json.load(f)
print(f"valid json: {path}")
PY
}

replace_placeholders() {
  local template="$1"
  template="${template//'{{SOURCE_PATH}}'/$SOURCE_PATH}"
  template="${template//'{{TOPIC}}'/$TOPIC}"
  template="${template//'{{ROOT_DIR}}'/$ROOT_DIR}"
  printf '%s' "$template"
}

run_stage() {
  local stage_name="$1"
  local cmd="$2"

  log "Running stage: $stage_name"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run][%s] %s\n' "$stage_name" "$cmd"
    return 0
  fi

  bash -lc "$cmd"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_PATH="${2:-}"
      shift 2
      ;;
    --topic)
      TOPIC="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$SOURCE_PATH" ]] || die "--source is required"
[[ -n "$TOPIC" ]] || die "--topic is required"

if [[ "$SOURCE_PATH" != /* ]]; then
  SOURCE_PATH="$ROOT_DIR/$SOURCE_PATH"
fi
[[ -e "$SOURCE_PATH" ]] || die "Source path does not exist: $SOURCE_PATH"

mkdir -p "$PIPELINE_DIR" "$NOTEBOOK_DIR"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)"
fi

TIMESTAMP_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

DEFAULT_CONCEPT_TEMPLATE='claude -p "You are agent concept-extractor. Follow {{ROOT_DIR}}/.claude/agents/concept-extractor.md. Analyze source: {{SOURCE_PATH}} and write output to {{ROOT_DIR}}/pipeline_outputs/01_concepts.json. Return only a short completion note."'
DEFAULT_STRUCTURE_TEMPLATE='claude -p "You are agent notebook-architect. Follow {{ROOT_DIR}}/.claude/agents/notebook-architect.md. Read {{ROOT_DIR}}/pipeline_outputs/01_concepts.json and write {{ROOT_DIR}}/pipeline_outputs/02_notebook_structure.json. Return only a short completion note."'
DEFAULT_ANALYSIS_TEMPLATE='claude -p "You are agent cell-analyzer. Follow {{ROOT_DIR}}/.claude/agents/cell-analyzer.md. Read {{ROOT_DIR}}/pipeline_outputs/02_notebook_structure.json and write {{ROOT_DIR}}/pipeline_outputs/03_cell_analysis.json. Return only a short completion note."'
DEFAULT_CODE_TEMPLATE='claude -p "You are agent demo-coder. Follow {{ROOT_DIR}}/.claude/agents/demo-coder.md. Read {{ROOT_DIR}}/pipeline_outputs/03_cell_analysis.json. Generate notebook for topic {{TOPIC}} and write {{ROOT_DIR}}/pipeline_outputs/04_generation_report.json. Return only a short completion note."'

CONCEPT_TEMPLATE="${CONCEPT_CMD_TEMPLATE:-$DEFAULT_CONCEPT_TEMPLATE}"
STRUCTURE_TEMPLATE="${STRUCTURE_CMD_TEMPLATE:-$DEFAULT_STRUCTURE_TEMPLATE}"
ANALYSIS_TEMPLATE="${ANALYSIS_CMD_TEMPLATE:-$DEFAULT_ANALYSIS_TEMPLATE}"
CODE_TEMPLATE="${CODE_CMD_TEMPLATE:-$DEFAULT_CODE_TEMPLATE}"

CONCEPT_CMD="$(replace_placeholders "$CONCEPT_TEMPLATE")"
STRUCTURE_CMD="$(replace_placeholders "$STRUCTURE_TEMPLATE")"
ANALYSIS_CMD="$(replace_placeholders "$ANALYSIS_TEMPLATE")"
CODE_CMD="$(replace_placeholders "$CODE_TEMPLATE")"

run_stage "concept-extractor" "$CONCEPT_CMD"
[[ "$DRY_RUN" -eq 1 ]] || json_validate "$PIPELINE_DIR/01_concepts.json"

run_stage "notebook-architect" "$STRUCTURE_CMD"
[[ "$DRY_RUN" -eq 1 ]] || json_validate "$PIPELINE_DIR/02_notebook_structure.json"

run_stage "cell-analyzer" "$ANALYSIS_CMD"
[[ "$DRY_RUN" -eq 1 ]] || json_validate "$PIPELINE_DIR/03_cell_analysis.json"

run_stage "demo-coder" "$CODE_CMD"
[[ "$DRY_RUN" -eq 1 ]] || json_validate "$PIPELINE_DIR/04_generation_report.json"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Dry-run complete."
  exit 0
fi

python3 - <<PY
import json
from pathlib import Path

root = Path(r"$ROOT_DIR")
pipe = root / "pipeline_outputs"
report_path = pipe / "04_generation_report.json"
run_log_path = pipe / "run_log.json"
topic = "$TOPIC"
run_id = "$RUN_ID"
timestamp_utc = "$TIMESTAMP_UTC"
source_path = r"$SOURCE_PATH"

report = json.loads(report_path.read_text(encoding="utf-8"))
final_notebook = report.get("final_notebook_path", f"notebooks/{topic}_interactive_skill.ipynb")
runnable = report.get("execution_status", {}).get("top_to_bottom_runnable", False)

run_log = {
    "run_id": run_id,
    "timestamp_utc": timestamp_utc,
    "course_source_path": source_path,
    "selected_concept": topic,
    "stage_status": {
        "concept_extractor": "completed",
        "notebook_architect": "completed",
        "cell_analyzer": "completed",
        "demo_coder": "completed"
    },
    "artifacts": {
        "concepts": "pipeline_outputs/01_concepts.json",
        "structure": "pipeline_outputs/02_notebook_structure.json",
        "analysis": "pipeline_outputs/03_cell_analysis.json",
        "generation_report": "pipeline_outputs/04_generation_report.json",
        "final_notebook": final_notebook
    },
    "summary": {
        "top_to_bottom_runnable": bool(runnable)
    },
    "errors": []
}

run_log_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"wrote: {run_log_path}")
PY

json_validate "$PIPELINE_DIR/run_log.json"

log "Pipeline completed successfully."
log "Run ID: $RUN_ID"
log "Run log: $PIPELINE_DIR/run_log.json"
