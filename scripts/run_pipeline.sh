#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/lib/pipeline_runner.py" \
  --root-dir "$ROOT_DIR" \
  --claude-bin "$CLAUDE_BIN" \
  "$@"
