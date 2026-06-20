#!/usr/bin/env bash
# Build a Google-Colab-matching Python runtime and register it as the `colab`
# Jupyter kernel that Stage 5 (notebook_verifier) uses by default.
#
# Requires a Python 3.12 interpreter (Colab runs 3.12.13). Override with PYTHON_BIN.
# The venv location can be overridden with COLAB_VENV.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${COLAB_VENV:-$HOME/.colab-runtime-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PYV="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$PYV" != "3.12" ]; then
  echo "[setup_colab_kernel] WARNING: PYTHON_BIN is Python $PYV, but Colab runs 3.12." >&2
  echo "[setup_colab_kernel] Set PYTHON_BIN to a 3.12 interpreter for an exact match." >&2
fi

echo "[setup_colab_kernel] Creating venv at $VENV"
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$HERE/colab-runtime-requirements.txt"

"$VENV/bin/python" -m ipykernel install --user \
  --name colab \
  --display-name "Colab runtime (py3.12, numpy2.0.2)"

echo "[setup_colab_kernel] Registered Jupyter kernel: colab"

# Pre-warm: build the matplotlib font cache and trigger first-imports now, so the
# first Stage 5 verification isn't penalised (a cold font-cache build can exceed
# the per-cell timeout and look like a hang).
echo "[setup_colab_kernel] Pre-warming caches (matplotlib fonts, first imports)..."
"$VENV/bin/python" - <<'PY'
import sys, numpy, scipy, pandas, sklearn, ipywidgets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(); plt.plot([0, 1], [0, 1]); plt.title("warm"); fig.canvas.draw(); plt.close(fig)
print("[setup_colab_kernel] python", sys.version.split()[0],
      "| numpy", numpy.__version__,
      "| scipy", scipy.__version__,
      "| pandas", pandas.__version__,
      "| sklearn", sklearn.__version__,
      "| ipywidgets", ipywidgets.__version__)
PY
