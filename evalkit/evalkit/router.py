"""Detect how a demo notebook realizes interactivity, and route it to a lane.

Pure-stdlib, static scan of the .ipynb (no execution). A notebook can trip
several signals; we report all of them and pick a primary lane by priority.
See docs/EVAL_TESTSET_DESIGN.md §6.3 for the routing table.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class InteractivityType(str, Enum):
    IPYWIDGETS = "ipywidgets"
    COLAB_PARAM = "colab_param"
    SERVER_APP = "server_app"
    EMBED_HTML = "embed_html"
    CANVAS_WEBGL = "canvas_webgl"
    CODE_TWEAK = "code_tweak"   # v2: no widgets, but runnable code with a tweakable surface
    NONE = "none"


# Lane each type is actuated through (Axis A). v2: #@param and the generic
# code-tweak surface are both driven IN-KERNEL by the code lane (symbol override +
# downstream-slice re-run), so they route to "kernel", not the never-wired "param".
LANE = {
    InteractivityType.IPYWIDGETS: "kernel",     # Layer 0, observe headlessly
    InteractivityType.COLAB_PARAM: "kernel",    # v2 code lane (#@param static parse)
    InteractivityType.SERVER_APP: "browser",    # Layer 1 DOM (not wired yet)
    InteractivityType.EMBED_HTML: "browser",    # Layer 1 DOM (not wired yet)
    InteractivityType.CANVAS_WEBGL: "kernel",   # v2: view-only -> reasoned in kernel lane
    InteractivityType.CODE_TWEAK: "kernel",     # v2: planner-found semantic surface
    InteractivityType.NONE: "none",
}

# (signal name, regex) — matched against concatenated code-cell source.
_SIGNALS: list[tuple[InteractivityType, str, str]] = [
    (InteractivityType.COLAB_PARAM, "colab_param", r"#@param"),
    (InteractivityType.IPYWIDGETS, "ipywidgets",
     r"\bimport\s+ipywidgets\b|\bfrom\s+ipywidgets\b|\binteract(?:ive)?\s*\(|\bwidgets\.(?:FloatSlider|IntSlider|Dropdown|Button|Checkbox|interactive)"),
    (InteractivityType.SERVER_APP, "gradio", r"\bimport\s+gradio\b|gr\.(?:Interface|Blocks)|\.launch\s*\("),
    (InteractivityType.SERVER_APP, "streamlit", r"\bimport\s+streamlit\b|\bst\.\w+\("),
    (InteractivityType.SERVER_APP, "flask", r"\bFlask\s*\(|app\.run\s*\("),
    (InteractivityType.SERVER_APP, "panel_bokeh", r"\.servable\s*\(|\bpn\.serve\b|\bbokeh\s+serve\b"),
    (InteractivityType.SERVER_APP, "dash", r"\bimport\s+dash\b|Dash\s*\("),
    (InteractivityType.EMBED_HTML, "html_js",
     r"display\(\s*HTML\(|IPython\.display\.HTML\(|<script\b"),
    (InteractivityType.CANVAS_WEBGL, "threejs_canvas",
     r"\bpythreejs\b|\bipycanvas\b|\bbqplot\b|\bipyleaflet\b"),
    (InteractivityType.CANVAS_WEBGL, "mpl_widget", r"%matplotlib\s+(?:widget|notebook|ipympl)"),
    (InteractivityType.CANVAS_WEBGL, "plotly_interactive", r"\bimport\s+plotly\b|plotly\.graph_objects|px\."),
]

# Lower number = higher priority when several types are present.
_PRIORITY = [
    InteractivityType.IPYWIDGETS,
    InteractivityType.SERVER_APP,
    InteractivityType.EMBED_HTML,
    InteractivityType.CANVAS_WEBGL,
    InteractivityType.COLAB_PARAM,
    InteractivityType.CODE_TWEAK,
    InteractivityType.NONE,
]


@dataclass
class DetectionResult:
    primary: InteractivityType
    types: list[InteractivityType] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    code_cell_count: int = 0

    @property
    def lane(self) -> str:
        return LANE[self.primary]

    @property
    def runnable(self) -> bool:
        """v2: anything with runnable code has a *potential* tweakable surface
        (the planner decides if it's real), so the kernel code lane applies."""
        return self.code_cell_count > 0


def _code_cells(nb_path: Path) -> list[str]:
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8", errors="ignore"))
    out: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            src = src if isinstance(src, str) else "".join(src)
            if src.strip():
                out.append(src)
    return out


def detect(nb_path: str | Path) -> DetectionResult:
    cells = _code_cells(Path(nb_path))
    code = "\n".join(cells)
    n_code = len(cells)
    found_types: set[InteractivityType] = set()
    signals: list[str] = []
    for itype, name, pat in _SIGNALS:
        if re.search(pat, code):
            found_types.add(itype)
            signals.append(name)
    if not found_types:
        # v2: not 'none' just because there are no widgets — a runnable notebook
        # may still expose signposted constants/inputs the planner can find.
        primary = (InteractivityType.CODE_TWEAK if n_code
                   else InteractivityType.NONE)
        return DetectionResult(primary, [primary], [], code_cell_count=n_code)
    if n_code:
        found_types.add(InteractivityType.CODE_TWEAK)  # code lane always available
        signals.append("code_tweak")
    primary = next(t for t in _PRIORITY if t in found_types)
    ordered = [t for t in _PRIORITY if t in found_types]
    return DetectionResult(primary=primary, types=ordered, signals=signals,
                           code_cell_count=n_code)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        r = detect(p)
        print(f"{p}\n  primary={r.primary.value} lane={r.lane} "
              f"types={[t.value for t in r.types]} signals={r.signals}")
