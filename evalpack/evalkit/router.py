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
    NONE = "none"


# Lane each type is actuated through (Axis A).
LANE = {
    InteractivityType.IPYWIDGETS: "kernel",     # Layer 0, observe in browser
    InteractivityType.COLAB_PARAM: "param",     # kernel param-injection
    InteractivityType.SERVER_APP: "browser",    # Layer 1 DOM
    InteractivityType.EMBED_HTML: "browser",    # Layer 1 DOM
    InteractivityType.CANVAS_WEBGL: "vision",   # Layer 2 (v2)
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
    InteractivityType.NONE,
]


@dataclass
class DetectionResult:
    primary: InteractivityType
    types: list[InteractivityType] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)

    @property
    def lane(self) -> str:
        return LANE[self.primary]


def _code_source(nb_path: Path) -> str:
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8", errors="ignore"))
    parts: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            parts.append(src if isinstance(src, str) else "".join(src))
    return "\n".join(parts)


def detect(nb_path: str | Path) -> DetectionResult:
    code = _code_source(Path(nb_path))
    found_types: set[InteractivityType] = set()
    signals: list[str] = []
    for itype, name, pat in _SIGNALS:
        if re.search(pat, code):
            found_types.add(itype)
            signals.append(name)
    if not found_types:
        return DetectionResult(InteractivityType.NONE, [InteractivityType.NONE], [])
    primary = next(t for t in _PRIORITY if t in found_types)
    ordered = [t for t in _PRIORITY if t in found_types]
    return DetectionResult(primary=primary, types=ordered, signals=signals)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        r = detect(p)
        print(f"{p}\n  primary={r.primary.value} lane={r.lane} "
              f"types={[t.value for t in r.types]} signals={r.signals}")
