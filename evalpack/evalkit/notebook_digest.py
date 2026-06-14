"""Serialize a notebook into the evidence the quality judges read: the full
content as TEXT (markdown + code + textual outputs) plus the rendered visual
outputs as image files. We never screenshot text/code (lossy); the source is
lossless (docs/EVAL_TESTSET_DESIGN.md §6 "Judge inputs").

Output coverage matters: a judge can only score what it can see. Besides
image/png we extract image/jpeg, reduce text/html to text, and explicitly mark
ipywidgets widget-view outputs (whose live output is NOT stored in the .ipynb)
so a judge knows to rely on the interaction frames instead of concluding the
notebook "has no output".
"""
from __future__ import annotations

import base64
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from textbudget import truncate_middle


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return s if isinstance(s, str) else "".join(s)


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):  # noqa: ANN001
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):  # noqa: ANN001
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):  # noqa: ANN001
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def _html_to_text(html: str, max_chars: int) -> str:
    p = _HTMLText()
    try:
        p.feed(html)
        txt = " ".join(p.parts)
    except Exception:  # noqa: BLE001
        txt = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", txt).strip()[:max_chars]


def digest(nb_path: str | Path, image_dir: str | Path,
           max_out_chars: int = 1500,
           budget: int | None = None) -> tuple[str, list[tuple[str, str]]]:
    """Return (text_digest, [(image_path, origin_label)]).

    Image outputs stored in the .ipynb are written to files under image_dir;
    origin_label names the producing cell (e.g. "output of code cell 14") so
    callers can burn it onto the image / build a citation manifest.

    `budget` (chars) is enforced HONESTLY: markdown + code are always kept
    whole; if over budget, per-output excerpts shrink first (1500 -> 400), and
    only then is the digest cut from the middle with a visible marker — never
    a silent prefix cut that hides the notebook's tail from the judge.
    """
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8", errors="ignore"))
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    images: list[tuple[str, str]] = []

    def _build(out_chars: int, save_images: bool) -> str:
        lines: list[str] = []
        img_i = 0

        def _save_image(b64: str, ext: str, cell_i: int) -> None:
            nonlocal img_i
            p = image_dir / f"nbout_{img_i:02d}_cell{cell_i:02d}.{ext}"
            if save_images:
                p.write_bytes(base64.b64decode(b64))
                images.append((str(p), f"output of code cell {cell_i}"))
            lines.append(f"[image output of cell {cell_i} -> {p.name}]")
            img_i += 1

        for i, cell in enumerate(nb.get("cells", [])):
            ct = cell.get("cell_type")
            src = _src(cell).rstrip()
            if ct == "markdown":
                lines.append(f"### [markdown cell {i}]\n{src}")
            elif ct == "code":
                lines.append(f"### [code cell {i}]\n```python\n{src}\n```")
                for out in cell.get("outputs", []):
                    ot = out.get("output_type")
                    if ot == "stream":
                        txt = out.get("text", "")
                        txt = txt if isinstance(txt, str) else "".join(txt)
                        lines.append(f"[stdout]\n{txt[:out_chars]}")
                    elif ot in ("execute_result", "display_data"):
                        data = out.get("data", {}) or {}
                        if "image/png" in data:
                            _save_image(data["image/png"], "png", i)
                        elif "image/jpeg" in data:
                            _save_image(data["image/jpeg"], "jpeg", i)
                        elif "application/vnd.jupyter.widget-view+json" in data:
                            lines.append(
                                "[ipywidgets widget view — its live rendered "
                                "output is NOT stored in the .ipynb; see the "
                                "attached interaction frames for what this "
                                "widget shows]")
                        elif "image/svg+xml" in data:
                            svg = data["image/svg+xml"]
                            svg = svg if isinstance(svg, str) else "".join(svg)
                            lines.append(f"[svg output (not rendered); source "
                                         f"head]\n{svg[:600]}")
                        elif "text/html" in data:
                            html = data["text/html"]
                            html = html if isinstance(html, str) else "".join(html)
                            lines.append(f"[html output, reduced to text]\n"
                                         f"{_html_to_text(html, out_chars)}")
                        elif "text/plain" in data:
                            txt = data["text/plain"]
                            txt = txt if isinstance(txt, str) else "".join(txt)
                            lines.append(f"[result]\n{txt[:out_chars]}")
                    elif ot == "error":
                        tb = "\n".join(out.get("traceback", []))
                        lines.append(f"[error]\n{tb[:out_chars]}")
        return "\n\n".join(lines)

    text = _build(max_out_chars, save_images=True)
    if budget is not None and len(text) > budget:
        text = _build(min(400, max_out_chars), save_images=False)
        if len(text) > budget:
            text = truncate_middle(text, budget, "the notebook")
    return text, images
