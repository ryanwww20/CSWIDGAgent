"""Static dataflow over a notebook's code cells (pure-AST, no execution).

Two jobs for the v2 #5 code lane:
  1. `forward_slice` — given the cell that defines a tweakable symbol, find the
     downstream cells that consume it and produce the displayed output. That slice
     is what code_controls re-runs after injecting a perturbed value (the defining
     cell already ran at baseline, so we override the symbol, not re-patch it).
  2. `parse_colab_params` — statically enumerate Colab `#@param` form fields, the
     deterministic half of the non-widget surface (the planner handles the rest).

Conservative by construction: a cell that fails to parse contributes no
assigns/uses (it simply won't extend a slice), and forward taint *over*-includes
rather than under-includes — re-running an extra independent cell is safe, missing
a dependent one is not.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CellInfo:
    index: int                                  # notebook cell index
    source: str
    assigns: set[str] = field(default_factory=set)   # names bound at top level
    uses: set[str] = field(default_factory=set)       # names loaded anywhere
    displays: bool = False                       # produces visible output (plot/print/expr)


# Heuristic markers that a cell renders something a learner would look at — used
# to tell "the slice reaches a displayed output" (wired-in) from a dead tweak.
_DISPLAY_RE = re.compile(
    r"\b(?:plt|sns|px|go|fig)\.|\.show\s*\(|\.plot\s*\(|display\s*\(|print\s*\(|"
    r"imshow|matshow|\.savefig\s*\(|plt\.figure\b")


def _name_ids(nodes) -> set[str]:
    out: set[str] = set()
    for n in nodes:
        if isinstance(n, ast.Name):
            out.add(n.id)
    return out


def _analyze(source: str) -> tuple[set[str], set[str]]:
    """(assigns, uses) for one cell. Best-effort; unparseable -> ({}, {})."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(), set()
    assigns: set[str] = set()
    uses: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (assigns if isinstance(node.ctx, ast.Store) else uses).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assigns.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                assigns.add((a.asname or a.name).split(".")[0])
    return assigns, uses


def parse_cells(nb_path: str | Path) -> list[CellInfo]:
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8", errors="ignore"))
    cells: list[CellInfo] = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src = src if isinstance(src, str) else "".join(src)
        if not src.strip():
            continue
        assigns, uses = _analyze(src)
        cells.append(CellInfo(index=i, source=src, assigns=assigns, uses=uses,
                              displays=bool(_DISPLAY_RE.search(src))))
    return cells


def find_symbol_cell(cells: list[CellInfo], symbol: str) -> int | None:
    """Index of the (last) code cell that assigns `symbol`, or None."""
    found = [c.index for c in cells if symbol in c.assigns]
    return found[-1] if found else None


def forward_slice(cells: list[CellInfo], start_index: int, symbol: str) -> list[int]:
    """Cells AFTER `start_index` that (transitively) consume `symbol` — the
    dependent cells of a perturbation. Forward taint: a consuming cell's own
    assignments join the taint; a cell that REASSIGNS a tainted name without using
    it kills that taint (the value is overwritten independently of our
    perturbation — e.g. `epochs` redefined in a later section)."""
    tainted = {symbol}
    sliced: list[int] = []
    for c in cells:
        if c.index <= start_index:
            continue
        if c.uses & tainted:
            sliced.append(c.index)
            tainted |= c.assigns
        else:
            tainted -= (c.assigns - c.uses)   # independent reassignment kills taint
        if not tainted:
            break
    return sliced


def rerun_range(cells: list[CellInfo], start_index: int, symbol: str) -> list[int]:
    """The CONTIGUOUS cell range to re-execute for a perturbation: from the
    defining cell through the last dependent cell, INCLUDING the intermediate
    cells in between (a dependent cell's own setup may live in a cell that does
    not itself consume the symbol). Re-running the whole span keeps state
    consistent; we snapshot after the last cell so the captured output is the
    control-relevant one, not a later unrelated section."""
    dependents = forward_slice(cells, start_index, symbol)
    if not dependents:
        return [start_index] if any(c.index == start_index for c in cells) else []
    last = max(dependents)
    return [c.index for c in cells if start_index <= c.index <= last]


_ASSIGN_RE_CACHE: dict[str, "re.Pattern"] = {}


def patch_assignment(source: str, symbol: str, value) -> str | None:
    """Rewrite the first top-level `symbol = <rhs>` line to assign `value`,
    preserving indentation and any trailing comment. Returns the patched cell
    source, or None if there's no simple assignment to patch (caller then falls
    back to a kernel-level value injection). This lets us re-run the DEFINING cell
    with the new value so same-cell consumers (a training loop in the same cell)
    and freshly-built objects (an optimizer) pick the perturbation up."""
    pat = _ASSIGN_RE_CACHE.get(symbol)
    if pat is None:
        pat = re.compile(rf"^(?P<indent>[ \t]*){re.escape(symbol)}[ \t]*="
                         rf"[ \t]*[^#\n]*?(?P<comment>[ \t]*#.*)?$")
        _ASSIGN_RE_CACHE[symbol] = pat
    lines = source.splitlines()
    for i, line in enumerate(lines):
        m = pat.match(line)
        if m:
            indent = m.group("indent")
            comment = (m.group("comment") or "").strip()
            tail = f"  {comment}" if comment else ""
            lines[i] = f"{indent}{symbol} = {value!r}{tail}"
            return "\n".join(lines)
    return None


def slice_reaches_display(cells: list[CellInfo], slice_indices: list[int]) -> bool:
    """dataflow_only signal: does the perturbed symbol flow into a cell that
    actually renders output? A slice with no displaying cell is a dead tweak."""
    by_index = {c.index: c for c in cells}
    return any(by_index[i].displays for i in slice_indices if i in by_index)


# --- Colab #@param static parse --------------------------------------------

_PARAM_LINE = re.compile(
    r"^[ \t]*([A-Za-z_]\w*)[ \t]*=[ \t]*(.+?)[ \t]*#@param\b[ \t]*(.*)$")


def _literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except Exception:  # noqa: BLE001
        return text.strip().strip("'\"")


def _param_values(default: Any, spec: str) -> list[Any]:
    """Perturbation values for a #@param field from its form spec.
      ["a","b",...]   -> the listed options
      {type:slider/number, min, max} -> [min, mid, max]
      {type:boolean}  -> [True, False]
      {type:string}   -> [] (freeform; no deterministic sweep -> reasoned)
    """
    spec = spec.strip()
    if spec.startswith("["):
        try:
            opts = ast.literal_eval(spec)
            return [o for o in opts if o != default] or list(opts)
        except Exception:  # noqa: BLE001
            return []
    if isinstance(default, bool):
        return [not default, default]
    if "boolean" in spec:
        return [True, False]
    lo = _spec_num(spec, "min")
    hi = _spec_num(spec, "max")
    if lo is not None and hi is not None and hi > lo:
        mid = (lo + hi) / 2.0
        pts = [lo, mid, hi]
        if isinstance(default, int) and "number" not in spec.lower():
            pts = sorted({int(round(p)) for p in pts})
        return pts
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        # numeric param, no declared range: probe an order of magnitude either way
        return sorted({default, default * 10 or 1, default / 10.0})
    return []


def _spec_num(spec: str, key: str) -> float | None:
    m = re.search(rf"{key}\s*[:=]\s*(-?\d+(?:\.\d+)?)", spec)
    return float(m.group(1)) if m else None


def parse_colab_params(cells: list[CellInfo]) -> list[dict]:
    """Enumerate #@param fields as control dicts (the enumerable code surface).
    Returns dicts so the caller builds schemas.Control with the slice attached."""
    out: list[dict] = []
    for c in cells:
        for line in c.source.splitlines():
            m = _PARAM_LINE.match(line)
            if not m:
                continue
            symbol, default_src, spec = m.group(1), m.group(2), m.group(3)
            default = _literal(default_src)
            out.append({
                "name": symbol, "source": "param", "cell": c.index,
                "symbol": symbol, "baseline_value": default,
                "values": _param_values(default, spec),
                "intent": f"Colab #@param form field ({spec.strip() or 'value'})",
                "signpost": line.strip()[:200],
            })
    return out
