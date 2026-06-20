"""LLM planner (Axis B, exploration): turn the enumerated controls + the demo's
stated intent into a *declarative* action plan — a human-like, multi-control
sequence a curious learner would try to understand the concept.

The planner only emits data (list[Action]); it never actuates. Every action it
proposes is validated against the real WidgetSpecs (unknown controls dropped,
numbers clamped to range, selections snapped to valid options), so a careless
plan degrades gracefully instead of driving phantom controls.
"""
from __future__ import annotations

import json
from typing import Any

from llm import LLMClient
from schemas import Action, Control, WidgetSpec

_SYSTEM = (
    "You are exploring an interactive teaching demo to judge whether its controls "
    "actually illustrate the concept. You do NOT grade it — you only decide which "
    "control settings a thoughtful learner should try. Propose a short, purposeful "
    "sequence (contrasts, edge cases, and combinations that should reveal cause and "
    "effect). Output ONLY JSON."
)

_USER_TMPL = """Demo intent / guided observation:
{intent}

Drivable controls (drive these by their exact "name"):
{controls}

Return JSON: {{"actions": [ ... ]}} where each action is one of:
  {{"set": {{"<name>": <value>, ...}}, "note": "why"}}      # set one or MORE controls at once
  {{"click": "<button name>", "repeat": <int>, "note": "why"}}

Rules:
- Use ONLY the control names listed. For selections use one of the listed options;
  for numerics stay within [min, max]; for bools use true/false.
- Prefer 6-10 actions that contrast settings to expose the concept (e.g. set a
  control to a revealing combination, then change one thing to isolate its effect).
- "note" must say what the learner is trying to observe. No prose outside JSON."""


def _control_card(w: WidgetSpec) -> dict[str, Any]:
    card: dict[str, Any] = {"name": w.name, "kind": w.kind, "type": w.type}
    if w.kind == "numeric":
        card["min"], card["max"], card["step"] = w.min, w.max, w.step
    if w.kind == "selection":
        card["options"] = w.options
    if w.kind == "bool":
        card["values"] = [False, True]
    card["current"] = w.value
    return card


def _coerce(spec: WidgetSpec, value: Any) -> Any | None:
    """Snap a proposed value to something the control can actually take."""
    if spec.kind == "numeric":
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if spec.min is not None:
            v = max(v, float(spec.min))
        if spec.max is not None:
            v = min(v, float(spec.max))
        return int(round(v)) if str(spec.type).startswith("Int") else v
    if spec.kind == "selection":
        opts = [str(o) for o in (spec.options or [])]
        sval = str(value)
        if sval in opts:
            return sval
        if isinstance(value, int) and 0 <= value < len(opts):
            return opts[value]
        return opts[0] if opts else None
    if spec.kind == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("true", "1", "yes", "on")
    if spec.kind == "text":
        return str(value)
    return None


def plan(client: LLMClient, widgets: list[WidgetSpec], demo_intent: str,
         *, start_step: int = 0, max_actions: int = 10) -> tuple[list[Action], dict]:
    """Return (validated actions, debug info incl. raw + usage)."""
    drivable = [w for w in widgets if w.drivable]
    by_name = {w.name: w for w in drivable}
    buttons = {w.name for w in drivable if w.kind == "button"}
    if not drivable:
        return [], {"skipped": "no drivable controls"}

    controls = json.dumps([_control_card(w) for w in drivable],
                          ensure_ascii=False, indent=2)
    user = _USER_TMPL.format(intent=(demo_intent or "(none provided)")[:4000],
                            controls=controls)
    res = client.complete(system=_SYSTEM, user=user, json_mode=True,
                          role_tag="planner")
    data = res.parsed_json or {}
    raw_actions = data.get("actions", []) if isinstance(data, dict) else \
        (data if isinstance(data, list) else [])

    actions: list[Action] = []
    step = start_step
    for ra in raw_actions[:max_actions]:
        if not isinstance(ra, dict):
            continue
        if "click" in ra and ra["click"] in buttons:
            step += 1
            actions.append(Action(step=step, kind="click", target=ra["click"],
                                   repeat=int(ra.get("repeat", 1) or 1),
                                   note=str(ra.get("note", ""))[:200]))
            continue
        if "set" in ra and isinstance(ra["set"], dict):
            controls_set: dict[str, Any] = {}
            for name, val in ra["set"].items():
                spec = by_name.get(name)
                if spec is None or spec.kind == "button":
                    continue
                coerced = _coerce(spec, val)
                if coerced is not None:
                    controls_set[name] = coerced
            if controls_set:
                step += 1
                actions.append(Action(step=step, kind="set", controls=controls_set,
                                       note=str(ra.get("note", ""))[:200]))

    info = {"model": res.model, "provider": res.provider,
            "usage": {"in": res.usage.input_tokens, "out": res.usage.output_tokens},
            "raw_count": len(raw_actions), "kept": len(actions)}
    return actions, info


# --- v2 (#5): the SEMANTIC tweakable surface (constants / inputs / view-only) ---

_SEMANTIC_SYSTEM = (
    "You identify the INTENDED tweakable surface of a teaching notebook beyond its "
    "widgets and Colab #@param fields, for an interactivity evaluation. A learner "
    "is meant to edit certain values and re-run to see an effect.\n\n"
    "BE CONSERVATIVE AND SIGNPOST-ONLY. Include a constant or input ONLY when there "
    "is EXPLICIT intent that the learner tweak it: an inline comment (e.g. "
    "'# try changing this', '# experiment with'), a markdown instruction (e.g. "
    "'change X and re-run', 'try different prompts'), or an obvious swappable input "
    "the demo tells the learner to vary (the prompt text, the input image/dataset). "
    "Do NOT treat every numeric literal or string as tweakable — an un-signposted "
    "constant is NOT a control. If nothing is signposted, return an empty list.\n"
    "For each control you MUST quote the exact signpost text. Output ONLY JSON."
)

_SEMANTIC_USER = """Demo intent / guided observation (markdown):
{markdown}

Code cells (drive a control by patching `symbol` in its `cell` index):
{listing}

Already-covered controls (widgets + #@param) — do NOT propose these symbols again:
{covered}

Return JSON: {{"controls": [ ... ]}} where each control is:
  {{"cell": <code-cell index from the listing>,
    "symbol": "<the variable assigned in that cell that the learner edits>",
    "source": "constant" | "input" | "view_only",
    "values": [<2-3 concrete literal values to try; [] for view_only>],
    "intent": "<what the learner is meant to observe by changing it>",
    "signpost": "<the EXACT comment or markdown sentence that licenses this>"}}

Rules:
- `cell` must be a code-cell index shown above; `symbol` must be assigned in it.
- `values` are concrete literals (numbers, strings, or listed choices) — never code.
- "view_only" = a figure meant to be hovered/zoomed or an animation (no value to
  set); use it sparingly and only when the markdown points the learner at it.
- Omit anything without an explicit signpost. No prose outside JSON."""


def cells_listing(cells, markdown: str, *, max_chars: int = 9000) -> tuple[str, str]:
    """Render (code-cell listing, markdown blob) for the semantic planner, each
    truncated so a huge notebook can't blow the context. Cells keep their index."""
    parts: list[str] = []
    used = 0
    for c in cells:
        block = f"[code cell {c.index}]\n{c.source}\n"
        if used + len(block) > max_chars:
            parts.append(f"[... {len(cells) - len(parts)} more cells truncated ...]")
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts), (markdown or "(none)")[:4000]


def plan_semantic_controls(client: LLMClient, cells, markdown: str,
                           covered_symbols: set[str], *, max_controls: int = 8
                           ) -> tuple[list[Control], dict]:
    """Return (validated semantic Controls, debug info). Each control is checked:
    the symbol must actually be assigned in the named cell, must not already be
    covered, and must carry a non-empty signpost (the with-teeth conservatism)."""
    import dataflow  # local import to avoid a cycle at module load

    listing, md = cells_listing(cells, markdown)
    covered = ", ".join(sorted(covered_symbols)) or "(none)"
    user = _SEMANTIC_USER.format(markdown=md, listing=listing, covered=covered)
    res = client.complete(system=_SEMANTIC_SYSTEM, user=user, json_mode=True,
                          role_tag="planner_semantic")
    data = res.parsed_json or {}
    raw = data.get("controls", []) if isinstance(data, dict) else (
        data if isinstance(data, list) else [])

    assigned_in = {c.index: c.assigns for c in cells}
    controls: list[Control] = []
    dropped: list[str] = []
    for rc in raw[:max_controls]:
        if not isinstance(rc, dict):
            continue
        symbol = str(rc.get("symbol", "")).strip()
        source = str(rc.get("source", "constant")).strip()
        signpost = str(rc.get("signpost", "")).strip()
        cell = rc.get("cell")
        if not symbol or symbol in covered_symbols:
            dropped.append(f"{symbol or '?'}:covered_or_empty")
            continue
        if not signpost and source != "view_only":
            dropped.append(f"{symbol}:no_signpost")   # with-teeth: signpost required
            continue
        # Resolve / validate the defining cell from real dataflow.
        if not isinstance(cell, int) or symbol not in assigned_in.get(cell, set()):
            cell = dataflow.find_symbol_cell(cells, symbol)
        if cell is None and source != "view_only":
            dropped.append(f"{symbol}:not_assigned")
            continue
        values = rc.get("values") if isinstance(rc.get("values"), list) else []
        controls.append(Control(
            name=symbol, source=source if source in (
                "constant", "input", "view_only") else "constant",
            cell=cell if isinstance(cell, int) else -1, symbol=symbol,
            values=[v for v in values if not isinstance(v, (list, dict))],
            intent=str(rc.get("intent", ""))[:300], signpost=signpost[:300]))

    info = {"model": res.model, "provider": res.provider,
            "usage": {"in": res.usage.input_tokens, "out": res.usage.output_tokens},
            "raw_count": len(raw), "kept": len(controls), "dropped": dropped}
    return controls, info
