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
from schemas import Action, WidgetSpec

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
