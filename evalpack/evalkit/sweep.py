"""Axis-B deterministic exploration: one-control-at-a-time min/mid/max (+ a few
discrete points). No LLM. This is the cheap *effectiveness floor* — it catches
dead/cosmetic controls and is the gameability guard for interactivity scoring
(docs/EVAL_TESTSET_DESIGN.md §6.1).
"""
from __future__ import annotations

from schemas import Action, RESET_NOTE, WidgetSpec


def _numeric_points(w: WidgetSpec, max_points: int = 3) -> list[float]:
    if w.min is not None and w.max is not None:
        lo, hi = w.min, w.max
        if max_points <= 2:
            pts = [lo, hi]
        else:
            pts = [lo, (lo + hi) / 2.0, hi]
        if str(w.type).startswith("Int"):
            pts = sorted({int(round(p)) for p in pts})
        return pts
    # Unbounded: probe around the current value.
    v = w.value if isinstance(w.value, (int, float)) else 1
    return sorted({v, (v * 10) or 1, (v // 10) if isinstance(v, int) else v / 10.0})


def build_sweep(widgets: list[WidgetSpec], max_points: int = 3) -> list[Action]:
    """Return a flat, one-control-at-a-time plan.

    After sweeping each control, it is restored to its enumerated default so the
    next control is probed from a clean state — otherwise control B would be
    swept with control A still pinned at its max, and B's frames/effect would be
    conflated with A's (state-carryover bias).
    """
    actions: list[Action] = []
    step = 0

    def _reset(w: WidgetSpec) -> None:
        nonlocal step
        if w.value is None or w.kind == "button":
            return
        # Selection defaults are restored by label; a raw value that doesn't
        # resolve to a listed option can't be restored — skip rather than emit
        # a step that is guaranteed to fail (false robustness hit).
        if w.kind == "selection" and str(w.value) not in (w.options or []):
            return
        step += 1
        actions.append(Action(step=step, kind="set",
                               controls={w.name: w.value}, note=RESET_NOTE))

    for w in widgets:
        if not w.drivable:
            continue
        if w.kind == "button":
            step += 1
            actions.append(Action(step=step, kind="click", target=w.name,
                                   repeat=3, note="click x3"))
        elif w.kind == "selection":
            opts = w.options or []
            chosen = opts if len(opts) <= max_points else \
                [opts[0], opts[len(opts) // 2], opts[-1]]
            for o in chosen:
                step += 1
                actions.append(Action(step=step, kind="sweep_point",
                                       controls={w.name: o}, note=f"select {o}"))
            _reset(w)
        elif w.kind == "numeric":
            for v in _numeric_points(w, max_points):
                step += 1
                actions.append(Action(step=step, kind="sweep_point",
                                       controls={w.name: v}, note=f"{w.name}={v}"))
            _reset(w)
        elif w.kind == "bool":
            for v in (False, True):
                step += 1
                actions.append(Action(step=step, kind="sweep_point",
                                       controls={w.name: v}, note=f"{w.name}={v}"))
            _reset(w)
        # text / other: skipped by the deterministic sweep (planner can probe).
    return actions
