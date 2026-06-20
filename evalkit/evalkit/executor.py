"""Executor: apply a plan (list[Action]) through ANY Actuator, snapshotting the
output before/after each step, and assemble the Filmstrip evidence bundle.

Lane-agnostic by design — it only talks to the Actuator interface, so the same
code drives Layer 0 (kernel) and Layer 1 (browser), and runs both deterministic
sweeps and LLM-planned actions.
"""
from __future__ import annotations

from actuator import Actuator
from schemas import Action, Filmstrip, TraceEntry, WidgetSpec


def run(actuator: Actuator, plan: list[Action], *, notebook: str,
        interactivity_type: str, lane: str,
        widgets: list[WidgetSpec] | None = None) -> Filmstrip:
    """Execute `plan` and return a Filmstrip. Assumes actuator already started
    (so the caller can enumerate + plan first)."""
    if widgets is None:
        widgets = actuator.enumerate()
    names = {w.name for w in widgets}

    entries: list[TraceEntry] = []
    prev_hash, _ = actuator.snapshot("step00_baseline")
    prev_stdout = ""

    driven: set[str] = set()
    failed = 0
    for action in plan:
        controls_set: dict = {}
        target = action.target
        target_exists = True
        set_ok = True

        if action.kind in ("set", "sweep_point"):
            controls_set = dict(action.controls)
            target_exists = all(n in names for n in controls_set)
            res = actuator.set_values(controls_set)
            set_ok = bool(res) and all(res.values())
            driven.update(controls_set)
        elif action.kind == "click":
            target_exists = target in names
            set_ok = actuator.click(target, repeat=action.repeat) if target_exists else False
            if target:
                driven.add(target)
        if not set_ok:
            failed += 1

        # What the action's callbacks printed / raised (kernel lane). stdout is
        # the demo's *output* for text-rendering demos; stderr is judge evidence.
        stdout = (getattr(actuator, "last_stdout", "") or "").strip()
        stderr = getattr(actuator, "last_stderr", None)

        cur_hash, images = actuator.snapshot(f"step{action.step:02d}")
        # A demo that renders by printing (no figure) still "changed output"
        # when its callbacks printed something different than the last step.
        text_changed = bool(stdout) and stdout != prev_stdout
        # Widget controls are driven live in the kernel -> tier "executed".
        control = (next(iter(controls_set)) if controls_set else target)
        entries.append(TraceEntry(
            step=action.step, action=action.kind, controls_set=controls_set,
            target=target, target_exists=target_exists, set_ok=set_ok,
            output_changed=(cur_hash != prev_hash) or text_changed,
            hash_before=prev_hash, hash_after=cur_hash,
            image=images[0] if images else "", note=action.note,
            stdout=stdout[:500] or None, stderr=stderr,
            control=control, tier="executed"))
        prev_hash = cur_hash
        if stdout:
            prev_stdout = stdout

    coverage = {
        "found": len(widgets),
        "drivable": sum(1 for w in widgets if w.drivable),
        "driven": len(driven),
        "failed_steps": failed,
    }
    return Filmstrip(notebook=notebook, interactivity_type=interactivity_type,
                     lane=lane, widgets=widgets, entries=entries,
                     coverage=coverage)
