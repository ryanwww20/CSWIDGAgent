"""Code-lane evidence gathering for v2 #5 — the non-widget surface.

Given the baseline kernel (KernelActuator, already run = the cost oracle), a list
of candidate `Control`s (#@param fields + planner-found semantic controls), and
the static dataflow for each, this:

  1. Tiers every control (executed / reasoned), using the free baseline cost
     oracle first and a single live probe with a 10-min/cell ceiling otherwise.
  2. Gathers evidence for `executed` controls: re-run the DEFINING cell (with the
     perturbed literal patched in) through the CONTIGUOUS range up to its last
     dependent cell, snapshot before/after into Filmstrip frames. Re-running the
     contiguous range (not just the dependent cells) keeps intermediate setup
     consistent; we snapshot after the last cell so the frame is control-relevant.
     A cell that exceeds the ceiling demotes the WHOLE control to `reasoned` (its
     other values are NOT tried) and the kernel is restarted + replayed clean.
  3. Isolation (spec §2.8): for cheap notebooks, restore a clean slate between
     controls and re-run the baseline prefix before each value so the defining
     cell sees fresh upstream inputs; expensive notebooks go best-effort (their
     controls are mostly reasoned anyway).
  4. Attaches the `dataflow_only` companion signal (does the perturbation reach a
     displayed output?) so a reasoned/dead control is distinguishable.

Frames carry `control` + `tier` so nothing reasoned is ever shown as observed.
"""
from __future__ import annotations

from pathlib import Path

import dataflow
from actuate_kernel import KernelActuator
from schemas import Control, TraceEntry

# 10-minute per-cell ceiling (spec §2.4 / §4).
CEILING_S = 600.0
# Oracle shortcut: if any cell in the re-run range already took longer than this
# at baseline, re-running it for a sweep is too costly -> reasoned, no probe.
EXPENSIVE_S = 120.0
# Cap how many perturbation values an executed control actually runs.
MAX_VALUES = 3
# Below this total baseline cost (s) we do full prefix-replay isolation between
# controls/values; above it we go best-effort (those notebooks' controls are
# mostly reasoned anyway).
ISOLATION_BUDGET = 90.0


def _cost(act: KernelActuator, indices: list[int]) -> float:
    return sum(act.cell_times.get(i, 0.0) for i in indices)


def _clean(act: KernelActuator, indices: list[int]) -> bool:
    return all(act.cell_clean.get(i, True) for i in indices)


def _expensive(act: KernelActuator, indices: list[int]) -> bool:
    return any(act.cell_times.get(i, 0.0) > EXPENSIVE_S for i in indices)


def assign_tier(act: KernelActuator, ctrl: Control, run_set: list[int]) -> str:
    """Tier from the free oracle alone (no execution). `executed` here means
    'cheap enough to attempt' — a live ceiling hit can still demote it later."""
    if ctrl.source == "view_only":
        return "reasoned"                      # can't be execute-verified
    if not ctrl.values:
        return "reasoned"                      # no deterministic perturbation to run
    if not _clean(act, run_set):
        return "reasoned"                      # notebook can't run this in-env
    if _expensive(act, run_set):
        return "reasoned"                      # oracle says too costly to sweep
    return "executed"


def run(act: KernelActuator, cells: list[dataflow.CellInfo],
        controls: list[Control], *, start_step: int = 0,
        max_values: int = MAX_VALUES) -> tuple[list[TraceEntry], list[Control], int]:
    """Tier + gather evidence for code-lane controls. Returns (frames, controls
    with tier assigned, next free step number)."""
    entries: list[TraceEntry] = []
    step = start_step
    src_by_idx = dict(act.code_cells)

    for ctrl in controls:
        dependents = ctrl.downstream_cells or dataflow.forward_slice(
            cells, ctrl.cell, ctrl.symbol)
        ctrl.downstream_cells = dependents
        run_set = dataflow.rerun_range(cells, ctrl.cell, ctrl.symbol)
        ctrl.tier = assign_tier(act, ctrl, run_set)

        # dataflow_only companion signal — does the perturbation reach a display?
        wired = dataflow.slice_reaches_display(cells, run_set)
        ctrl.notes = (ctrl.notes + f" wired_in={wired}").strip()

        if ctrl.tier != "executed":
            continue  # reasoned: judged from code + author's stored output, no frames

        defining_src = src_by_idx.get(ctrl.cell, "")
        after_defining = [i for i in run_set if i > ctrl.cell]
        isolate = sum(act.cell_times.values()) <= ISOLATION_BUDGET
        if isolate:
            act.replay_baseline_inplace(timeout_per_cell=CEILING_S)

        def _prep():
            if isolate:
                act.replay_prefix(ctrl.cell, timeout_per_cell=CEILING_S)

        # clean baseline of this control's run range
        _prep()
        act.run_cells(run_set, timeout_per_cell=CEILING_S)
        base_hash, _ = act.snapshot(f"ctrl_{ctrl.symbol}_base")
        base_stdout = act.last_stdout

        demoted = False
        for value in ctrl.values[:max_values]:
            patched = dataflow.patch_assignment(defining_src, ctrl.symbol, value)
            _prep()
            if patched is not None:
                set_ok = True
                res = act.run_cells(run_set, timeout_per_cell=CEILING_S,
                                    patches={ctrl.cell: patched})
            else:
                # No simple assignment to patch -> run defining as-is, override the
                # value in-kernel, re-run the rest of the range.
                act.run_cells([ctrl.cell], timeout_per_cell=CEILING_S)
                set_ok = act.set_symbol(ctrl.symbol, value)
                res = act.run_cells(after_defining, timeout_per_cell=CEILING_S)
            if res["timed_out_cell"] is not None:
                # Ceiling hit: demote the whole control, do not try other values.
                ctrl.tier = "reasoned"
                ctrl.notes = (ctrl.notes +
                              f" demoted=ceiling@cell{res['timed_out_cell']}").strip()
                act.reset_kernel_to_baseline()
                demoted = True
                break

            cur_hash, images = act.snapshot(f"step{step + 1:02d}_{ctrl.symbol}")
            stdout = res["stdout"] or ""
            text_changed = bool(stdout) and stdout != base_stdout
            # "Broke" = regression vs the clean baseline: a cell errored now but was
            # clean at baseline (a pre-existing env failure is not charged).
            regressed = any(act.cell_clean.get(i, True) for i in res["errored_cells"])
            step += 1
            entries.append(TraceEntry(
                step=step, action="set", controls_set={ctrl.symbol: value},
                target_exists=True, set_ok=set_ok,
                output_changed=(cur_hash != base_hash) or text_changed,
                hash_before=base_hash, hash_after=cur_hash,
                image=images[0] if images else "",
                note=ctrl.intent or f"{ctrl.symbol}={value!r}",
                stdout=(stdout[:500] or None), stderr=res["stderr"],
                control=ctrl.name, tier="executed",
                errored_vs_baseline=regressed))

        # Non-isolated path: best-effort restore of the run range to baseline.
        if not demoted and not isolate:
            act.run_cells(run_set, timeout_per_cell=CEILING_S)

    return entries, controls, step


def baseline_frame(act: KernelActuator, out_dir: Path) -> str:
    """A single snapshot of the author's stored/rendered output, attached as
    context for reasoned controls (which have no perturbation frame of their own)."""
    _, images = act.snapshot("step00_baseline")
    return images[0] if images else ""
