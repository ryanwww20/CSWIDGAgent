"""Interactivity-eval orchestrator (v2 #5 = effectiveness only; robustness deleted).

Pipeline (INTERACTIVITY_V2_SPEC §3):
  1. router          -> is the notebook runnable / does it have a tweakable surface?
  2. baseline run    -> KernelActuator.start(): run once, record per-cell wall-times
                        (free cost oracle) + which cells ran clean (regression ref).
  3. enumerable surface (deterministic):
        - ipywidgets  (live registry)   -> sweep + optional widget planner
        - #@param     (static parse)    -> code-lane controls
  4. semantic surface (planner LLM, conservative + signposted): editable constants,
        swappable inputs, view-only figures — told which controls are already covered.
  5. tier each control (executed / reasoned) via the oracle + a 10-min probe, and
        gather evidence (code lane: symbol override + downstream-slice re-run).
  6. tier-aware blind judge -> per-control usefulness.
  7. score: per-control usefulness, capped at 4.5 if reasoned, aggregated by MEAN.
        #5 = effectiveness.  #5 = NA only if there is no tweakable surface at all.

The LLM stages are optional: with --no-llm there is no planner/judge, so the
deterministic filmstrip is still produced but #5 effectiveness is NA (the judge
is what scores usefulness).

Usage:
  python evalkit/run_interactivity.py <notebook.ipynb> [--out DIR] [--no-llm]
      [--planner-model M] [--judge-model M] [--source slides|slides+transcript]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except Exception:  # noqa: BLE001
    pass

import router  # noqa: E402
import dataflow  # noqa: E402
import executor  # noqa: E402
import code_controls  # noqa: E402
from actuator import get_actuator  # noqa: E402
from sweep import build_sweep  # noqa: E402
from schemas import (Control, ControlEval, Filmstrip, InteractivityScore,  # noqa: E402
                     cap_for_tier, mean_score, to_json)


def _markdown(nb_path: Path) -> str:
    """Concatenate markdown cells — the demo's guided observation / intent."""
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return ""
    md = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", "")
            md.append(src if isinstance(src, str) else "".join(src))
    return "\n\n".join(md).strip()


def _to_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _na(out: Path, notebook: str, reason: str) -> int:
    """Write an NA #5 score and return. NA is a real property here (no tweakable
    surface / no judge), never a faked zero."""
    score = InteractivityScore(effectiveness=None, score=None, notebook=notebook)
    (out / "interactivity_score.json").write_text(to_json(score), encoding="utf-8")
    print(f"\n#5 Interactivity = NA  ({reason})")
    print(f"wrote {out/'interactivity_score.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-points", type=int, default=3)
    ap.add_argument("--cell-timeout", type=float, default=600.0,
                    help="per-cell timeout for the baseline + slice re-runs (10-min ceiling)")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip planner + judge (deterministic filmstrip only; #5 = NA)")
    ap.add_argument("--planner-model", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--source", default="slides")
    args = ap.parse_args()

    out = args.out or args.notebook.parent / "interactivity_eval"
    out.mkdir(parents=True, exist_ok=True)

    det = router.detect(args.notebook)
    print(f"router: type={det.primary.value} lane={det.lane} "
          f"runnable={det.runnable} signals={det.signals}")
    if not det.runnable:
        return _na(out, str(args.notebook),
                   "no runnable code — a genuinely linear demo, no tweakable surface")

    planner_model = judge_model = None
    if not args.no_llm:
        from llm import pick_models
        dp, dj = pick_models()
        planner_model = args.planner_model or dp
        judge_model = args.judge_model or dj
    intent = _markdown(args.notebook)
    cells = dataflow.parse_cells(args.notebook)

    # --- 2. baseline run (cost oracle) on the kernel lane --------------------
    act = get_actuator("kernel", notebook_path=str(args.notebook),
                       workdir=str(out / "frames"), cell_timeout=args.cell_timeout)
    act.start()
    widgets = []
    widget_entries = []
    code_frames = []
    code_controls_list: list[Control] = []
    plan_info = sem_info = judge_info = None
    baseline_frame = ""
    try:
        if getattr(act, "cell_errors", None):
            print(f"  baseline cell errors (continued): {len(act.cell_errors)}")
            for e in act.cell_errors[:5]:
                print("   ", e)
        print(f"  baseline: {len(act.code_cells)} code cells, "
              f"{sum(act.cell_clean.values())} clean, "
              f"total {round(sum(act.cell_times.values()), 1)}s")

        # author's stored-output frame (context for reasoned controls)
        baseline_frame = code_controls.baseline_frame(act, out / "frames")

        # --- 3a. enumerable widgets (live registry) -> sweep + widget planner
        widgets = act.enumerate()
        drivable = [w for w in widgets if w.drivable]
        print(f"  widgets: {len(widgets)} enumerated, {len(drivable)} drivable")
        if drivable:
            plan = build_sweep(widgets, max_points=args.max_points)
            if planner_model:
                try:
                    from llm import LLMClient
                    from planner import plan as make_plan
                    planned, plan_info = make_plan(
                        LLMClient(planner_model), widgets, intent, start_step=len(plan))
                    plan += planned
                    print(f"  widget planner: +{len(planned)} step(s) {plan_info}")
                except Exception as e:  # noqa: BLE001
                    print(f"  widget planner skipped: {e}")
            wfilm = executor.run(act, plan, notebook=str(args.notebook),
                                 interactivity_type=det.primary.value,
                                 lane=det.lane, widgets=widgets)
            widget_entries = wfilm.entries

        # --- 3b. enumerable #@param + 4. semantic surface (planner LLM) ------
        covered = {w.name for w in drivable}
        param_dicts = dataflow.parse_colab_params(cells)
        for pd in param_dicts:
            covered.add(pd["symbol"])
            pd["downstream_cells"] = dataflow.forward_slice(
                cells, pd["cell"], pd["symbol"])
            code_controls_list.append(Control(**pd))
        print(f"  #@param fields: {len(param_dicts)}")

        if planner_model:
            try:
                from llm import LLMClient
                from planner import plan_semantic_controls
                sem, sem_info = plan_semantic_controls(
                    LLMClient(planner_model), cells, intent, covered)
                for c in sem:
                    c.downstream_cells = dataflow.forward_slice(cells, c.cell, c.symbol) \
                        if c.cell >= 0 else []
                code_controls_list += sem
                print(f"  semantic planner: +{len(sem)} control(s) {sem_info}")
            except Exception as e:  # noqa: BLE001
                print(f"  semantic planner skipped: {e}")

        # Disambiguate controls that share a symbol across cells (e.g. `epochs` in
        # two training sections) so per-control judging/scoring don't collide.
        from collections import Counter
        counts = Counter(c.name for c in code_controls_list)
        for c in code_controls_list:
            if counts[c.name] > 1:
                c.name = f"{c.symbol}@c{c.cell}"

        # --- 5. tier + gather evidence for code-lane controls ---------------
        start_step = max([e.step for e in widget_entries], default=0)
        code_frames, code_controls_list, _ = code_controls.run(
            act, cells, code_controls_list, start_step=start_step)
    finally:
        act.stop()

    # --- assemble the filmstrip --------------------------------------------
    entries = widget_entries + code_frames
    film = Filmstrip(
        notebook=str(args.notebook), interactivity_type=det.primary.value,
        lane=det.lane, widgets=widgets, entries=entries,
        controls=code_controls_list,
        coverage={"widgets": len(widgets), "code_controls": len(code_controls_list)},
        meta={"baseline_frame": baseline_frame, "router_signals": det.signals})
    (out / "filmstrip.json").write_text(to_json(film), encoding="utf-8")
    print(f"\nwrote {out/'filmstrip.json'}  (+ frames in {out/'frames'})")

    # the universe of controls #5 scores: drivable widgets (executed) + code controls
    scored_controls = [(w.name, "widget", "executed") for w in widgets if w.drivable]
    scored_controls += [(c.name, c.source, c.tier) for c in code_controls_list]
    if not scored_controls:
        return _na(out, str(args.notebook),
                   "no tweakable surface found (no widgets, #@param, or signposted controls)")
    n_reasoned = sum(1 for _, _, t in scored_controls if t == "reasoned")
    print(f"  controls scored: {len(scored_controls)} "
          f"({n_reasoned} reasoned, capped at 4.5)")

    # --- 6. tier-aware judge ------------------------------------------------
    if not judge_model:
        print("\njudge: skipped (no model/key) — #5 effectiveness needs the judge")
        return _na(out, str(args.notebook), "no judge run (--no-llm)")
    try:
        from llm import LLMClient
        from judge import judge as run_judge
        # temperature=0 for the blind judge → lower run-to-run variance in #5.
        verdict, judge_info = run_judge(
            LLMClient(judge_model, temperature=0.0), film, intent, args.source)
        (out / "judge.json").write_text(to_json(verdict), encoding="utf-8")
        print(f"\njudge ({judge_model}): overall={verdict.usefulness} "
              f"flags={verdict.flags}  {judge_info}")
    except Exception as e:  # noqa: BLE001
        print(f"\njudge failed: {e}")
        return _na(out, str(args.notebook), f"judge error: {e}")

    # --- 7. scoring: per-control cap 4.5 if reasoned, mean aggregate --------
    wired = {c.name: dataflow.slice_reaches_display(cells, [c.cell, *c.downstream_cells])
             for c in code_controls_list}
    per_control: list[ControlEval] = []
    for name, source, tier in scored_controls:
        pc = verdict.per_control.get(name, {}) if isinstance(verdict.per_control, dict) else {}
        raw = _to_float(pc.get("usefulness"))
        if raw is None:
            raw = verdict.usefulness  # fallback to the overall rating
        capped = cap_for_tier(raw, tier)
        has_effect = pc.get("has_effect")
        if tier == "executed":  # deterministic ground truth overrides for widgets/code
            det_effect = any(e.output_changed for e in entries
                             if (e.control == name) or (name in e.controls_set))
            has_effect = det_effect if has_effect is None else has_effect
        errored = any(e.errored_vs_baseline for e in entries if e.control == name)
        per_control.append(ControlEval(
            name=name, source=source, tier=tier, usefulness=raw, capped=capped,
            has_effect=has_effect, wired_in=wired.get(name),
            errored_vs_baseline=errored, comment=str(pc.get("comment", ""))[:300]))

    effectiveness = mean_score([ce.capped for ce in per_control])
    score = InteractivityScore(
        effectiveness=effectiveness, score=effectiveness, per_control=per_control,
        aggregation="mean", n_controls=len(per_control), n_reasoned=n_reasoned,
        notebook=str(args.notebook))
    (out / "interactivity_score.json").write_text(to_json(score), encoding="utf-8")

    print("\nper-control effectiveness:")
    for ce in per_control:
        cap_note = f" (capped {ce.usefulness}->{ce.capped})" if (
            ce.tier == "reasoned" and ce.usefulness != ce.capped) else ""
        print(f"    {ce.name} [{ce.source}/{ce.tier}] = {ce.capped}{cap_note}")
    print(f"\n#5 Interactivity = {score.score}  "
          f"(effectiveness, mean of {len(per_control)} controls; {n_reasoned} reasoned)")
    print(f"wrote {out/'interactivity_score.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
