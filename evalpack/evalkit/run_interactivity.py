"""Interactivity-eval orchestrator.

Pipeline: router -> actuator.enumerate -> deterministic sweep (+ optional LLM
planner) -> executor -> Filmstrip -> deterministic effectiveness (+ optional
blind LLM judge).

The LLM stages are optional: with no API key (or --no-llm) the harness still
produces a Filmstrip and deterministic effectiveness, so it runs anywhere.

Usage:
  python scripts/eval_harness/run_interactivity.py <notebook.ipynb> [--out DIR]
      [--no-llm] [--planner-model M] [--judge-model M]
      [--source slides|slides+transcript]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load API keys from a .env (repo root or cwd) so keys can live in one file.
try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except Exception:  # noqa: BLE001
    pass

import router  # noqa: E402
from actuator import get_actuator  # noqa: E402
from schemas import InteractivityScore, harmonic_mean, to_json  # noqa: E402
import executor  # noqa: E402
from sweep import build_sweep  # noqa: E402


def _demo_intent(nb_path: Path) -> str:
    """Concatenate the notebook's markdown cells — the demo's own guided
    observation / 'what to take away' is the planner & judge's intent context."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="artifact dir (default: <notebook_dir>/interactivity_eval)")
    ap.add_argument("--max-points", type=int, default=3)
    ap.add_argument("--cell-timeout", type=float, default=120.0)
    ap.add_argument("--no-llm", action="store_true",
                    help="skip planner + judge (deterministic only)")
    ap.add_argument("--planner-model", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--source", default="slides",
                    help="what the demo was built from (faithfulness context)")
    args = ap.parse_args()

    out = args.out or args.notebook.parent / "interactivity_eval"
    out.mkdir(parents=True, exist_ok=True)

    det = router.detect(args.notebook)
    print(f"router: type={det.primary.value} lane={det.lane} signals={det.signals}")
    if det.lane not in ("kernel",):
        print(f"NOTE: lane '{det.lane}' not wired yet (kernel only). Stopping.")
        return 2

    # Resolve LLM models (graceful: None => stage skipped).
    planner_model = judge_model = None
    if not args.no_llm:
        from llm import pick_models
        dp, dj = pick_models()
        planner_model = args.planner_model or dp
        judge_model = args.judge_model or dj
    intent = _demo_intent(args.notebook)

    act = get_actuator(det.lane, notebook_path=str(args.notebook),
                       workdir=str(out / "frames"), cell_timeout=args.cell_timeout)
    act.start()
    plan_info = judge_info = None
    try:
        if getattr(act, "cell_errors", None):
            print(f"  notebook cell errors (continued): {len(act.cell_errors)}")
            for e in act.cell_errors[:5]:
                print("   ", e)
        widgets = act.enumerate()
        print(f"  enumerated {len(widgets)} widget(s); "
              f"{sum(1 for w in widgets if w.drivable)} drivable")
        for w in widgets:
            if w.drivable:
                print(f"    - {w.name} [{w.type}/{w.kind}] "
                      f"options={w.options} range={w.min}..{w.max}")

        plan = build_sweep(widgets, max_points=args.max_points)
        print(f"  deterministic sweep: {len(plan)} step(s)")

        if planner_model:
            try:
                from llm import LLMClient
                from planner import plan as make_plan
                client = LLMClient(planner_model)
                planned, plan_info = make_plan(client, widgets, intent,
                                               start_step=len(plan))
                plan += planned
                print(f"  planner ({planner_model}): +{len(planned)} step(s) "
                      f"{plan_info}")
            except Exception as e:  # noqa: BLE001
                print(f"  planner skipped: {e}")
        else:
            print("  planner: skipped (no model/key)")

        film = executor.run(act, plan, notebook=str(args.notebook),
                            interactivity_type=det.primary.value, lane=det.lane,
                            widgets=widgets)
    finally:
        act.stop()

    eff = film.effectiveness()
    rob = film.robustness()
    (out / "filmstrip.json").write_text(to_json(film), encoding="utf-8")
    print(f"\ncoverage: {film.coverage}")
    print(f"effectiveness (deterministic sweep): {eff}")
    print(f"#5 robustness = {rob['score']}  "
          f"({rob['broken_steps']}/{rob['attempted_steps']} steps broke; "
          f"set_failures={rob['set_failures']} errors/NaN={rob['error_or_nan_steps']})")
    print(f"wrote {out/'filmstrip.json'}  (+ frames in {out/'frames'})")

    # #5 effectiveness is the blind judge's intent-aware usefulness (no det. ceiling).
    effectiveness_score = None
    if judge_model:
        try:
            from llm import LLMClient
            from judge import judge as run_judge
            client = LLMClient(judge_model)
            verdict, judge_info = run_judge(client, film, intent, args.source)
            (out / "judge.json").write_text(to_json(verdict), encoding="utf-8")
            effectiveness_score = verdict.usefulness
            print(f"\njudge ({judge_model}): usefulness={verdict.usefulness} "
                  f"flags={verdict.flags}")
            print(f"  rationale: {verdict.rationale[:400]}")
            print(f"  {judge_info}")
            print(f"wrote {out/'judge.json'}")
        except Exception as e:  # noqa: BLE001
            print(f"\njudge skipped: {e}")
    else:
        print("\njudge: skipped (no model/key)")

    # #5 = harmonic mean of effectiveness (judge) + robustness (deterministic).
    # NA when EITHER sub-score is missing: harmonic_mean() drops None values, so
    # without this guard a --no-llm run would silently report robustness alone
    # as the headline #5 (e.g. a perfect-looking 5.0 with no judge involved).
    final = (harmonic_mean([effectiveness_score, rob["score"]])
             if effectiveness_score is not None and rob["score"] is not None
             else None)
    score = InteractivityScore(
        effectiveness=effectiveness_score, robustness=rob["score"],
        score=final,
        robustness_detail=rob, notebook=str(args.notebook))
    (out / "interactivity_score.json").write_text(to_json(score), encoding="utf-8")
    if score.score is not None:
        print(f"\n#5 Interactivity = {score.score}  "
              f"(effectiveness={score.effectiveness} robustness={score.robustness})")
    else:
        print(f"\n#5 Interactivity = NA  (robustness={score.robustness} "
              f"effectiveness={score.effectiveness} — both sub-scores are "
              f"required; effectiveness needs the judge, rerun without --no-llm)")
    print(f"wrote {out/'interactivity_score.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
