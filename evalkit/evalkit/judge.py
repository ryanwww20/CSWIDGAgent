"""Blind, TIER-AWARE judge (v2 #5): read the Filmstrip evidence the executor
produced — the control list (widgets + code-lane controls, each with its evidence
tier), the per-step trace, and a curated set of before/after frames — and rate,
PER CONTROL, whether varying it actually teaches the concept.

The judge NEVER actuates and is blind to method identity. It must cite specific
steps/frames so its rating is auditable. Controls whose tier is `reasoned` were
NOT executed (too expensive / view-only / env can't run them); the judge is told
so it does not present speculation as observation. The reasoned 4.5 cap is applied
later, in scoring — not here (docs/EVAL_TESTSET_DESIGN.md §6.1, INTERACTIVITY_V2_SPEC §2.6).
"""
from __future__ import annotations

from pathlib import Path

from llm import LLMClient, score_of
from schemas import Filmstrip, JudgeOutput, RESET_NOTE

_SYSTEM = (
    "You grade interactivity *effectiveness* of a teaching demo from recorded "
    "evidence only. You did not run it and cannot run it. For EACH control, judge "
    "whether changing it produces meaningful, concept-relevant changes in the "
    "output. Be skeptical: a control whose output never changes is cosmetic, and a "
    "tweak that changes the picture but teaches nothing about the concept is not "
    "effective either (a control earns a high score only when its variation "
    "illustrates the concept).\n\n"
    "CALIBRATION (be demanding, PER CONTROL; when between two bands choose the "
    "LOWER): 5 = changing it cleanly isolates and reveals the concept, with a clear "
    "visible effect tied to the lesson (rare); 4 = a real, concept-relevant effect "
    "but incremental, noisy, or not fully isolated; 3 = it changes the output but "
    "the pedagogical link is weak or the change is visible-but-cosmetic; 2 = barely "
    "changes anything meaningful; 1 = dead / cosmetic / no useful effect.\n\n"
    "EVIDENCE TIERS — read carefully. Each control is tagged with a tier:\n"
    " - executed: it was actually perturbed and re-run; the frames show real "
    "before/after output. Rate from what you SEE.\n"
    " - reasoned: it could NOT be executed (too expensive, view-only, or the env "
    "can't run it). There is NO observed before/after — only the code and the "
    "author's stored output. Reason about the LIKELY effect, and be more "
    "skeptical; do not claim you observed a change you did not.\n\n"
    "CITATIONS: each attached image has a black banner at the TOP-LEFT reading "
    "exactly 'FRAME n | STEP m'. Cite evidence with that pairing, e.g. "
    "'frame 3 (step 6)'. Never invent a frame/step not in the manifest. "
    "Output ONLY JSON."
)

_USER_TMPL = """Demo intent / guided observation:
{intent}

Source the demo was built from: {source}

Controls present — rate EACH one by its exact name:
{controls}

Per-step action log (deterministic; output_changed is a pixel-hash comparison of
the rendered output before vs. after the step; "intent" is why the step was run;
"tier" is how the evidence was gathered):
{trace}

Deterministic effectiveness summary: {eff}

Frame manifest — the ONLY valid (frame, step) pairs you may cite. Each maps to one
attached image whose top-left banner shows the same 'FRAME n | STEP m':
{frames}

Return JSON:
{{
  "per_control": {{
    "<control name>": {{
      "usefulness": <1-5 float>,         // does varying THIS control teach the concept?
      "has_effect": <bool>,              // did/should its output change meaningfully?
      "tier": "executed" | "reasoned",   // echo the tier you judged under
      "comment": "<short; cite 'frame n (step m)' for executed, or the cell/code for reasoned>"
    }}, ...
  }},
  "usefulness": <1-5 float>,             // overall, only as a fallback
  "rationale": "<2-3 sentences, cite frames/cells>",
  "flags": ["<e.g. dead_control:<name>, only_cosmetic, strong_concept_link, speculative:<name>>"]
}}
1 = the control does nothing useful / cosmetic; 5 = changing it cleanly isolates
and reveals the concept. Rate every listed control. Ground executed claims in a
cited frame (step); for reasoned controls, ground them in the code + stored output."""


def _control_lines(film: Filmstrip) -> str:
    rows: list[str] = []
    for w in film.widgets:
        if w.drivable:
            rows.append(f"  - {w.name} [widget:{w.type}] tier=executed"
                        + (f" options={w.options}" if w.options else "")
                        + (f" range={w.min}..{w.max}" if w.min is not None else ""))
    for c in film.controls:
        sp = f' signpost="{c.signpost[:80]}"' if c.signpost else ""
        rows.append(f"  - {c.name} [{c.source}] tier={c.tier} cell={c.cell}{sp}"
                    f" intent={c.intent[:80]!r}")
    return "\n".join(rows) or "  (none)"


def _control_key(entry) -> str | None:
    if entry.control:
        return entry.control
    if entry.controls_set:
        return next(iter(entry.controls_set))
    return entry.target


def _select_frames(film: Filmstrip, max_frames: int = 14) -> list[tuple[int, str, str]]:
    """Pick auditable extremes: first & last frame per control (widget OR code),
    plus the baseline frame and any multi-control frames. Returns [(step,label,path)]."""
    picked: dict[int, tuple[int, str, str]] = {}

    by_control: dict[str, list] = {}
    for e in film.entries:
        if not e.image or e.note == RESET_NOTE or not Path(e.image).exists():
            continue
        key = _control_key(e)
        if key is None:
            continue
        by_control.setdefault(key, []).append(e)

    for grp in by_control.values():
        for e in grp[:1] + grp[-1:]:
            label = f"step {e.step}: " + (
                ", ".join(f"{k}={v}" for k, v in e.controls_set.items())
                if e.controls_set else f"click {e.target}")
            picked[e.step] = (e.step, label, e.image)

    # The author's baseline render (step 0) — context, esp. for reasoned controls.
    base = film.meta.get("baseline_frame")
    if base and Path(base).exists():
        picked[0] = (0, "baseline (author's stored output)", base)

    out = sorted(picked.values(), key=lambda t: t[0])[:max_frames]
    return out


def _trace_lines(film: Filmstrip) -> str:
    rows = []
    for e in film.entries:
        what = (", ".join(f"{k}={v}" for k, v in e.controls_set.items())
                if e.controls_set else f"click {e.target}")
        note = f" intent={e.note!r}" if e.note else ""
        ctl = f" control={e.control}" if e.control else ""
        extra = ""
        if e.stdout:
            extra += f" printed={e.stdout[:120]!r}"
        if e.stderr:
            extra += f" stderr={e.stderr[:120]!r}"
        if e.errored_vs_baseline:
            extra += " REGRESSED_VS_BASELINE"
        rows.append(f"  step {e.step}: {e.action} [{what}]{ctl} tier={e.tier} "
                    f"set_ok={e.set_ok} output_changed={e.output_changed}{note}{extra}")
    return "\n".join(rows) or "  (no steps)"


def _label_frame(src: str, dst: Path, frame_no: int, step: int) -> str:
    from imaging import label_image
    return label_image(src, dst, f"FRAME {frame_no} | STEP {step}")


def judge(client: LLMClient, film: Filmstrip, demo_intent: str,
          source_provided: str = "", *, max_frames: int = 14
          ) -> tuple[JudgeOutput, dict]:
    frames = _select_frames(film, max_frames)

    label_dir = Path(frames[0][2]).parent / "_judge_labeled" if frames else None
    if label_dir:
        label_dir.mkdir(exist_ok=True)
    labeled: list[Path] = []
    manifest_rows: list[str] = []
    for i, (step, label, path) in enumerate(frames, 1):
        dst = label_dir / f"frame{i:02d}_step{step:02d}.png" if label_dir else None
        labeled.append(Path(_label_frame(path, dst, i, step) if dst else path))
        manifest_rows.append(f"  frame {i} = step {step}: {label.split(': ', 1)[-1]}")
    frame_manifest = "\n".join(manifest_rows) or "  (no frames captured)"

    user = _USER_TMPL.format(
        intent=(demo_intent or "(none provided)")[:4000],
        source=source_provided or "unknown",
        controls=_control_lines(film),
        trace=_trace_lines(film),
        eff=film.effectiveness(),
        frames=frame_manifest)

    res = client.complete(system=_SYSTEM, user=user, images=labeled,
                          json_mode=True, role_tag="judge",
                          required={"usefulness": "score", "rationale": "str"})
    data = res.parsed_json or {}
    per_control = data.get("per_control", {})
    out = JudgeOutput(
        usefulness=score_of(data, "usefulness"),
        rationale=str(data.get("rationale", "")),
        per_control=per_control if isinstance(per_control, dict) else {},
        flags=[str(f) for f in data.get("flags", [])] if isinstance(
            data.get("flags"), list) else [])
    info = {"model": res.model, "provider": res.provider,
            "usage": {"in": res.usage.input_tokens, "out": res.usage.output_tokens},
            "frames_shown": len(frames), "parsed_ok": bool(data)}
    return out, info
