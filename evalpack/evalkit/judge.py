"""Blind judge (Axis: judgment): read the Filmstrip evidence the executor
produced — control list, per-step trace with deterministic output-changed
signals, and a curated set of before/after frames — and rate whether varying
the controls actually teaches the concept.

The judge NEVER actuates and is blind to method identity (no notebook path, no
system name). It must cite specific steps/frames so its rating is auditable
against the same evidence a human could replay (docs/EVAL_TESTSET_DESIGN.md §6.1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from llm import LLMClient, score_of
from schemas import Filmstrip, JudgeOutput, RESET_NOTE

_SYSTEM = (
    "You grade interactivity *effectiveness* of a teaching demo from recorded "
    "evidence only. You did not run it and cannot run it. Judge whether changing "
    "the controls produces meaningful, concept-relevant changes in the output, "
    "using the frames and the per-step change log. Be skeptical: controls that "
    "never change the output are cosmetic.\n\n"
    "CITATIONS — read carefully: each attached image has a black banner at the "
    "TOP-LEFT reading exactly 'FRAME n | STEP m'. When you cite evidence you MUST "
    "use that pairing, e.g. 'frame 3 (step 6)'. Never invent a frame or step "
    "number that is not in the manifest below, and never cite a step number as a "
    "frame number. If you compare two frames, name both, e.g. 'frame 7 (step 20) "
    "vs frame 6 (step 19)'. Output ONLY JSON."
)

_USER_TMPL = """Demo intent / guided observation:
{intent}

Source the demo was built from: {source}

Controls present:
{controls}

Per-step action log (deterministic; output_changed is a pixel-hash comparison of
the rendered figure before vs. after the step; "intent" is why the step was run):
{trace}

Deterministic effectiveness summary: {eff}

Frame manifest — the ONLY valid (frame, step) pairs you may cite. Each maps to
one attached image whose top-left banner shows the same 'FRAME n | STEP m':
{frames}

Return JSON:
{{
  "usefulness": <1-5 float>,            // does varying controls teach the concept?
  "rationale": "<cite using 'frame n (step m)' from the manifest, e.g. 'frame 6 (step 19) vs frame 7 (step 20) shows...'>",
  "per_control": {{"<name>": {{"has_effect": <bool>, "comment": "<short, cite a frame (step)>"}}}},
  "flags": ["<e.g. dead_control:<name>, only_cosmetic, strong_concept_link>"]
}}
1 = controls do nothing useful / cosmetic; 5 = controls cleanly isolate and
reveal the concept. Ground every claim in a cited frame (step); do not assume."""


def _select_frames(film: Filmstrip, max_frames: int = 12) -> list[tuple[int, str, str]]:
    """Pick auditable extremes: first & last frame per driven control, plus any
    multi-control (planner) frames. Returns [(step, label, path)]."""
    picked: dict[int, tuple[int, str, str]] = {}

    def touched(entry, name) -> bool:
        return name in (entry.controls_set or {}) or entry.target == name

    for w in film.widgets:
        if not w.drivable:
            continue
        grp = [e for e in film.entries
               if touched(e, w.name) and e.image and e.note != RESET_NOTE]
        for e in (grp[:1] + grp[-1:]) if grp else []:
            if Path(e.image).exists():
                label = f"step {e.step}: " + (
                    ", ".join(f"{k}={v}" for k, v in e.controls_set.items())
                    if e.controls_set else f"click {e.target}")
                picked[e.step] = (e.step, label, e.image)

    # Add multi-control (planner) frames if room.
    for e in film.entries:
        if len(picked) >= max_frames:
            break
        if len(e.controls_set) > 1 and e.image and Path(e.image).exists():
            label = "step %d: " % e.step + ", ".join(
                f"{k}={v}" for k, v in e.controls_set.items())
            picked[e.step] = (e.step, label, e.image)

    out = sorted(picked.values(), key=lambda t: t[0])[:max_frames]
    return out


def _trace_lines(film: Filmstrip) -> str:
    rows = []
    for e in film.entries:
        what = (", ".join(f"{k}={v}" for k, v in e.controls_set.items())
                if e.controls_set else f"click {e.target}")
        note = f" intent={e.note!r}" if e.note else ""
        extra = ""
        if e.stdout:
            extra += f" printed={e.stdout[:120]!r}"
        if e.stderr:
            extra += f" stderr={e.stderr[:120]!r}"
        rows.append(f"  step {e.step}: {e.action} [{what}] "
                    f"set_ok={e.set_ok} output_changed={e.output_changed}{note}{extra}")
    return "\n".join(rows)


def _label_frame(src: str, dst: Path, frame_no: int, step: int) -> str:
    """Burn an ASCII 'FRAME n | STEP m' banner onto the frame so the image is
    self-identifying — the model can read the number it's citing, instead of
    inferring it from positional order. Falls back to the raw frame on error."""
    from imaging import label_image
    return label_image(src, dst, f"FRAME {frame_no} | STEP {step}")


def judge(client: LLMClient, film: Filmstrip, demo_intent: str,
          source_provided: str = "", *, max_frames: int = 12
          ) -> tuple[JudgeOutput, dict]:
    frames = _select_frames(film, max_frames)
    controls = "\n".join(
        f"  - {w.name} [{w.kind}/{w.type}] drivable={w.drivable}"
        + (f" options={w.options}" if w.options else "")
        + (f" range={w.min}..{w.max}" if w.min is not None else "")
        for w in film.widgets if w.drivable)

    # Burn a 'FRAME n | STEP m' banner so citations are verifiable in-pixel.
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
        controls=controls or "  (none)",
        trace=_trace_lines(film) or "  (no steps)",
        eff=film.effectiveness(),
        frames=frame_manifest)

    res = client.complete(system=_SYSTEM, user=user, images=labeled,
                          json_mode=True, role_tag="judge",
                          required={"usefulness": "score", "rationale": "str"})
    data = res.parsed_json
    out = JudgeOutput(
        usefulness=score_of(data, "usefulness"),
        rationale=str(data.get("rationale", "")),
        per_control=data.get("per_control", {}) if isinstance(
            data.get("per_control"), dict) else {},
        flags=[str(f) for f in data.get("flags", [])] if isinstance(
            data.get("flags"), list) else [])
    info = {"model": res.model, "provider": res.provider,
            "usage": {"in": res.usage.input_tokens, "out": res.usage.output_tokens},
            "frames_shown": len(frames),
            "parsed_ok": bool(data)}
    return out, info
