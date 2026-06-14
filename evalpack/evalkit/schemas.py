"""Contracts between the interactivity-eval roles (planner / executor / judge).

These dataclasses are the *only* things the roles exchange. They are designed to
be JSON-serializable, frozen as artifacts, and hashable so a scored run can be
replayed deterministically (see docs/EVAL_TESTSET_DESIGN.md §6.1, §6.3).

Axis A (actuation) = how a control is poked: kernel / dom / vision.
Axis B (exploration) = what sequence of pokes: sweep / planned.
The judge never actuates; it only reads a Filmstrip the executor produced.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

Driver = Literal["kernel", "dom", "vision"]
ActionKind = Literal["set", "click", "sweep_point"]

# Note used by sweep-generated steps that restore a control to its default value
# between widgets (state isolation). Judges/frame-selection treat these as
# bookkeeping, not exploration.
RESET_NOTE = "reset to default"


@dataclass
class WidgetSpec:
    """One interactive control, as enumerated by an actuator (ground truth)."""
    id: str                       # stable handle (model_id / dom ref / mark id)
    name: str                     # unique display name (description or type#i)
    type: str                     # FloatSlider, Dropdown, Button, range, select…
    driver: Driver                # which layer can drive it
    kind: str = "other"           # button|selection|numeric|bool|text|other
    drivable: bool = True         # False => flagged not_drivable, never faked
    value: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[list[Any]] = None   # labels, for dropdowns / selects
    notes: str = ""


# Widget type-name -> driving kind. Selection widgets are driven by option
# index/label; numeric by value; bool by True/False; button by click.
_SELECTION = {"Dropdown", "Select", "SelectMultiple", "SelectionSlider",
              "SelectionRangeSlider", "RadioButtons", "ToggleButtons"}
_NUMERIC = {"FloatSlider", "IntSlider", "FloatLogSlider", "FloatText", "IntText",
            "BoundedFloatText", "BoundedIntText", "FloatRangeSlider", "IntRangeSlider"}
_BOOL = {"Checkbox", "ToggleButton", "Valid"}
_TEXT = {"Text", "Textarea", "Combobox", "Password"}


def widget_kind(type_name: str, has_options: bool = False,
                value: Any = None) -> str:
    if type_name == "Button":
        return "button"
    if type_name in _SELECTION or has_options:
        return "selection"
    if type_name in _NUMERIC:
        return "numeric"
    if type_name in _BOOL or isinstance(value, bool):
        return "bool"
    if type_name in _TEXT:
        return "text"
    return "other"


@dataclass
class Action:
    """One step of an exploration plan (declarative; emitted as data)."""
    step: int
    kind: ActionKind
    controls: dict[str, Any] = field(default_factory=dict)  # name -> value (set)
    target: Optional[str] = None                            # widget name (click)
    repeat: int = 1
    note: str = ""


@dataclass
class TraceEntry:
    """What the executor recorded after applying one Action — the filmstrip frame.

    `output_changed` + hashes give the deterministic *effectiveness* signal for
    free; `target_exists`/`set_ok` make a plan referencing a missing control
    visible rather than silently dropped.
    """
    step: int
    action: ActionKind
    controls_set: dict[str, Any] = field(default_factory=dict)
    target: Optional[str] = None
    target_exists: bool = True
    set_ok: bool = True
    output_changed: bool = False
    hash_before: str = ""
    hash_after: str = ""
    image: str = ""               # relative path to PNG frame
    note: str = ""                # planner/sweep intent for this step
    stdout: Optional[str] = None  # text the action's callbacks printed (text demos)
    stderr: Optional[str] = None


@dataclass
class Filmstrip:
    """The full evidence bundle for one demo — the only thing the judge reads."""
    notebook: str
    interactivity_type: str       # from router.InteractivityType
    lane: str                     # "kernel" | "browser" | "vision" | "param" | "none"
    widgets: list[WidgetSpec] = field(default_factory=list)
    entries: list[TraceEntry] = field(default_factory=list)
    coverage: dict[str, int] = field(default_factory=dict)  # found/driven/failed
    trace_zip: Optional[str] = None      # Playwright trace.zip (browser lane)
    meta: dict[str, Any] = field(default_factory=dict)

    def effectiveness(self) -> dict[str, Any]:
        """Deterministic effectiveness (no LLM): did pokes change the output?"""
        changed = sum(1 for e in self.entries if e.output_changed)
        total = len(self.entries)
        dead = [w.name for w in self.widgets if w.drivable and not any(
            e.output_changed and w.name in e.controls_set for e in self.entries)]
        return {"changed_steps": changed, "total_steps": total,
                "fraction_changed": round(changed / total, 3) if total else 0.0,
                "dead_or_cosmetic_controls": dead}

    def robustness(self) -> dict[str, Any]:
        """Deterministic interaction-robustness (no LLM): do pokes BREAK the demo?

        Metric #5's robustness sub-score (docs/EVAL_TESTSET_DESIGN.md §6 / #5):
        among steps that drove an EXISTING control, what fraction failed to set
        (`set_ok` False) or surfaced an error / NaN / inf in stderr. Dead-but-
        stable controls are NOT penalized here — that's effectiveness's job;
        robustness asks only "does driving controls crash or NaN the demo?".

        Maps the broken fraction onto a 1-5 score. `score` is None when no control
        was driven (nothing to be robust about → NA, not a free 5).
        """
        attempted = [e for e in self.entries
                     if e.target_exists and e.action in ("set", "sweep_point", "click")]
        n = len(attempted)
        set_failures = [e.step for e in attempted if not e.set_ok]
        error_steps = [e.step for e in attempted if _stderr_is_error(e.stderr)]
        broken = sorted(set(set_failures) | set(error_steps))
        missing = [e.step for e in self.entries if not e.target_exists]
        frac = len(broken) / n if n else 0.0
        return {
            "attempted_steps": n,
            "broken_steps": len(broken),
            "fraction_broken": round(frac, 3),
            "set_failures": set_failures,
            "error_or_nan_steps": error_steps,
            "plan_referenced_missing": missing,
            "score": _robustness_score(frac) if n else None,
        }


@dataclass
class JudgeInput:
    """Blind judge input. Method identity is deliberately absent."""
    filmstrip: Filmstrip
    demo_intent: str = ""         # the demo's own guided-observation markdown
    source_provided: str = ""     # "slides" | "slides+transcript" (faithfulness)


@dataclass
class JudgeOutput:
    usefulness: float             # does varying controls teach the concept? 1-5
    rationale: str                # must cite specific steps/frames
    per_control: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


@dataclass
class InteractivityScore:
    """Metric #5 — Interactivity (harmonic mean of effectiveness + robustness).

    effectiveness = the blind judge's intent-aware `usefulness` (1-5); robustness
    = deterministic from the trace (1-5). `score` is the harmonic mean so a demo
    that is effective-but-fragile (or robust-but-cosmetic) can't average its way
    to a good number. NA (None) when either sub-score is missing (e.g. --no-llm
    leaves effectiveness unscored).
    """
    effectiveness: Optional[float] = None   # judge usefulness 1-5 (intent-aware)
    robustness: Optional[float] = None       # deterministic from trace 1-5
    score: Optional[float] = None            # harmonic_mean(effectiveness, robustness)
    robustness_detail: dict[str, Any] = field(default_factory=dict)
    notebook: str = ""


# ---- quality metrics (#2 faithfulness, #3 pedagogy, #4 topic-worthiness) ----

@dataclass
class FaithfulnessOutput:
    """Metric #2 — Faithfulness & Correctness (harmonic mean of three 1-5 sub-scores)."""
    assertional: float = 0.0      # 2a: no contradictions/fabrications vs source
    computational: float = 0.0    # 2b: code derives outcomes (not hardcoded)
    correctness: float = 0.0      # 2c: correct in truth (post-verification)
    score: float = 0.0            # harmonic_mean(2a_final, 2b, 2c_final)
    rationale: str = ""
    errors_cited: list[dict] = field(default_factory=list)     # 2c proposals
    errors_verified: list[dict] = field(default_factory=list)  # verifier verdicts
    flags: list[str] = field(default_factory=list)


@dataclass
class PedagogyOutput:
    """Metric #3 — Pedagogical Depth (marginal value over the slide)."""
    depth: float = 0.0            # 1-5
    added_value: list[str] = field(default_factory=list)  # what NB adds vs slide
    rationale: str = ""
    flags: list[str] = field(default_factory=list)


@dataclass
class TopicOutput:
    """Metric #4 — Topic-Worthiness."""
    worthiness: float = 0.0       # 1-5
    interactive_is_right_tool: bool = True
    rationale: str = ""
    flags: list[str] = field(default_factory=list)


@dataclass
class ClarityOutput:
    """Metric #7 — Exposition / Clarity (harmonic mean of three 1-5 sub-scores).

    Lane-guarded: scores ONLY how clearly the notebook communicates and lays out
    its material — NOT correctness (#2), depth (#3), or whether controls work (#5).
    """
    visual: float = 0.0           # 7a: figure layout, legibility, no overlap/clipping
    textual: float = 0.0          # 7b: prose structure, flow, equation rendering
    code_explanation: float = 0.0  # 7c: readable code, sane names, narrated steps
    score: float = 0.0            # harmonic_mean(7a, 7b, 7c)
    rationale: str = ""
    citations: list[dict] = field(default_factory=list)  # concrete per-cell/frame defects
    flags: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """The headline-quality vector for one demo (no composite; §6)."""
    notebook: str
    faithfulness: Optional[FaithfulnessOutput] = None
    pedagogy: Optional[PedagogyOutput] = None
    topic: Optional[TopicOutput] = None
    clarity: Optional[ClarityOutput] = None
    meta: dict[str, Any] = field(default_factory=dict)


def harmonic_mean(values: list[float]) -> Optional[float]:
    """Harmonic mean over strictly-positive scores (the §6 sub-score composition).

    Punishes a single low value far harder than the arithmetic mean — the point
    of harmonic composition for #2/#5/#7. Non-positive (missing/unparsed) values
    are dropped rather than zeroing the whole metric; returns None if nothing is
    valid (so callers can render NA instead of a misleading 0).
    """
    valid = [v for v in values if v is not None and v > 0]
    if not valid:
        return None
    return round(len(valid) / sum(1.0 / v for v in valid), 2)


# stderr that signals a real break: a traceback/exception, or a NaN/inf/overflow
# (an unhandled numerical degeneracy when a control is pushed to an extreme).
_ERR_MARKERS = ("traceback", "exception", "error:", "valueerror", "runtimeerror",
                "zerodivisionerror", "overflowerror")
_NAN_RE = re.compile(r"\b(nan|inf|-inf|invalid value|overflow encountered|"
                     r"divide by zero)\b")


def _stderr_is_error(stderr: Optional[str]) -> bool:
    if not stderr:
        return False
    s = stderr.lower()
    return any(m in s for m in _ERR_MARKERS) or bool(_NAN_RE.search(s))


def _robustness_score(fraction_broken: float) -> float:
    """Map the broken-step fraction onto a 1-5 robustness score (deterministic)."""
    if fraction_broken <= 0.0:
        return 5.0
    if fraction_broken <= 0.05:
        return 4.0
    if fraction_broken <= 0.15:
        return 3.0
    if fraction_broken <= 0.40:
        return 2.0
    return 1.0


def to_json(obj: Any) -> str:
    return json.dumps(asdict(obj), ensure_ascii=False, indent=2, sort_keys=True)


def stable_hash(obj: Any) -> str:
    """Content hash of a dataclass (for freeze/replay identity checks)."""
    payload = json.dumps(asdict(obj), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
