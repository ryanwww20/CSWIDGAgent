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
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

Driver = Literal["kernel", "dom", "vision"]
ActionKind = Literal["set", "click", "sweep_point"]

# v2 (#5): evidence tier a control's usefulness was established under.
#   executed       — perturbation applied and re-run; before/after is observed.
#   dataflow_only  — not executed; static check that the symbol flows into a
#                    displayed output (wired-in vs dead). Companion signal.
#   reasoned       — not executed (too expensive / view-only / env can't run it);
#                    usefulness inferred from code + author's stored output. Capped.
Tier = Literal["executed", "dataflow_only", "reasoned"]
REASONED_CAP = 4.5            # a reasoned control's usefulness can never exceed this

# How a control's tweakable surface was discovered (provenance of the candidate).
#   widget   — live ipywidgets registry (enumerable)
#   param    — Colab #@param static parse (enumerable)
#   constant — signposted editable constant (semantic, planner-found)
#   input    — swappable input: prompt / image / dataset (semantic)
#   view_only— hover/zoom/animation; cannot be executed-verified (semantic)
ControlSource = Literal["widget", "param", "constant", "input", "view_only"]

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
class Control:
    """A v2 tweakable control beyond the live-widget registry — a #@param field,
    a signposted editable constant, a swappable input, or a view-only figure.

    Emitted as data (by the static #@param parser or the LLM planner), validated,
    then tiered + evidence-gathered by code_controls.py. The patch target is a
    *code cell index* + *symbol*; values are the perturbations to try.
    """
    name: str                              # display name (symbol or short label)
    source: str                            # ControlSource: param|constant|input|view_only
    cell: int = -1                         # code-cell index that assigns `symbol`
    symbol: str = ""                       # variable patched in that cell
    baseline_value: Any = None             # value at baseline (repr if not JSON-able)
    values: list[Any] = field(default_factory=list)   # perturbations to apply
    downstream_cells: list[int] = field(default_factory=list)  # static forward slice
    intent: str = ""                       # what the learner is meant to observe
    signpost: str = ""                     # the explicit "tweak me" evidence (required)
    tier: str = "executed"                 # Tier; assigned by the tiering step
    notes: str = ""


@dataclass
class ControlEval:
    """Per-control #5 result: the tier it was judged under, the judge's raw
    usefulness, and the reasoned-capped score that feeds the #5 aggregate."""
    name: str
    source: str = ""                       # ControlSource
    tier: str = "executed"                 # Tier the evidence was gathered under
    usefulness: Optional[float] = None     # judge's raw per-control usefulness 1-5
    capped: Optional[float] = None         # min(usefulness, 4.5) if reasoned else usefulness
    has_effect: Optional[bool] = None      # did perturbation change a displayed output?
    wired_in: Optional[bool] = None        # dataflow_only: symbol reaches a displayed cell?
    errored_vs_baseline: bool = False      # executed: regressed vs the clean baseline?
    comment: str = ""                      # judge rationale (must cite a frame/cell)


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
    # v2 (#5): which control this frame perturbs, and the evidence tier it was
    # gathered under. `errored_vs_baseline` is True only when this perturbation
    # raised an error that the clean baseline run did NOT (a real regression, not
    # a pre-existing env failure) — see EXECUTED gating in code_controls.py.
    control: Optional[str] = None
    tier: str = "executed"        # executed | dataflow_only | reasoned
    errored_vs_baseline: bool = False


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
    # v2 (#5): the non-widget tweakable surface (#@param + signposted semantic
    # controls), each carrying its assigned evidence `tier`. Widget controls stay
    # in `widgets`; both feed the per-control judge + scorer.
    controls: list[Control] = field(default_factory=list)
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


@dataclass
class JudgeInput:
    """Blind judge input. Method identity is deliberately absent."""
    filmstrip: Filmstrip
    demo_intent: str = ""         # the demo's own guided-observation markdown
    source_provided: str = ""     # "slides" | "slides+transcript" (faithfulness)


@dataclass
class JudgeOutput:
    """Tier-aware blind judge verdict. `per_control` is primary in v2: one entry
    per control with its own usefulness; `usefulness` is the overall/fallback."""
    usefulness: float             # overall, or fallback when per_control is empty
    rationale: str                # must cite specific steps/frames
    per_control: dict[str, Any] = field(default_factory=dict)  # name -> {usefulness, ...}
    flags: list[str] = field(default_factory=list)


def cap_for_tier(usefulness: Optional[float], tier: str) -> Optional[float]:
    """Apply the reasoned-tier ceiling (§2.6): a reasoned control caps at 4.5;
    executed / dataflow_only controls keep their raw usefulness. Per-control."""
    if usefulness is None:
        return None
    return min(usefulness, REASONED_CAP) if tier == "reasoned" else usefulness


@dataclass
class InteractivityScore:
    """Metric #5 — Interactivity (effectiveness only; v2).

    Robustness was deleted (it never discriminated). #5 = the aggregate of
    per-control usefulness, each capped at 4.5 if its evidence tier is `reasoned`
    (§2.6). Aggregation is the MEAN of per-control capped scores. If every control
    is reasoned, #5 ≤ 4.5 follows naturally. `score` == `effectiveness` (kept as a
    field so run_eval's summary wiring is unchanged). NA (None) only when there is
    genuinely no tweakable surface, or the judge could not be run (e.g. --no-llm).
    """
    effectiveness: Optional[float] = None   # mean of per-control capped usefulness
    score: Optional[float] = None            # == effectiveness (no robustness term)
    per_control: list[ControlEval] = field(default_factory=list)
    aggregation: str = "mean"                # how per_control -> effectiveness
    n_controls: int = 0
    n_reasoned: int = 0                      # how many landed in the reasoned tier
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


def mean_score(values: list[float]) -> Optional[float]:
    """Plain mean over present (non-None) scores — the §8-Q1 aggregation of
    per-control capped usefulness into #5. None if nothing valid (render NA)."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 2)


def to_json(obj: Any) -> str:
    return json.dumps(asdict(obj), ensure_ascii=False, indent=2, sort_keys=True)


def stable_hash(obj: Any) -> str:
    """Content hash of a dataclass (for freeze/replay identity checks)."""
    payload = json.dumps(asdict(obj), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
