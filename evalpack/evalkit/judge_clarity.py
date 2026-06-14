"""Metric #7 — Exposition / Clarity (docs/EVAL_TESTSET_DESIGN.md §6.0).

Question: "Can a student actually read and follow this notebook?" Three 1-5
sub-scores from a code-aware, vision-capable judge, composed by HARMONIC mean so
one badly-clear dimension can't be hidden behind two good ones:

  7a visual  — figure layout, legibility, no overlap/clipping/redundant repr boxes.
  7b textual — prose structure, logical flow, equation rendering, no wall-of-text.
  7c code-explanation — readable code, sane names, each step narrated.

LANE-GUARD: this is the softest, most subjective axis, so the judge is held to
the same discipline as the others — it must cite CONCRETE per-cell / per-frame
defects, and it must score ONLY communication/layout quality, NOT correctness
(#2), depth (#3), topic choice (#4), or whether controls work (#5). It is also
reference-free (blind to the slides) on purpose: clarity is about whether the
artifact reads well on its own, not whether it matches a source.
"""
from __future__ import annotations

from pathlib import Path

from llm import LLMClient, score_of
from schemas import ClarityOutput, harmonic_mean
from textbudget import TRUNCATION_RULE, truncate

_SYSTEM = (
    "You judge EXPOSITION / CLARITY: can a student actually READ and FOLLOW this "
    "teaching notebook? Score three INDEPENDENT 1-5 axes of communication quality. "
    "Be a STRICT, demanding reviewer: 5 is reserved for genuinely clean work and "
    "must be EARNED, not given by default. When you are between two bands, choose "
    "the LOWER one. Output ONLY JSON.\n\n"
    "7a VISUAL CLARITY (from the rendered output images). Inspect every figure "
    "closely for these DEFECTS and treat them as serious, not cosmetic:\n"
    "  - overlapping or colliding text: any two labels/titles/annotations that "
    "touch or sit on top of each other;\n"
    "  - lines/arrows/markers crossing THROUGH text: e.g. a dashed limit/threshold "
    "line, gridline, or arrow that strikes through or overlaps a box label or "
    "caption (this is a real overlap defect, not decoration);\n"
    "  - clipped or cut-off content: titles, legends, axis labels, or boxes "
    "running off the edge of the figure or truncated;\n"
    "  - illegible text: fonts too small to read at the rendered size, or labels "
    "truncated to a cryptic fragment;\n"
    "  - poor default scaling: the default/first view a student sees compresses the "
    "key data into a sliver or hides the differences the figure is meant to show;\n"
    "  - chartjunk and redundant artifacts: a bare function/object repr box "
    "(e.g. '<function ...>' or '<no docstring>') dumped as output, duplicate "
    "legends, etc.\n"
    "SCORE BY THE WORST PROMINENT FIGURE, NOT THE AVERAGE. A student is harmed by "
    "the broken figure even if ten other figures are clean; do NOT let clean "
    "figures offset a defective one, and weight the FIRST / default / most-"
    "referenced ('hero') figures most. Calibration: 5 = every figure is clean and "
    "immediately legible, nothing overlaps, nothing clipped, sensible default "
    "scale; 4 = essentially clean, at most ONE trivial nit in a MINOR figure; "
    "3 = at least one prominent figure has a real readability defect from the list "
    "above (e.g. a limit line striking through labels, colliding text, a clipped "
    "legend, or a default view that hides the key data); 2 = multiple figures have "
    "such defects OR a hero/first figure is substantially unreadable; 1 = key "
    "figures fail to communicate. Any single prominent overlap / clipping / "
    "illegibility defect CAPS this axis at 3.\n\n"
    "7b TEXTUAL CLARITY (from the markdown prose): is the writing organized and "
    "easy to follow? Reward clear headings, logical flow, well-rendered equations, "
    "and scaffolding; penalize walls of text, disorganized ordering, broken/raw or "
    "mis-rendered LaTeX, inconsistent/undeclared notation, and missing explanation "
    "between steps. Calibration: 5 = consistently well-structured throughout, "
    "equations render correctly, no wall-of-text; 4 = mostly clean with at most one "
    "minor lapse; 3 = at least one real structural problem (a wall of text, a "
    "broken/raw equation, or a confusing/undeclared notation or ordering); 2 = "
    "multiple such problems; 1 = confusing or disorganized throughout. A single "
    "broken/unreadable equation or a genuine wall-of-text caps this axis at 3.\n\n"
    "7c CODE-EXPLANATION CLARITY (from the code + adjacent markdown): is the code "
    "readable and is each meaningful step narrated? Reward sane names, reasonable "
    "cell sizes, and prose that explains what a cell does and why; penalize an "
    "unexplained dump of code, cryptic single-letter names everywhere, and "
    "uncommented dense blocks. Calibration: 5 = readable code with each step "
    "explained; 4 = mostly readable, minor gaps; 3 = at least one long/dense cell "
    "with no narration; 2 = several such cells; 1 = opaque code dump with no "
    "narration.\n\n"
    "STAY STRICTLY IN YOUR LANE. Do NOT judge whether claims are correct or "
    "faithful (#2), how much the notebook adds over a slide (#3), whether the "
    "concept was worth demoing (#4), or whether the interactive controls actually "
    "work (#5). A demo can be perfectly clear yet wrong, shallow, or non-"
    "interactive — those are scored elsewhere; score ONLY how clearly it "
    "communicates and lays out its material. You are blind to the source slides on "
    "purpose: judge the notebook as it reads on its own. BURDEN OF PROOF: every "
    "point you dock must be backed by a CONCRETE, checkable citation (which cell or "
    "frame, and the specific defect) — but the converse does NOT hold: do NOT award "
    "5 just because nothing came to mind. A 5 means you actively inspected the "
    "figures and prose and confirmed they are clean. Look hard for the defects "
    "above before settling on a score."
) + TRUNCATION_RULE

_USER_TMPL = """NOTEBOOK (full content: markdown + code; rendered outputs attached as images):
{notebook}

Return JSON:
{{
  "visual": <1-5>,
  "textual": <1-5>,
  "code_explanation": <1-5>,
  "rationale": "<overall read; cite specific cells/frames>",
  "citations": [
    {{"where": "<cell N / frame name>",
      "axis": "<visual|textual|code_explanation>",
      "defect": "<the concrete, checkable clarity defect>"}}
  ],
  "flags": ["<e.g. overlapping_labels, line_through_labels, clipped_legend, illegible_small_text, poor_default_scale, redundant_repr_box, wall_of_text, broken_equation, unexplained_code_dump>"]
}}"""


def judge_clarity(judge: LLMClient, notebook_text: str,
                  notebook_images: list[str]) -> tuple[ClarityOutput, dict]:
    user = _USER_TMPL.format(notebook=truncate(notebook_text, 60000, "the notebook"))
    res = judge.complete(system=_SYSTEM, user=user,
                         images=[Path(p) for p in notebook_images],
                         json_mode=True, role_tag="clarity",
                         required={"visual": "score", "textual": "score",
                                   "code_explanation": "score",
                                   "rationale": "str"})
    d = res.parsed_json
    visual = score_of(d, "visual")
    textual = score_of(d, "textual")
    code_expl = score_of(d, "code_explanation")
    score = harmonic_mean([visual, textual, code_expl])

    citations = d.get("citations", [])
    out = ClarityOutput(
        visual=visual, textual=textual, code_explanation=code_expl, score=score,
        rationale=str(d.get("rationale", "")),
        citations=[c for c in citations if isinstance(c, dict)] if isinstance(
            citations, list) else [],
        flags=[str(f) for f in d.get("flags", [])] if isinstance(
            d.get("flags"), list) else [])
    info = {"model": res.model, "provider": res.provider,
            "usage": {"in": res.usage.input_tokens, "out": res.usage.output_tokens},
            "n_citations": len(out.citations)}
    return out, info
