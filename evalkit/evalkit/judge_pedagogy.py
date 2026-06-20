"""Metric #3 — Pedagogical Depth (docs/EVAL_TESTSET_DESIGN.md §6.0).

Reframed question: "Would a learner understand the slide's concept *more deeply*
from this notebook than from the slide alone?" Scores the MARGINAL value the demo
adds over the static slide. Slide-aware on purpose; the judge must name what the
notebook adds that the slide alone does not (mitigates slide-leakage/halo).
"""
from __future__ import annotations

from pathlib import Path

from llm import LLMClient, score_of
from schemas import PedagogyOutput
from textbudget import TRUNCATION_RULE, truncate

_SYSTEM = (
    "You judge PEDAGOGICAL DEPTH: would a learner understand the slide's concept "
    "MORE DEEPLY from this notebook than from the slide alone? Score the marginal "
    "INSTRUCTIONAL value — the depth of understanding gained over the slide — "
    "REGARDLESS of how it is achieved. That added depth can come from a concrete "
    "worked example, an illuminating visualization of the mechanism, a runnable "
    "derivation, a revealing what-if or edge case, a head-to-head comparison, OR "
    "interaction — any means that builds understanding counts equally. Be a "
    "discerning reviewer and USE THE FULL 1-5 RANGE. Output ONLY JSON.\n\n"
    "DO NOT DOUBLE-COUNT OTHER METRICS — this is about understanding gained, nothing "
    "else:\n"
    "  - Interactivity is NOT required and is NOT the point here: a notebook that "
    "deepens understanding purely through a clear static visualization or a strong "
    "worked example can score 5. Whether the interactive controls mechanically "
    "work, isolate variables, or are 'tweakable' is scored by metric #5, NOT here — "
    "do not reward or penalize the interaction mechanism itself, only the "
    "understanding the notebook conveys.\n"
    "  - Correctness/faithfulness is metric #2: do NOT penalize a toy / synthetic / "
    "simplified / hardcoded model here.\n"
    "  - Writing/figure clarity is metric #7: do NOT score how clean the prose or "
    "layout is here.\n\n"
    "REALIZED, NOT INTENDED: score what the learner ACTUALLY sees in the attached "
    "rendered outputs, not what the code intends. If the key figures/outputs are "
    "missing, empty, broken, or errored, the realized teaching value is low (1-2) no "
    "matter how ambitious the design. You MUST name the specific things the notebook "
    "adds beyond the slide and confirm they are visible in the outputs. Do NOT "
    "reward surface polish, length, or a long feature list that doesn't deepen "
    "understanding.\n\n"
    "CALIBRATION (use the full range):\n"
    "  5 = makes the concept substantially clearer or deeper than the slide — e.g. a "
    "concrete worked example that grounds an abstraction, a vivid visualization of "
    "the mechanism, or a case that reveals a non-obvious relationship — clearly "
    "realized in the outputs.\n"
    "  4 = solid added understanding the slide lacks, visible in the outputs, but "
    "incremental.\n"
    "  3 = some added value, but much of it restates the slide, OR it is thin / a "
    "single static rendering / only partly realized.\n"
    "  2 = little marginal value: largely re-presents the slide, or the intended "
    "value is mostly UNREALIZED (broken / empty / errored outputs).\n"
    "  1 = adds nothing beyond the slide.\n\n"
    "STAY IN YOUR LANE: judge ONLY the depth of understanding gained over the slide."
) + TRUNCATION_RULE

_USER_TMPL = """SLIDE(S) the concept comes from:
{slides}

NOTEBOOK (full content; rendered outputs attached as images):
{notebook}

Return JSON:
{{
  "depth": <1-5>,
  "added_value": ["<specific thing the notebook adds beyond the slide>", ...],
  "rationale": "<cite cells/outputs; explain the marginal gain>",
  "flags": ["<e.g. restates_slide, strong_interactive_intuition>"]
}}"""


def judge_pedagogy(judge: LLMClient, notebook_text: str,
                   notebook_images: list[str], slides: str
                   ) -> tuple[PedagogyOutput, dict]:
    user = _USER_TMPL.format(slides=slides or "(none provided)",
                            notebook=truncate(notebook_text, 60000, "the notebook"))
    res = judge.complete(system=_SYSTEM, user=user,
                         images=[Path(p) for p in notebook_images],
                         json_mode=True, role_tag="pedagogy",
                         required={"depth": "score", "rationale": "str"})
    d = res.parsed_json
    out = PedagogyOutput(
        depth=score_of(d, "depth"),
        added_value=[str(x) for x in d.get("added_value", [])] if isinstance(
            d.get("added_value"), list) else [],
        rationale=str(d.get("rationale", "")),
        flags=[str(f) for f in d.get("flags", [])] if isinstance(
            d.get("flags"), list) else [])
    info = {"model": res.model, "provider": res.provider,
            "usage": {"in": res.usage.input_tokens, "out": res.usage.output_tokens}}
    return out, info
