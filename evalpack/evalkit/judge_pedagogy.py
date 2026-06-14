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
    "MORE DEEPLY from this notebook than from the slide alone? Score the MARGINAL "
    "value the notebook adds over the static slide — interaction-driven intuition, "
    "worked examples, visualization, what-if exploration, things only a runnable "
    "interactive notebook can do. You MUST name specifically what the notebook adds "
    "that the slide alone does not. If it merely restates the slide in code, score "
    "low. Do NOT reward surface polish that adds no understanding.\n\n"
    "STAY IN YOUR LANE: judge ONLY the marginal instructional value of what the "
    "notebook shows and lets the learner do. Do NOT penalize the notebook for using "
    "a synthetic / toy / simplified or even hardcoded model — whether the mechanism "
    "is faithful to reality is scored by a SEPARATE metric, not here. Take the "
    "displayed outputs as given and ask: does interacting with and reading this "
    "notebook build more understanding of the slide's concept than the slide alone? "
    "Output ONLY JSON. 5 = substantially deepens understanding via things only an "
    "interactive notebook can do; 1 = adds nothing beyond the slide."
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
