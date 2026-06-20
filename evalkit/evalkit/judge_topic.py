"""Metric #4 — Topic-Worthiness (docs/EVAL_TESTSET_DESIGN.md §6.0).

Given the source, was the concept the notebook chose actually demo-worthy, and is
an interactive code notebook the RIGHT tool for it (vs a static figure or prose)?
The open-ended differentiator + soft guard against "pick a trivially easy concept
and nail it" gaming.
"""
from __future__ import annotations

from llm import LLMClient, score_of
from schemas import TopicOutput
from textbudget import TRUNCATION_RULE, truncate

_SYSTEM = (
    "You judge TOPIC-WORTHINESS: given the source materials and the concept this "
    "notebook chose to demo, how good was that CHOICE? Be a discerning reviewer and "
    "USE THE FULL 1-5 RANGE — do not default every demo to 4 or 5. A genuinely "
    "central, rich, interaction-suited concept earns a 5; a standard or mostly-"
    "observational choice is a 4; a narrow, peripheral, or poorly-suited one is a 3 "
    "or below. Award 5 when it is earned, and 2-3 without hesitation when the choice "
    "is thin or narrow. Output ONLY JSON.\n\n"
    "Weigh THREE factors explicitly and name them in your rationale:\n"
    "  (A) CENTRALITY — is this THE core idea of the lecture/segment, or a narrow "
    "sub-point or peripheral aside? A safe, easy, or tangential pick is not central.\n"
    "  (B) RICHNESS / NON-TRIVIALITY — is the concept deep and multi-faceted (a "
    "mechanism with moving parts, a space to explore, a non-obvious behaviour), or "
    "a single simple fact / one-parameter point that is quick to state?\n"
    "  (C) INTERACTIVITY PAYOFF — does making it RUNNABLE and TWEAKABLE unlock "
    "understanding a static slide cannot, by letting the learner explore a space, "
    "watch dynamics, or discover a relationship? Distinguish genuine exploration "
    "('manipulate X and discover Y') from mere OBSERVATION ('watch a real model do "
    "its thing' / 'press run and read the number') — observation is a weaker payoff "
    "than exploration.\n\n"
    "CALIBRATION (use the full range):\n"
    "  5 = central AND rich AND genuinely suited to interaction — manipulating or "
    "running it unlocks intuition a static slide cannot give. The best, most "
    "ambitious choices earn this; award it when all three factors are strong (e.g. "
    "exploring a mechanism's internals, a space, or learned dynamics).\n"
    "  4 = a central, worthwhile concept with a clear interactivity benefit, but "
    "either fairly foundational/standard, OR the payoff is good-not-exceptional, OR "
    "it leans on observation more than exploration. A solid, sensible choice.\n"
    "  3 = a valid and relevant concept that is somewhat narrow/peripheral, OR one "
    "where interactivity adds only modest value over a static figure; a safe pick.\n"
    "  2 = a peripheral, thin, or one-note sub-point, or a concept poorly suited to "
    "an interactive notebook.\n"
    "  1 = trivial/tangential, or the wrong medium entirely (better as prose/a "
    "single static figure).\n\n"
    "STAY IN YOUR LANE: judge ONLY the CHOICE — (a) whether the concept is worth "
    "demoing and (b) whether an interactive code notebook is the right medium. Do "
    "NOT penalize implementation quality, correctness, clarity, or whether the demo "
    "actually ran — those are SEPARATE metrics. A worthy concept scores on its "
    "merits even if THIS implementation is flawed or broken."
) + TRUNCATION_RULE

_USER_TMPL = """SOURCE MATERIALS (slides):
{slides}

{transcript}NOTEBOOK (full content):
{notebook}

Return JSON:
{{
  "worthiness": <1-5>,
  "interactive_is_right_tool": <bool>,
  "rationale": "<what concept it chose; why (not) worthy; tool fit>",
  "flags": ["<e.g. safe_easy_concept, central_concept, better_as_static>"]
}}"""


def judge_topic(judge: LLMClient, notebook_text: str, slides: str,
                transcript: str = "") -> tuple[TopicOutput, dict]:
    tr = f"SOURCE MATERIALS (transcript):\n{transcript}\n\n" if transcript else ""
    user = _USER_TMPL.format(slides=slides or "(none provided)", transcript=tr,
                            notebook=truncate(notebook_text, 60000, "the notebook"))
    res = judge.complete(system=_SYSTEM, user=user, json_mode=True,
                         role_tag="topic",
                         required={"worthiness": "score", "rationale": "str",
                                   "interactive_is_right_tool": "bool"})
    d = res.parsed_json
    out = TopicOutput(
        worthiness=score_of(d, "worthiness"),
        interactive_is_right_tool=bool(d.get("interactive_is_right_tool", True)),
        rationale=str(d.get("rationale", "")),
        flags=[str(f) for f in d.get("flags", [])] if isinstance(
            d.get("flags"), list) else [])
    info = {"model": res.model, "provider": res.provider,
            "usage": {"in": res.usage.input_tokens, "out": res.usage.output_tokens}}
    return out, info
