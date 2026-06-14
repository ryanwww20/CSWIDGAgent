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
    "You judge TOPIC-WORTHINESS. Given the source materials and the concept this "
    "notebook chose to demo, decide: (1) was that concept actually demo-worthy — "
    "central/important and the kind of thing that benefits from being made "
    "interactive/runnable — or a trivially easy / peripheral pick chosen to be "
    "safe? (2) Is an interactive code notebook the RIGHT tool for it, versus a "
    "static figure or prose? Reward choosing a concept where interaction or "
    "computation genuinely adds understanding; penalize trivial or poorly-suited "
    "choices.\n\n"
    "STAY IN YOUR LANE: judge ONLY (a) whether the chosen concept is worth demoing "
    "and (b) whether an interactive code notebook is the right medium for it. Do NOT "
    "penalize implementation quality, correctness, or whether the simulation is "
    "real/faithful — those are scored by SEPARATE metrics. A worthy concept in the "
    "right medium scores high even if THIS particular implementation is flawed. "
    "Output ONLY JSON. 5 = important concept, ideally suited to an interactive demo; "
    "1 = trivial/peripheral or a poor fit for interactivity."
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
