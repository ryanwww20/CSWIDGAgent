"""Metric #2 — Faithfulness & Correctness (docs/EVAL_TESTSET_DESIGN.md §6.0).

Three 1-5 sub-scores from a code-aware judge, composed by HARMONIC mean:
  2a assertional  — nothing CONTRADICTS the source or is fabricated/wrong
                    (correct elaborations beyond the slides are fine, not penalized).
  2b computational— the code DERIVES outcomes; not a tunable just-so story whose
                    conclusion is hardcoded via constants / thresholds / proxy-success.
  2c correctness  — correct in truth per the field.

BURDEN OF PROOF applies to 2a AND 2c: a sub-5 score must cite concrete, checkable
errors (labeled with their axis), and each cited error is independently confirmed
by a separate VERIFIER agent before the deduction stands. Partially-confirmed
deductions are scaled by the confirmed fraction; with no verifier configured,
unverified deductions are discarded (revert toward 5) per the documented rule,
and the run is flagged. 2b is exempt: it cites the notebook's own code, which the
judge reads directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from llm import LLMClient, score_of
from schemas import FaithfulnessOutput, harmonic_mean
from textbudget import TRUNCATION_RULE, truncate

_SYSTEM = (
    "You evaluate a teaching notebook's FAITHFULNESS & CORRECTNESS against the "
    "source materials it was built from. You read the full notebook (markdown + "
    "code + outputs) and the source (slides/transcript). Output ONLY JSON. Score "
    "three INDEPENDENT 1-5 axes:\n\n"
    "2a ASSERTIONAL: does any text/label/claim/result CONTRADICT the source, or is "
    "it FABRICATED / unsupported-and-wrong? Adding *correct* context beyond the "
    "slides is GOOD — do NOT penalize it. Illustrative, simulated, or toy values "
    "chosen for a demo are NOT 'fabrications' here unless they are presented as "
    "real-world facts that contradict the source. Penalize only contradictions and "
    "hallucinations. BURDEN OF PROOF: only score below 5 if you list each "
    "contradiction/fabrication in `errors` with axis=\"assertional\", QUOTING the "
    "source text it contradicts in `source_evidence`. If you cannot cite one, "
    "score 5. 5=nothing contradicts/fabricated; 1=central claims contradict.\n\n"
    "2b COMPUTATIONAL: does the CODE actually implement the mechanism it depicts, "
    "or are outcomes ASSERTED via tuned constants / hard-coded thresholds / a "
    "'success' proxy that never checks the real task? Look for: outcome-determining "
    "magic numbers; an 'ours' option winning by a fixed margin; sharp cliffs at "
    "round numbers; success=(count>=k) proxies; inputs that should matter being "
    "ignored. 5=outcomes genuinely derived from a faithful model; 1=tunable "
    "just-so story whose conclusion is hardcoded.\n\n"
    "2c CORRECTNESS-IN-TRUTH: independent of the source, are the EXPLAINED CONCEPTS "
    "and CLAIMS correct per the field (catches errors the source itself may "
    "contain)? SCOPE — this is ONLY about whether the stated ideas are true in the "
    "world. Do NOT raise implementation realism, hardcoded constants, tuned "
    "formulas, magic numbers, or 'the simulation is toy / the result is "
    "predetermined' here: those are 2b concerns and must NOT lower 2c. A demo whose "
    "code fakes its outcome can still teach a perfectly correct concept — if the "
    "concept is right, 2c = 5. A valid 2c error is a statement that is wrong in the "
    "world (a wrong definition, a wrong formula in the explanatory text, a claim "
    "that contradicts established theory) — NOT a criticism of how a number was "
    "computed. BURDEN OF PROOF: only score below 5 if you cite such a CONCRETE, "
    "CHECKABLE conceptual error in `errors` with axis=\"correctness\". If you "
    "cannot, score 5.\n\n"
    "Every entry in `errors` MUST carry an `axis` field: \"assertional\" for "
    "source contradictions/fabrications (with the contradicted source quoted in "
    "`source_evidence`), \"correctness\" for wrong-in-the-world claims (with the "
    "correct fact in `established_fact`). Each cited error will be independently "
    "fact-checked; deductions whose errors are not confirmed are discarded."
) + TRUNCATION_RULE

_USER_TMPL = """SOURCE MATERIALS (slides):
{slides}

{transcript}NOTEBOOK (full content; images attached separately):
{notebook}

Return JSON:
{{
  "assertional": <1-5>,
  "computational": <1-5>,
  "correctness": <1-5>,
  "rationale": "<cite cell numbers, code snippets, slide numbers>",
  "errors": [
    {{"axis": "<assertional|correctness>",
      "claim": "<what the notebook asserts>",
      "error": "<the concrete, checkable error>",
      "established_fact": "<correctness axis: the correct fact and why the claim is wrong>",
      "source_evidence": "<assertional axis: the quoted source text the claim contradicts>",
      "where": "<cell / location>"}}
  ],
  "flags": ["<e.g. hardcoded_narrative, proxy_success, contradicts_slide_N>"]
}}"""

_VERIFY_SYSTEM = (
    "You are a strict fact-checker. You receive a CLAIM made by a teaching notebook "
    "and an ALLEGED ERROR a reviewer flagged about it. Two kinds:\n"
    "- axis=correctness: the reviewer says the claim is wrong in the world, with "
    "their stated correct fact. Confirm true ONLY if the claim is genuinely "
    "incorrect per established knowledge AND the reviewer's correction is itself "
    "correct.\n"
    "- axis=assertional: the reviewer says the claim contradicts the quoted source "
    "material or fabricates a source attribution. Confirm true ONLY if the quoted "
    "source evidence genuinely contradicts the claim (or the claimed attribution "
    "is genuinely absent/false) — not when the claim is merely an elaboration "
    "beyond what the source covers.\n"
    "Never confirm matters of opinion, reasonable simplifications, or cases where "
    "the reviewer themselves is wrong. "
    "Output ONLY JSON: {\"confirmed\": <bool>, \"reason\": \"<short>\"}."
)


def _verify_errors(verifier: LLMClient, errors: list[dict]) -> list[dict]:
    verdicts: list[dict] = []
    for e in errors:
        user = (f"AXIS: {e.get('axis', 'correctness')}\n"
                f"CLAIM: {e.get('claim','')}\n"
                f"ALLEGED ERROR: {e.get('error','')}\n"
                f"REVIEWER'S CORRECT FACT: {e.get('established_fact','')}\n"
                f"QUOTED SOURCE EVIDENCE: {e.get('source_evidence','')}")
        try:
            res = verifier.complete(system=_VERIFY_SYSTEM, user=user,
                                    json_mode=True, role_tag="verifier",
                                    required={"confirmed": "bool",
                                              "reason": "str"})
            d = res.parsed_json
            verdicts.append({**e, "confirmed": bool(d["confirmed"]),
                             "reason": str(d.get("reason", ""))})
        except Exception as ex:  # noqa: BLE001
            # If the verifier fails, be conservative: do NOT confirm.
            verdicts.append({**e, "confirmed": False, "reason": f"verifier_error: {ex}"})
    return verdicts


def _final_axis_score(raw: float, errs: list[dict],
                      verdicts: Optional[list[dict]],
                      flags: list[str], axis: str) -> float:
    """Apply burden-of-proof to one axis (2a or 2c).

    The judge priced ALL its cited errors into `raw`; only confirmed errors may
    keep their share of the deduction. verdicts=None means no verifier was
    configured -> documented rule: unverified deductions are discarded.
    """
    if raw >= 5.0:
        return raw
    if not errs:
        flags.append(f"{axis}_reverted_no_cited_error")
        return 5.0
    if verdicts is None:
        flags.append(f"{axis}_unverified_errors_discarded_no_verifier")
        return 5.0
    confirmed = sum(1 for v in verdicts if v.get("confirmed"))
    if confirmed == 0:
        return 5.0
    if confirmed == len(errs):
        return raw
    flags.append(f"{axis}_deduction_scaled_{confirmed}of{len(errs)}_confirmed")
    return round(5.0 - (5.0 - raw) * confirmed / len(errs), 2)


def judge_faithfulness(judge: LLMClient, notebook_text: str,
                       notebook_images: list[str], slides: str,
                       transcript: str = "", *,
                       verifier: Optional[LLMClient] = None
                       ) -> tuple[FaithfulnessOutput, dict]:
    tr = f"SOURCE MATERIALS (transcript):\n{transcript}\n\n" if transcript else ""
    user = _USER_TMPL.format(slides=slides or "(none provided)", transcript=tr,
                            notebook=truncate(notebook_text, 60000, "the notebook"))
    res = judge.complete(system=_SYSTEM, user=user,
                         images=[Path(p) for p in notebook_images],
                         json_mode=True, role_tag="faithfulness",
                         required={"assertional": "score",
                                   "computational": "score",
                                   "correctness": "score",
                                   "rationale": "str", "errors": "list"})
    d = res.parsed_json

    a_raw = score_of(d, "assertional")
    b = score_of(d, "computational")
    c_raw = score_of(d, "correctness")
    errors = [e for e in d["errors"] if isinstance(e, dict)]
    for e in errors:
        if e.get("axis") not in ("assertional", "correctness"):
            e["axis"] = "correctness"   # legacy default: unlabeled = 2c
    a_errors = [e for e in errors if e["axis"] == "assertional"]
    c_errors = [e for e in errors if e["axis"] == "correctness"]

    # Propose -> verify (2a AND 2c): a sub-5 score only keeps the share of its
    # deduction backed by CONFIRMED errors. 2b is judged from the code itself.
    a_verified = (_verify_errors(verifier, a_errors)
                  if verifier is not None and a_raw < 5 and a_errors else None)
    c_verified = (_verify_errors(verifier, c_errors)
                  if verifier is not None and c_raw < 5 and c_errors else None)

    flags = [str(f) for f in d.get("flags", [])] if isinstance(
        d.get("flags"), list) else []
    a_final = _final_axis_score(a_raw, a_errors, a_verified, flags, "2a")
    c_final = _final_axis_score(c_raw, c_errors, c_verified, flags, "2c")
    verified = (a_verified or []) + (c_verified or [])

    score = harmonic_mean([a_final, b, c_final]) or 0.0
    out = FaithfulnessOutput(
        assertional=a_final, computational=b, correctness=c_final, score=score,
        rationale=str(d.get("rationale", "")), errors_cited=errors,
        errors_verified=verified, flags=flags)
    info = {"model": res.model, "provider": res.provider,
            "usage": {"in": res.usage.input_tokens, "out": res.usage.output_tokens},
            "assertional_raw": a_raw, "assertional_final": a_final,
            "correctness_raw": c_raw, "correctness_final": c_final,
            "n_errors": len(errors),
            "n_confirmed": sum(1 for v in verified if v.get("confirmed"))}
    return out, info
