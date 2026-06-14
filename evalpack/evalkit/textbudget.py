"""Honest text budgeting for judge inputs.

Every cap on what a judge sees must be VISIBLE to the judge: a silent
`text[:60000]` makes the judge score "the whole notebook/deck" while seeing an
arbitrary prefix. These helpers cut with an explicit marker, and TRUNCATION_RULE
is appended to every judge system prompt so withheld content is flagged
(`evidence_truncated`) instead of silently mis-scored.
"""
from __future__ import annotations

TRUNCATION_MARK = "[TRUNCATED"

# Appended to every quality-judge system prompt.
TRUNCATION_RULE = (
    "\n\nTRUNCATION: if any provided material contains a '[TRUNCATED ...]' or "
    "'[TRANSCRIPT WINDOW ...]' marker, part of that material was withheld from "
    "you for budget reasons. NEVER penalize or reward content you cannot see, "
    "and never treat its absence as evidence (e.g. 'the notebook never "
    "explains X' is invalid if the relevant part may be truncated). If a "
    "marker is present and it plausibly affects your judgment, add the flag "
    "'evidence_truncated'."
)


def truncate(text: str, limit: int, label: str = "content") -> str:
    """Head-cut at `limit` with a visible marker (never a silent cut)."""
    if len(text) <= limit:
        return text
    marker = (f"\n\n{TRUNCATION_MARK} — showing the first {limit:,} of "
              f"{len(text):,} chars of {label}; the rest was NOT shown to you.]")
    return text[:limit] + marker


def truncate_middle(text: str, limit: int, label: str = "content",
                    head_frac: float = 0.7) -> str:
    """Cut from the MIDDLE, keeping head + tail (for notebooks: the opening
    declaration cell and the closing summary matter most)."""
    if len(text) <= limit:
        return text
    head = int(limit * head_frac)
    tail = limit - head
    marker = (f"\n\n{TRUNCATION_MARK} — {len(text) - limit:,} chars omitted "
              f"from the MIDDLE of {label}; you are seeing its beginning and "
              f"end only.]\n\n")
    return text[:head] + marker + text[-tail:]


def is_truncated(text: str) -> bool:
    return TRUNCATION_MARK in text or "[TRANSCRIPT WINDOW" in text
