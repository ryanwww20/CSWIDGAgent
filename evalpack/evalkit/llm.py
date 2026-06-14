"""Thin multi-provider LLM client for the interactivity harness.

The planner and the judge are deliberately kept in *different model families*
(anti-bias: the model that proposes the probes should not also grade them — see
docs/EVAL_TESTSET_DESIGN.md §6.1). This client abstracts OpenAI / Anthropic /
Gemini behind one `complete()` with text + image (vision) inputs and optional
JSON mode, so the planner and judge code is provider-agnostic.

Keys (any subset): OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY.
If no key is available the harness still runs (deterministic sweep only).
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResult:
    text: str
    parsed_json: Optional[Any]
    usage: LLMUsage
    model: str
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)


# ---- provider discovery --------------------------------------------------

def _provider_for(model: str) -> str:
    m = model.lower()
    if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gemini"
    raise ValueError(f"cannot infer provider for model={model!r}")


def _has_key(provider: str) -> bool:
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "gemini": bool(os.environ.get("GEMINI_API_KEY")
                       or os.environ.get("GOOGLE_API_KEY")),
    }.get(provider, False)


def available_providers() -> list[str]:
    return [p for p in ("openai", "anthropic", "gemini") if _has_key(p)]


# Default model per provider (kept current-ish; override via env).
_DEFAULT_MODEL = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-2.5-flash",
}


def pick_models() -> tuple[Optional[str], Optional[str]]:
    """Choose (planner_model, judge_model), honouring env overrides and keeping
    them in different families when more than one provider key is present."""
    planner = os.environ.get("EVAL_PLANNER_MODEL")
    judge = os.environ.get("EVAL_JUDGE_MODEL")
    provs = available_providers()
    if not provs and not (planner or judge):
        return None, None
    if planner is None and provs:
        planner = _DEFAULT_MODEL[provs[0]]
    if judge is None and provs:
        # prefer a different family than the planner if we can
        planner_prov = _provider_for(planner) if planner else None
        other = [p for p in provs if p != planner_prov]
        judge = _DEFAULT_MODEL[other[0] if other else provs[0]]
    return planner, judge


def pick_verifier(judge_model: Optional[str]) -> Optional[str]:
    """A model in a DIFFERENT family than the judge (so it can't rubber-stamp
    its own cited error). Falls back to any available provider."""
    env = os.environ.get("EVAL_VERIFIER_MODEL")
    if env:
        return env
    provs = available_providers()
    if not provs:
        return None
    jp = _provider_for(judge_model) if judge_model else None
    other = [p for p in provs if p != jp]
    return _DEFAULT_MODEL[other[0] if other else provs[0]]


# ---- client --------------------------------------------------------------

class LLMClient:
    def __init__(self, model: str, *, temperature: float = 0.2,
                 max_tokens: int = 4000, timeout: int = 180) -> None:
        self.model = model
        self.provider = _provider_for(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._is_reasoning = self.provider == "openai" and model.lower().startswith(
            ("o1", "o3", "o4", "gpt-5"))
        # Set when an Anthropic model rejects assistant-prefill (newer models);
        # remembered per-client so we don't pay a failed request every call.
        self._no_prefill = False
        if self._is_reasoning:
            # Reasoning models spend hidden tokens before emitting output; a low
            # budget yields empty/truncated JSON. Give it room.
            self.max_tokens = max(self.max_tokens, 16000)
        if not _has_key(self.provider):
            raise RuntimeError(
                f"no API key for provider {self.provider!r} (model {model!r})")

    # public --------------------------------------------------------------
    def complete(self, *, system: str, user: str,
                 images: list[Path] | None = None, json_mode: bool = False,
                 role_tag: str = "",
                 required: Optional[dict[str, str]] = None) -> LLMResult:
        """One LLM call with retries.

        `required` (json_mode only) is a {field: type_spec} contract the parsed
        JSON must satisfy — type_spec one of "score" (number in [1,5]), "number",
        "str", "list", "dict", "bool". On violation the call is RETRIED with the
        validation problems quoted back to the model; if every attempt fails a
        RuntimeError is raised so the caller renders the metric NA instead of
        silently scoring a parse failure as 0.
        """
        last: Optional[BaseException] = None
        user_cur = user
        for attempt in range(4):
            try:
                if self.provider == "openai":
                    res = self._openai(system, user_cur, images, json_mode)
                elif self.provider == "anthropic":
                    res = self._anthropic(system, user_cur, images, json_mode)
                else:
                    res = self._gemini(system, user_cur, images, json_mode)
            except ModuleNotFoundError as e:
                # A missing provider SDK won't fix itself on retry — fail fast.
                raise RuntimeError(
                    f"provider SDK missing for {self.provider} ({self.model}): {e}. "
                    f"Install it in the ml-colab-eval env.") from e
            except Exception as e:  # noqa: BLE001
                last = e
                _log_call(role_tag, self.model, self.provider, attempt, system,
                          user_cur, images, error=str(e))
                time.sleep(min(2 ** attempt, 20))
                continue
            if json_mode and res.parsed_json is None:
                res.parsed_json = _extract_json(res.text)
            if json_mode and required:
                problems = validate_json(res.parsed_json, required)
                if problems:
                    last = RuntimeError(
                        f"invalid JSON from {self.model}: {'; '.join(problems)} "
                        f"(text head: {res.text[:200]!r})")
                    _log_call(role_tag, self.model, self.provider, attempt,
                              system, user_cur, images, result=res,
                              rejected="; ".join(problems))
                    user_cur = user + (
                        "\n\nIMPORTANT — your previous reply was rejected: "
                        + "; ".join(problems)
                        + ". Respond again with ONLY one valid JSON object "
                          "containing all required fields. No prose, no markdown "
                          "fences.")
                    continue
            _log_call(role_tag, self.model, self.provider, attempt, system,
                      user_cur, images, result=res)
            return res
        raise RuntimeError(
            f"LLM call failed (role={role_tag}, model={self.model}): {last!r}") from last

    # providers -----------------------------------------------------------
    def _openai(self, system, user, images, json_mode) -> LLMResult:
        from openai import OpenAI
        client = OpenAI(timeout=self.timeout)
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for img in images or []:
            content.append({"type": "image_url",
                            "image_url": {"url": _data_url(img)}})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": content}],
            "max_completion_tokens": self.max_tokens,
        }
        if not self._is_reasoning:
            kwargs["temperature"] = self.temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        comp = client.chat.completions.create(**kwargs)
        text = comp.choices[0].message.content or ""
        u = getattr(comp, "usage", None)
        return LLMResult(text, None,
                         LLMUsage(getattr(u, "prompt_tokens", 0) or 0,
                                  getattr(u, "completion_tokens", 0) or 0),
                         comp.model, "openai")

    def _anthropic(self, system, user, images, json_mode) -> LLMResult:
        import anthropic
        client = anthropic.Anthropic(timeout=self.timeout)
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for img in images or []:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.b64encode(Path(img).read_bytes()).decode("ascii")}})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        prefilled = False
        if json_mode and not self._no_prefill:
            # Anthropic has no response_format; prefill "{" so the reply IS the
            # JSON object (no prose preamble, no markdown fence to strip).
            messages.append({"role": "assistant", "content": "{"})
            prefilled = True
        kwargs: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_tokens,
            "temperature": self.temperature, "system": system,
            "messages": messages}
        try:
            msg = client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            err = str(e).lower()
            # Newer models (e.g. Sonnet 4.6 / Opus 4.8) reject assistant
            # prefill and/or temperature — drop the offending part and retry.
            retry = False
            if prefilled and "prefill" in err:
                self._no_prefill = True
                prefilled = False
                kwargs["messages"] = messages[:-1]
                retry = True
            if "temperature" in err:
                kwargs.pop("temperature", None)
                retry = True
            if not retry:
                raise
            msg = client.messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        if prefilled:
            text = "{" + text
        return LLMResult(text, None,
                         LLMUsage(msg.usage.input_tokens, msg.usage.output_tokens),
                         self.model, "anthropic")

    def _gemini(self, system, user, images, json_mode) -> LLMResult:
        from google import genai
        from google.genai import types
        client = genai.Client()
        parts: list[Any] = [user]
        for img in images or []:
            parts.append(types.Part.from_bytes(
                data=Path(img).read_bytes(), mime_type="image/png"))
        cfg = types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json" if json_mode else None)
        resp = client.models.generate_content(
            model=self.model, contents=parts, config=cfg)
        meta = getattr(resp, "usage_metadata", None)
        return LLMResult(resp.text or "", None,
                         LLMUsage(getattr(meta, "prompt_token_count", 0) or 0,
                                  getattr(meta, "candidates_token_count", 0) or 0),
                         self.model, "gemini")


# ---- raw-call preservation -------------------------------------------------

def _log_call(role_tag: str, model: str, provider: str, attempt: int,
              system: str, user: str, images: list[Path] | None, *,
              result: "LLMResult | None" = None, error: str = "",
              rejected: str = "") -> None:
    """Append the full raw exchange to $EVAL_LLM_LOG (JSONL), when set.

    Every judge/planner/verifier call — including failed attempts and
    contract-rejected replies — is preserved verbatim so a scored run can be
    audited (what did the judge actually see and say?) without re-running it.
    """
    path = os.environ.get("EVAL_LLM_LOG")
    if not path:
        return
    rec: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "role": role_tag, "model": model, "provider": provider,
        "attempt": attempt,
        "system": system, "user": user,
        "images": [str(p) for p in images or []],
    }
    if result is not None:
        rec.update(response=result.text, parsed_ok=result.parsed_json is not None,
                   usage={"in": result.usage.input_tokens,
                          "out": result.usage.output_tokens})
    if error:
        rec["error"] = error
    if rejected:
        rec["rejected"] = rejected
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not write LLM log {path}: {e}")


# ---- helpers --------------------------------------------------------------

def validate_json(parsed: Any, required: dict[str, str]) -> list[str]:
    """Check a parsed judge reply against its field contract.

    Returns a list of human-readable problems (empty = valid). Used by
    LLMClient.complete(required=...) to drive corrective retries.
    """
    if not isinstance(parsed, dict):
        return [f"reply is not a JSON object (got {type(parsed).__name__})"]
    problems: list[str] = []
    for key, spec in required.items():
        if key not in parsed or parsed[key] is None:
            problems.append(f"missing required field {key!r}")
            continue
        v = parsed[key]
        if spec in ("score", "number"):
            try:
                f = float(v)
            except (TypeError, ValueError):
                problems.append(f"field {key!r} must be a number, got {v!r}")
                continue
            if spec == "score" and not (1.0 <= f <= 5.0):
                problems.append(f"field {key!r} must be a score in [1, 5], got {v!r}")
        elif spec == "str" and not isinstance(v, str):
            problems.append(f"field {key!r} must be a string")
        elif spec == "list" and not isinstance(v, list):
            problems.append(f"field {key!r} must be a list")
        elif spec == "dict" and not isinstance(v, dict):
            problems.append(f"field {key!r} must be an object")
        elif spec == "bool" and not isinstance(v, bool):
            problems.append(f"field {key!r} must be true/false")
    return problems


def score_of(d: dict, key: str) -> float:
    """Read a validated 1-5 score field (clamped defensively)."""
    return max(1.0, min(5.0, float(d[key])))


def _data_url(img: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(img).read_bytes()).decode("ascii")


def _extract_json(text: str) -> Optional[Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # last resort: grab the outermost {...} or [...]
    for lo, hi in (("{", "}"), ("[", "]")):
        i, j = text.find(lo), text.rfind(hi)
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None
