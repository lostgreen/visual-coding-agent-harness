"""Model-owned final decision helpers.

This module is deliberately small and side-effect free. Runtime code may use it
to parse and label final decisions, but it must not promote framework guesses
into submitted answers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class FinalDecisionOwner(str, Enum):
    MODEL = "model"
    FORMAT_REPAIR = "format_repair"
    NONE = "none"
    FRAMEWORK = "framework"


@dataclass(frozen=True)
class FinalDiagnostics:
    status: str
    reason_code: str = ""
    repair_hint: str = ""
    supporting_evidence_ids: Sequence[str] = field(default_factory=tuple)
    missing_target_refs: Sequence[str] = field(default_factory=tuple)
    missing_relation_refs: Sequence[str] = field(default_factory=tuple)
    verifier_status: str = ""
    verifier_answer: str = ""
    verifier_disagrees: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "repair_hint": self.repair_hint,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "missing_target_refs": list(self.missing_target_refs),
            "missing_relation_refs": list(self.missing_relation_refs),
            "verifier_status": self.verifier_status,
            "verifier_answer": self.verifier_answer,
            "verifier_disagrees": self.verifier_disagrees,
        }


@dataclass(frozen=True)
class ModelFinalDecision:
    status: str
    answer: str = ""
    citations: Sequence[str] = field(default_factory=tuple)
    evidence_ids: Sequence[str] = field(default_factory=tuple)
    confidence: float = 0.0
    rationale: str = ""
    evidence_sufficiency: str = ""
    reason: str = ""
    raw_text: str = ""
    owner: FinalDecisionOwner = FinalDecisionOwner.NONE

    @property
    def is_final(self) -> bool:
        return self.status == "final" and self.owner in {
            FinalDecisionOwner.MODEL,
            FinalDecisionOwner.FORMAT_REPAIR,
        }


def parse_model_final_response(
    text: str,
    *,
    allowed_options: Sequence[str] = (),
    locked_answer: str = "",
) -> ModelFinalDecision:
    """Parse a model final-decision payload without inferring an answer.

    The parser accepts only explicit JSON final decisions. A locked answer, when
    provided after a format repair attempt, must match the parsed answer.
    """

    raw_text = str(text or "")
    try:
        payload = json.loads(_extract_json_object(raw_text))
    except (json.JSONDecodeError, ValueError) as exc:
        return ModelFinalDecision(status="invalid", reason=f"json_parse_failed:{type(exc).__name__}", raw_text=raw_text)
    if not isinstance(payload, Mapping):
        return ModelFinalDecision(status="invalid", reason="payload_not_object", raw_text=raw_text)
    return parse_model_final_payload(payload, allowed_options=allowed_options, locked_answer=locked_answer, raw_text=raw_text)


def parse_model_final_payload(
    payload: Mapping[str, Any],
    *,
    allowed_options: Sequence[str] = (),
    locked_answer: str = "",
    raw_text: str = "",
) -> ModelFinalDecision:
    status = str(payload.get("status", "")).strip().lower()
    if status in {"continue", "tool", "tools"} or "program" in payload:
        return ModelFinalDecision(status="invalid", reason="model_declined_final_with_program", raw_text=raw_text)
    if status in {"no_model_final", "need_more_evidence", "abstain"}:
        return ModelFinalDecision(status="no_model_final", reason=status, raw_text=raw_text)
    if status != "final":
        return ModelFinalDecision(status="invalid", reason=f"unsupported_status:{status or 'missing'}", raw_text=raw_text)

    answer = str(payload.get("answer", "")).strip()
    resolved_answer = normalize_final_answer(answer, allowed_options=allowed_options)
    if not resolved_answer:
        return ModelFinalDecision(status="invalid", reason="missing_or_invalid_answer", raw_text=raw_text)
    if locked_answer and normalize_final_answer(locked_answer, allowed_options=allowed_options) != resolved_answer:
        return ModelFinalDecision(status="invalid", reason="format_repair_answer_changed", raw_text=raw_text)
    return ModelFinalDecision(
        status="final",
        answer=resolved_answer,
        citations=_string_list(payload.get("citations", [])),
        evidence_ids=_string_list(payload.get("evidence_ids", [])),
        confidence=_float_value(payload.get("confidence", 0.0)),
        rationale=str(payload.get("rationale", "")),
        evidence_sufficiency=str(payload.get("evidence_sufficiency", "")),
        raw_text=raw_text,
        owner=FinalDecisionOwner.FORMAT_REPAIR if locked_answer else FinalDecisionOwner.MODEL,
    )


def recover_locked_answer_from_malformed_final(text: str, *, allowed_options: Sequence[str] = ()) -> str:
    """Return an explicit answer from malformed final text, or empty string.

    This is intentionally narrow: it only recovers an answer when the raw text
    visibly declares a final status and an answer field. It does not choose from
    prose or option evidence.
    """

    raw = str(text or "")
    if not re.search(r"['\"]?status['\"]?\s*:\s*['\"]?final['\"]?", raw, flags=re.IGNORECASE):
        return ""
    match = re.search(r"['\"]?answer['\"]?\s*:\s*['\"]([^'\"]+)['\"]", raw, flags=re.IGNORECASE)
    if match is None:
        return ""
    return normalize_final_answer(match.group(1), allowed_options=allowed_options)


def normalize_final_answer(answer: str, *, allowed_options: Sequence[str] = ()) -> str:
    text = str(answer or "").strip()
    if not allowed_options:
        return text
    allowed = {
        _option_letter(option): str(option).strip()
        for option in allowed_options
        if _option_letter(option)
    }
    letter = _option_letter(text)
    return letter if letter in allowed else ""


def _extract_json_object(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("no_json_object")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char != "}":
            continue
        depth -= 1
        if depth == 0:
            return stripped[start : index + 1]
    raise ValueError("unterminated_json_object")


def _option_letter(value: str) -> str:
    match = re.match(r"\s*([A-H])(?:[.)]\s*|\s+|$)", str(value or ""), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item)]


def _float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
