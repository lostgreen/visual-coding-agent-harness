"""Text-only answer synthesis over compact evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend


@dataclass(frozen=True)
class AnswerAgentResult:
    status: str
    answer: str = ""
    rationale: str = ""
    citations: Sequence[str] = field(default_factory=list)
    missing_evidence: Sequence[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_text: str = ""


class AnswerAgent:
    """Generate a final answer from evidence text without raw video access."""

    def __init__(self, backend: VisionLanguageBackend) -> None:
        self.backend = backend

    def run(self, *, question: str, evidence_text: str) -> AnswerAgentResult:
        response = self.backend.generate(
            BackendRequest(
                task="answer_from_evidence",
                prompt=_answer_prompt(question=question, evidence_text=evidence_text),
                max_new_tokens=512,
            )
        )
        return _parse_answer_response(response.text)


def _answer_prompt(*, question: str, evidence_text: str) -> str:
    return (
        "You are the Answer Agent for a long-video evidence workspace.\n"
        "Use only the evidence table below. Do not use raw video or outside knowledge.\n"
        "Return only JSON with this schema:\n"
        '{"answer": string, "rationale": string, "citations": [observation_id], '
        '"missing_evidence": [string], "confidence": number}\n'
        "Rules:\n"
        "- Multiple-choice answers must start with exactly one option letter.\n"
        "- Cite at least one visual/ASR/OCR/QA observation id from the evidence.\n"
        '- If evidence is insufficient, set answer to "need_more_evidence" and explain missing_evidence.\n'
        "- Do not cite navigation-only evidence as sole support.\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence_text}\n"
    )


def _parse_answer_response(text: str) -> AnswerAgentResult:
    try:
        payload = json.loads(_extract_json_object(text))
    except (json.JSONDecodeError, ValueError) as exc:
        return AnswerAgentResult(
            status="need_more_evidence",
            missing_evidence=[f"answer_json_parse_failed: {type(exc).__name__}"],
            raw_text=text,
        )

    answer = str(payload.get("answer", "")).strip()
    citations = [str(item) for item in payload.get("citations", [])]
    missing_evidence = [str(item) for item in payload.get("missing_evidence", [])]
    status = "need_more_evidence" if answer.lower() == "need_more_evidence" or not citations else "final"
    return AnswerAgentResult(
        status=status,
        answer=answer,
        rationale=str(payload.get("rationale", "")),
        citations=citations,
        missing_evidence=missing_evidence,
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        raw_text=text,
    )


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return stripped[start : end + 1]
