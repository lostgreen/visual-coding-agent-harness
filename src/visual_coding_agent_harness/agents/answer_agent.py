"""Text-only answer synthesis over compact evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend


@dataclass(frozen=True)
class AnswerAgentResult:
    status: str
    answer: str = ""
    rationale: str = ""
    citations: Sequence[str] = field(default_factory=list)
    missing_evidence: Sequence[str] = field(default_factory=list)
    confidence: float = 0.0
    conflict: Mapping[str, Any] = field(default_factory=dict)
    raw_text: str = ""


class AnswerAgent:
    """Generate a final answer from evidence text without raw video access."""

    def __init__(self, backend: VisionLanguageBackend) -> None:
        self.backend = backend

    def run(
        self,
        *,
        question: str,
        evidence_text: str = "",
        evidence_table: Mapping[str, Any] | None = None,
    ) -> AnswerAgentResult:
        if evidence_table is not None and _has_option_support(evidence_table):
            return arbitrate_evidence_table(evidence_table)
        response = self.backend.generate(
            BackendRequest(
                task="answer_from_evidence",
                prompt=_answer_prompt(question=question, evidence_text=evidence_text),
                max_new_tokens=512,
            )
        )
        return _parse_answer_response(response.text)


GROUNDING_WEIGHTS = {
    "visually_confirmed": 1.0,
    "inferred": 0.35,
    "weak": 0.2,
    "external_knowledge": 0.1,
}
WEAK_GROUNDING = {"inferred", "weak", "external_knowledge"}


def arbitrate_evidence_table(table: Mapping[str, Any], *, min_margin: float = 0.12) -> AnswerAgentResult:
    """Pick or abstain from an MCQ answer using the complete option-grouped evidence table."""

    option_text = _option_text_map(table.get("options", []))
    option_scores = _score_options(table)
    ranked = sorted(
        [(option, score) for option, score in option_scores.items() if option != "unassigned" and score > 0],
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        return _need_more_evidence("targeted follow-up needed: inspect the most relevant option-specific window.")

    winner, winning_score = ranked[0]
    runner_up, runner_up_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
    conflict_options = sorted(option for option, score in option_scores.items() if option != "unassigned" and score > 0)
    winning_rows = _sorted_option_rows(table, winner)
    strong_winning_rows = [row for row in winning_rows if not _is_weak_grounding(row)]
    conflict = {
        "options": conflict_options,
        "winner": winner,
        "runner_up": runner_up,
        "scores": {option: round(score, 3) for option, score in option_scores.items()},
    }

    if not strong_winning_rows:
        return _need_more_evidence(
            f"targeted follow-up needed: option {winner} is only supported by weak or inferred evidence.",
            conflict=conflict,
        )
    if runner_up and winning_score - runner_up_score < min_margin:
        return _need_more_evidence(
            f"targeted follow-up needed: resolve close support between options {winner} and {runner_up}.",
            conflict=conflict,
        )

    citation_rows = strong_winning_rows[:2]
    citations = [str(row.get("obs_id", "")) for row in citation_rows if row.get("obs_id")]
    answer = option_text.get(winner, winner)
    rationale = (
        f"Option {winner} has the strongest weighted support "
        f"({winning_score:.2f} vs {runner_up_score:.2f}). "
        f"Cited evidence is {', '.join(citations)}."
    )
    return AnswerAgentResult(
        status="final",
        answer=answer,
        rationale=rationale,
        citations=citations,
        confidence=min(1.0, winning_score),
        conflict=conflict,
    )


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


def _need_more_evidence(reason: str, *, conflict: Mapping[str, Any] | None = None) -> AnswerAgentResult:
    return AnswerAgentResult(
        status="need_more_evidence",
        answer="need_more_evidence",
        rationale=reason,
        missing_evidence=[reason],
        confidence=0.0,
        conflict=dict(conflict or {}),
    )


def _score_options(table: Mapping[str, Any]) -> dict[str, float]:
    groups = table.get("groups", {}) if isinstance(table.get("groups", {}), Mapping) else {}
    scores: dict[str, float] = {}
    for option, rows in groups.items():
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        score = 0.0
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            score += _row_score(row)
        scores[str(option)] = score
    return scores


def _has_option_support(table: Mapping[str, Any]) -> bool:
    return any(option != "unassigned" and score > 0 for option, score in _score_options(table).items())


def _row_score(row: Mapping[str, Any]) -> float:
    return float(row.get("confidence", 0.0) or 0.0) * GROUNDING_WEIGHTS.get(
        str(row.get("grounding_quality", "weak")),
        0.2,
    )


def _sorted_option_rows(table: Mapping[str, Any], option: str) -> list[Mapping[str, Any]]:
    groups = table.get("groups", {}) if isinstance(table.get("groups", {}), Mapping) else {}
    rows = groups.get(option, [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return sorted(
        [row for row in rows if isinstance(row, Mapping)],
        key=lambda row: (-_row_score(row), str(row.get("obs_id", ""))),
    )


def _is_weak_grounding(row: Mapping[str, Any]) -> bool:
    return str(row.get("grounding_quality", "weak")) in WEAK_GROUNDING


def _option_text_map(options: Any) -> dict[str, str]:
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        return {}
    mapping = {}
    for index, option in enumerate(options):
        text = str(option).strip()
        match = re.match(r"^([A-Za-z])(?:[\.)]\s*|\s+|$)", text)
        letter = match.group(1).upper() if match else chr(ord("A") + index)
        mapping[letter] = text or letter
    return mapping


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return stripped[start : end + 1]
