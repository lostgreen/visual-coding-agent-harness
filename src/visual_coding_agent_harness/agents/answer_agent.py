"""Text-only answer synthesis over compact evidence."""

from __future__ import annotations

import json
import ast
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
    candidate_option_relations: Sequence[Mapping[str, Any]] = field(default_factory=list)
    missing_evidence: Sequence[str] = field(default_factory=list)
    confidence: float = 0.0
    conflict: Mapping[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    def has_partial_support(self) -> bool:
        return any(_is_visual_support_relation(relation) for relation in self.candidate_option_relations)

    def as_low_confidence_final(self) -> "AnswerAgentResult":
        supports = [relation for relation in self.candidate_option_relations if _is_visual_support_relation(relation)]
        if not supports:
            return self
        counts: dict[str, list[float]] = {}
        for relation in supports:
            option = str(relation.get("option", "")).strip().upper()[:1]
            if not option:
                continue
            counts.setdefault(option, []).append(float(relation.get("strength", relation.get("confidence", 0.0)) or 0.0))
        if not counts:
            return self
        option, strengths = sorted(counts.items(), key=lambda item: (-len(item[1]), -sum(item[1]), item[0]))[0]
        winning_supports = [
            relation
            for relation in supports
            if str(relation.get("option", "")).strip().upper().startswith(option)
        ]
        answer_grade = any(_is_answer_grade_relation(relation) for relation in winning_supports)
        confidence = (sum(strengths) / len(strengths)) * 0.7 if strengths else 0.0
        if not answer_grade:
            confidence = min(confidence, 0.5)
        citations = [
            str(relation.get("observation_id", ""))
            for relation in winning_supports
            if str(relation.get("observation_id", ""))
        ]
        rationale_suffix = "" if answer_grade else " (navigation-only, not answer-grade)"
        return AnswerAgentResult(
            status="low_confidence_final",
            answer=option,
            rationale=f"Follow-up budget exhausted; option {option} has partial support{rationale_suffix}.",
            citations=citations,
            candidate_option_relations=list(self.candidate_option_relations),
            missing_evidence=list(self.missing_evidence),
            confidence=confidence,
            conflict=dict(self.conflict),
            raw_text=self.raw_text,
        )


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
        hypothesis_option: str = "",
    ) -> AnswerAgentResult:
        if evidence_table is not None and _has_option_support(evidence_table):
            return arbitrate_evidence_table(evidence_table, hypothesis_option=hypothesis_option)
        response = self.backend.generate(
            BackendRequest(
                task="answer_from_evidence",
                prompt=_answer_prompt(question=question, evidence_text=evidence_text),
                max_new_tokens=512,
            )
        )
        return _parse_answer_response(response.text)


GROUNDING_WEIGHTS = {
    "global_sparse": 0.35,
    "visually_confirmed": 1.0,
    "indexed_transcript": 0.85,
    "inferred": 0.35,
    "weak": 0.2,
    "external_knowledge": 0.1,
}
WEAK_GROUNDING = {"global_sparse", "inferred", "weak", "external_knowledge"}


def arbitrate_evidence_table(
    table: Mapping[str, Any],
    *,
    min_margin: float = 0.12,
    hypothesis_option: str = "",
) -> AnswerAgentResult:
    """Pick or abstain from an MCQ answer using the complete option-grouped evidence table."""

    option_text = _option_text_map(table.get("options", []))
    mutex_conflict = _strong_mutex_conflict(table)
    if mutex_conflict is not None:
        left, right, mutex_group_id = mutex_conflict
        reason = f"mutex_conflict: {left.get('obs_id', '')} vs {right.get('obs_id', '')}"
        return _need_more_evidence(
            reason,
            conflict={
                "mutex_group_id": mutex_group_id,
                "left": dict(left),
                "right": dict(right),
            },
        )
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
            candidate_option_relations=_partial_support_relations(table, options=[winner]),
        )
    if runner_up and winning_score - runner_up_score < min_margin:
        return _need_more_evidence(
            f"targeted follow-up needed: resolve close support between options {winner} and {runner_up}.",
            conflict=conflict,
            candidate_option_relations=_partial_support_relations(table, options=[winner, runner_up]),
        )
    if runner_up and _options_overlap(option_text.get(winner, winner), option_text.get(runner_up, runner_up)):
        if not _has_distinguishing_evidence(
            winner_text=option_text.get(winner, winner),
            runner_up_text=option_text.get(runner_up, runner_up),
            rows=strong_winning_rows,
        ):
            return _need_more_evidence(
                "disambiguate_overlapping_options",
                conflict=conflict,
                candidate_option_relations=_partial_support_relations(table, options=[winner, runner_up]),
            )
    hypothesis_letter = _option_letter(hypothesis_option)
    if hypothesis_letter and hypothesis_letter != winner:
        return _need_more_evidence(
            f"hypothesis_disagreement: hypothesis option {hypothesis_letter} conflicts with evidence winner {winner}.",
            conflict={
                "type": "hypothesis_disagreement",
                "hypothesis_option": hypothesis_letter,
                "winner": winner,
                "runner_up": runner_up,
                "scores": {option: round(score, 3) for option, score in option_scores.items()},
            },
            candidate_option_relations=_partial_support_relations(table, options=[winner, hypothesis_letter]),
        )

    citation_rows = _temporal_citation_rows(strong_winning_rows) or strong_winning_rows[:2]
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
        '"candidate_option_relations": [{"option": "A", "relation": "support|contradict|neutral", '
        '"strength": number, "observation_id": "obs_0001", "rationale": string}], '
        '"missing_evidence": [string], "confidence": number}\n'
        "Rules:\n"
        "- Multiple-choice answers must start with exactly one option letter.\n"
        "- Cite at least one visual/ASR/OCR/QA observation id from the evidence.\n"
        "- Map facts to options only in candidate_option_relations; VisionAgent/local-worker text is not an option vote.\n"
        "- Every support relation must name the observation_id that directly supports it.\n"
        '- If evidence is insufficient, set answer to "need_more_evidence" and explain missing_evidence.\n'
        "- Do not cite navigation-only evidence as sole support.\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence_text}\n"
    )


def _parse_answer_response(text: str) -> AnswerAgentResult:
    try:
        payload = _parse_json_like_object(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return AnswerAgentResult(
            status="need_more_evidence",
            missing_evidence=[f"answer_json_parse_failed: {type(exc).__name__}"],
            raw_text=text,
        )

    answer = str(payload.get("answer", "")).strip()
    citations = [str(item) for item in payload.get("citations", [])]
    candidate_option_relations = _candidate_option_relations(payload.get("candidate_option_relations"))
    missing_evidence = [str(item) for item in payload.get("missing_evidence", [])]
    status = "need_more_evidence" if answer.lower() == "need_more_evidence" or not citations else "final"
    return AnswerAgentResult(
        status=status,
        answer=answer,
        rationale=str(payload.get("rationale", "")),
        citations=citations,
        candidate_option_relations=candidate_option_relations,
        missing_evidence=missing_evidence,
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        raw_text=text,
    )


def _option_letter(value: str) -> str:
    match = re.match(r"\s*([A-H])(?:[.)]\s*|\s+|$)", str(value or ""), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _parse_json_like_object(text: str) -> Mapping[str, Any]:
    raw_object = _extract_json_object(text)
    try:
        payload = json.loads(raw_object)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(raw_object)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("Answer response is not JSON-like") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Answer response JSON is not an object")
    return payload


def _need_more_evidence(
    reason: str,
    *,
    conflict: Mapping[str, Any] | None = None,
    candidate_option_relations: Sequence[Mapping[str, Any]] = (),
) -> AnswerAgentResult:
    return AnswerAgentResult(
        status="need_more_evidence",
        answer="need_more_evidence",
        rationale=reason,
        candidate_option_relations=list(candidate_option_relations),
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


def _partial_support_relations(
    table: Mapping[str, Any],
    *,
    options: Sequence[str] | None = None,
    max_per_option: int = 3,
) -> list[dict[str, Any]]:
    option_filter = {str(option).strip().upper()[:1] for option in options or [] if str(option).strip()}
    groups = table.get("groups", {}) if isinstance(table.get("groups", {}), Mapping) else {}
    relations: list[dict[str, Any]] = []
    for option in sorted(str(key).strip().upper()[:1] for key in groups if str(key) != "unassigned"):
        if option_filter and option not in option_filter:
            continue
        for row in _sorted_option_rows(table, option)[:max_per_option]:
            obs_id = str(row.get("obs_id", "")).strip()
            if not obs_id or _row_score(row) <= 0.0:
                continue
            try:
                strength = float(row.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                strength = 0.0
            if strength <= 0.0:
                strength = _row_score(row)
            relations.append(
                {
                    "option": option,
                    "relation": "support",
                    "strength": strength,
                    "observation_id": obs_id,
                    "grounding_quality": str(row.get("grounding_quality", "")),
                    "rationale": _compact_relation_rationale(str(row.get("claim", ""))),
                }
            )
    return relations


def _compact_relation_rationale(text: str, *, limit: int = 160) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."


def _temporal_citation_rows(rows: Sequence[Mapping[str, Any]], *, max_rows: int = 2) -> list[Mapping[str, Any]]:
    temporal_rows = [
        row
        for row in rows
        if str(row.get("event_label", "")).strip()
        and (row.get("observed_at_sec") is not None or row.get("time_range") is not None)
    ]
    if len(temporal_rows) < 2:
        return []
    temporal_rows.sort(key=lambda row: (_row_start_sec(row), str(row.get("obs_id", ""))))
    if max_rows <= 1:
        return temporal_rows[:1]
    return [temporal_rows[0], temporal_rows[-1]]


def _row_start_sec(row: Mapping[str, Any]) -> float:
    try:
        return float(row.get("observed_at_sec"))
    except (TypeError, ValueError):
        pass
    time_range = row.get("time_range")
    if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)) and time_range:
        try:
            return float(time_range[0])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _candidate_option_relations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    relations: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        relation = dict(item)
        option = str(relation.get("option", "")).strip()
        if not option:
            continue
        relation["option"] = option[:1].upper() if option[:1].isalpha() else option
        relation["relation"] = str(relation.get("relation", "support") or "support").strip().lower()
        try:
            relation["strength"] = float(relation.get("strength", 0.0) or 0.0)
        except (TypeError, ValueError):
            relation["strength"] = 0.0
        if relation["strength"] <= 0.0:
            relation["strength"] = 0.5
        relations.append(relation)
    return relations


def _is_visual_support_relation(relation: Mapping[str, Any]) -> bool:
    if str(relation.get("relation", "")).strip().lower() not in {"support", "supports", "supported"}:
        return False
    grounding = str(
        relation.get("grounding_quality")
        or relation.get("support_grounding_quality")
        or relation.get("grounding")
        or ""
    ).strip()
    return grounding in {"", "visually_confirmed", "indexed_transcript"}


def _is_answer_grade_relation(relation: Mapping[str, Any]) -> bool:
    grounding = str(
        relation.get("grounding_quality")
        or relation.get("support_grounding_quality")
        or relation.get("grounding")
        or ""
    ).strip()
    return grounding == "indexed_transcript" or bool(
        relation.get("evidence_id")
        or relation.get("evidence_binding")
        or relation.get("answer_grade")
        or relation.get("structured_support")
    )


def _strong_mutex_conflict(table: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], str] | None:
    by_mutex: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    groups = table.get("groups", {}) if isinstance(table.get("groups", {}), Mapping) else {}
    for group_option, rows in groups.items():
        if group_option == "unassigned" or not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if not isinstance(row, Mapping) or _is_weak_grounding(row):
                continue
            mutex_group_id = str(row.get("mutex_group_id", "")).strip()
            if not mutex_group_id:
                continue
            option = str(row.get("supported_option") or group_option).strip().upper()[:1]
            if not option:
                continue
            by_mutex.setdefault(mutex_group_id, []).append((option, row))
    for mutex_group_id, option_rows in by_mutex.items():
        for index, (left_option, left_row) in enumerate(option_rows):
            for right_option, right_row in option_rows[index + 1 :]:
                if left_option != right_option:
                    return left_row, right_row, mutex_group_id
    return None


def _has_option_support(table: Mapping[str, Any]) -> bool:
    return any(option != "unassigned" and score > 0 for option, score in _score_options(table).items())


def _row_score(row: Mapping[str, Any]) -> float:
    if _requires_visual_verification(row):
        return 0.0
    return float(row.get("confidence", 0.0) or 0.0) * GROUNDING_WEIGHTS.get(
        str(row.get("grounding_quality", "weak")),
        0.2,
    )


_OPTION_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "with",
    "without",
    "for",
    "from",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "why",
    "how",
    "what",
    "which",
    "option",
}


def _options_overlap(left: str, right: str, *, threshold: float = 0.6) -> bool:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens.intersection(right_tokens)) / max(1, min(len(left_tokens), len(right_tokens)))
    return overlap >= threshold


def _has_distinguishing_evidence(
    *,
    winner_text: str,
    runner_up_text: str,
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    distinguishing = _content_tokens(winner_text) - _content_tokens(runner_up_text)
    if not distinguishing:
        return False
    for row in rows:
        claim_tokens = _content_tokens(str(row.get("claim", "")))
        if distinguishing.intersection(claim_tokens):
            return True
    return False


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", str(text).lower())
        if len(token) >= 3 and token not in _OPTION_STOPWORDS
    }


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
    return _requires_visual_verification(row) or str(row.get("grounding_quality", "weak")) in WEAK_GROUNDING


def _requires_visual_verification(row: Mapping[str, Any]) -> bool:
    confidence_signal = str(row.get("confidence_signal", "")).strip().lower()
    return bool(row.get("requires_visual_verification")) or confidence_signal in {"text_inferred", "unverified"}


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
