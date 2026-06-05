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
        confidence = (sum(strengths) / len(strengths)) * 0.7 if strengths else 0.0
        citations = [
            str(relation.get("observation_id", ""))
            for relation in supports
            if str(relation.get("option", "")).strip().upper().startswith(option)
            and str(relation.get("observation_id", ""))
        ]
        return AnswerAgentResult(
            status="low_confidence_final",
            answer=option,
            rationale=f"Follow-up budget exhausted; option {option} has partial visually confirmed support.",
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
    "global_sparse": 1.0,
    "visually_confirmed": 1.0,
    "inferred": 0.35,
    "weak": 0.2,
    "external_knowledge": 0.1,
}
WEAK_GROUNDING = {"inferred", "weak", "external_knowledge"}


def arbitrate_evidence_table(table: Mapping[str, Any], *, min_margin: float = 0.12) -> AnswerAgentResult:
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
    global_floor = _global_floor_support(table)
    ranked = sorted(
        [(option, score) for option, score in option_scores.items() if option != "unassigned" and score > 0],
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        return _need_more_evidence("targeted follow-up needed: inspect the most relevant option-specific window.")

    if global_floor is not None:
        return _final_from_global_floor(
            option_text=option_text,
            option_scores=option_scores,
            global_floor=global_floor,
        )

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
    if runner_up and _options_overlap(option_text.get(winner, winner), option_text.get(runner_up, runner_up)):
        if not _has_distinguishing_evidence(
            winner_text=option_text.get(winner, winner),
            runner_up_text=option_text.get(runner_up, runner_up),
            rows=strong_winning_rows,
        ):
            return _need_more_evidence(
                "disambiguate_overlapping_options",
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
        payload = json.loads(_extract_json_object(text))
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


def _global_floor_support(table: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], float] | None:
    groups = table.get("groups", {}) if isinstance(table.get("groups", {}), Mapping) else {}
    candidates: list[tuple[str, Mapping[str, Any], float]] = []
    for option, rows in groups.items():
        if option == "unassigned" or not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("tool", "")) != "global_gist" and str(row.get("grounding_quality", "")) != "global_sparse":
                continue
            score = _row_score(row)
            if score > 0 and not _is_weak_grounding(row):
                candidates.append((str(option), row, score))
    if not candidates:
        return None
    if len(candidates) > 1:
        options = {option for option, _row, _score in candidates}
        if len(options) != 1:
            return None
    candidates.sort(key=lambda item: (-item[2], str(item[1].get("obs_id", ""))))
    return candidates[0]


def _final_from_global_floor(
    *,
    option_text: Mapping[str, str],
    option_scores: Mapping[str, float],
    global_floor: tuple[str, Mapping[str, Any], float],
) -> AnswerAgentResult:
    option, row, score = global_floor
    citations = [str(row.get("obs_id", ""))] if row.get("obs_id") else []
    conflict_options = sorted(option for option, value in option_scores.items() if option != "unassigned" and value > 0)
    conflict = {
        "options": conflict_options,
        "winner": option,
        "runner_up": "",
        "scores": {option: round(value, 3) for option, value in option_scores.items()},
        "global_floor": True,
    }
    return AnswerAgentResult(
        status="final",
        answer=option_text.get(option, option),
        rationale=(
            f"Option {option} is the global sparse whole-video floor "
            f"({score:.2f}); local evidence must not undercut it on gist questions."
        ),
        citations=citations,
        confidence=min(1.0, score),
        conflict=conflict,
    )


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
    return grounding in {"", "visually_confirmed"}


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
