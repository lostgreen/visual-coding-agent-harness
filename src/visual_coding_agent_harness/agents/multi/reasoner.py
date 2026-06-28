"""Reasoner agent for hypothesis and answer decisions."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ...evidence.answer_operators import ROUTE_TO_MODALITIES, derive_answer_operator, derive_modality_route
from ..workspace_agent import WorkspaceRunResult
from .mutator import WorkspaceMutator
from .protocol import SubGoalBudget, SubGoalConstraint, SubGoalSuccessCriteria

MAX_OPEN_SUB_GOALS = 3
ANSWER_GROUNDING_KINDS = frozenset({"visual_support", "synthesized_support", "answer_conflict_resolved"})
CONTRADICTING_KINDS = frozenset({"answer_conflict", "contradiction", "contradicting"})
FINAL_GROUNDING_KINDS = frozenset({"visual_support", "synthesized_support", "answer_conflict_resolved"})


class ReasonerAgent:
    """Minimal Reasoner implementation for the first multi-agent runner slice."""

    def __init__(
        self,
        *,
        backend: Any,
        mutator: WorkspaceMutator,
        workspace: Any,
        video_map: Any,
        log_root: Any,
    ) -> None:
        self.backend = backend
        self.mutator = mutator
        self.workspace = workspace
        self.video_map = video_map
        self.log_root = log_root
        self.answer_result: WorkspaceRunResult | None = None

    def step(self, *, round_number: int, question: str, options: Mapping[str, str]) -> bool:
        """Schedule option checks, then answer only from option-bound positive evidence."""

        if self.answer_result is not None:
            return False
        option_ids = _option_ids(options)
        sub_goals = self.mutator.sub_goals()
        active = [goal for goal in sub_goals if goal.status in {"open", "in_progress"}]
        active_options = {
            str(goal.constraint.option_id or "").strip().upper()[:1]
            for goal in active
            if str(goal.constraint.option_id or "").strip()
        }
        tested_options = {
            str(goal.constraint.option_id or "").strip().upper()[:1]
            for goal in sub_goals
            if goal.status in {"done", "abandoned"} and str(goal.constraint.option_id or "").strip()
        }
        untested_options = [option_id for option_id in option_ids if option_id not in tested_options | active_options]
        if untested_options and len(active) < MAX_OPEN_SUB_GOALS:
            created = 0
            for option_id in untested_options[: MAX_OPEN_SUB_GOALS - len(active)]:
                self._create_option_sub_goal(
                    option_id=option_id,
                    option_text=options.get(option_id, ""),
                    question=question,
                    round_number=round_number,
                )
                created += 1
            self.workspace.write_trace_event(
                "reasoner_action_emitted",
                {"round": round_number, "action": "emit_sub_goals", "n_sub_goals": created},
            )
            return created > 0

        if active:
            self.workspace.write_trace_event(
                "reasoner_action_emitted",
                {"round": round_number, "action": "wait", "n_sub_goals": len(active)},
            )
            return False

        findings = self.mutator.findings()
        scored_options = _score_positive_findings(findings, self.workspace, options=options)
        if scored_options:
            if untested_options and len(scored_options) > 1 and _needs_conflict_follow_up(scored_options, sub_goals):
                created = 0
                for option_id, _citations in scored_options[: MAX_OPEN_SUB_GOALS - len(active)]:
                    self._create_option_sub_goal(
                        option_id=option_id,
                        option_text=options.get(option_id, ""),
                        question=question,
                        round_number=round_number,
                        intent="disambiguate",
                    )
                    created += 1
                self.workspace.write_trace_event(
                    "reasoner_action_emitted",
                    {
                        "round": round_number,
                        "action": "emit_sub_goals",
                        "n_sub_goals": created,
                        "reason": "positive_evidence_conflict",
                    },
                )
                return created > 0
            self.answer_result = _answer_or_need_more(
                choice=scored_options[0][0],
                citations=scored_options[0][1],
                workspace=self.workspace,
                confidence="medium",
                rounds=round_number,
                metadata={"status": "final", "strategy": "multi_agent_v0"},
            )
            self.workspace.write_trace_event(
                "reasoner_action_emitted",
                {"round": round_number, "action": "answer", "n_sub_goals": 0},
            )
            return True

        elimination = _elimination_answer(self.workspace, options=options)
        if elimination is not None:
            choice, citations = elimination
            self.answer_result = _answer_or_need_more(
                choice=choice,
                citations=citations,
                workspace=self.workspace,
                confidence="medium",
                rounds=round_number,
                metadata={"status": "final", "strategy": "multi_agent_v0", "reason": "elimination"},
            )
            self.workspace.write_trace_event(
                "reasoner_action_emitted",
                {"round": round_number, "action": "answer", "n_sub_goals": 0, "reason": "elimination"},
            )
            return True

        if option_ids:
            self.workspace.write_trace_event(
                "reasoner_action_emitted",
                {"round": round_number, "action": "wait", "n_sub_goals": 0, "reason": "no_positive_evidence"},
            )
            return False

        option_id = ""
        option_text = options.get(option_id or "", "") if option_id else ""
        claim = option_text or "Find answer-relevant local video evidence."
        self._create_option_sub_goal(
            option_id=option_id,
            option_text=claim,
            question=question,
            round_number=round_number,
        )
        self.workspace.write_trace_event(
            "reasoner_action_emitted",
            {"round": round_number, "action": "emit_sub_goals", "n_sub_goals": 1},
        )
        return True

    def _create_option_sub_goal(
        self,
        *,
        option_id: str,
        option_text: str,
        question: str,
        round_number: int,
        intent: str = "verify",
    ) -> None:
        claim = _option_claim(question=question, option_id=option_id, option_text=option_text)
        operator = derive_answer_operator(question, route="", options=())
        route = derive_modality_route(question, operator=operator)
        modalities = ROUTE_TO_MODALITIES[route]
        self.mutator.create_sub_goal(
            intent=intent,  # type: ignore[arg-type]
            constraint=SubGoalConstraint(
                option_id=option_id or None,
                claim=claim,
                modality_hint=modalities,
            ),
            budget=SubGoalBudget(max_explores=1, max_verifies=1, max_frames=64),
            success_criteria=SubGoalSuccessCriteria(needs_visual_support=True, needs_option_relation=True),
            parent_question=question,
            created_by="reasoner",
            created_round=round_number,
            rationale=(
                f"Verify option {option_id} before choosing an answer."
                if option_id
                else "Create a local verification target before answering."
            ),
        )
        self.workspace.write_trace_event(
            "modality_route_chosen",
            {
                "round": round_number,
                "option_id": option_id,
                "question_operator": operator,
                "route": route,
                "modalities": list(modalities),
            },
        )


def best_effort_answer_from_workspace(workspace: Any, options: Mapping[str, str]) -> tuple[str, tuple[str, ...]] | None:
    """Return the strongest option-bound visual answer support in workspace memory."""

    scored_options = _score_positive_memory_entries(workspace, options=options)
    return scored_options[0] if scored_options else None


def _score_positive_findings(findings: Any, workspace: Any, *, options: Mapping[str, str]) -> list[tuple[str, tuple[str, ...]]]:
    finding_memory_ids: set[str] = set()
    for finding in findings:
        if finding.status != "satisfied":
            continue
        finding_memory_ids.update(str(memory_id) for memory_id in finding.memory_ids)
    return _score_positive_memory_entries(workspace, options=options, allowed_memory_ids=finding_memory_ids or None)


def _score_positive_memory_entries(
    workspace: Any,
    *,
    options: Mapping[str, str],
    allowed_memory_ids: set[str] | None = None,
) -> list[tuple[str, tuple[str, ...]]]:
    option_ids = set(_option_ids(options))
    memory_by_id = {entry.entry_id: entry for entry in workspace.memory_entries()}
    refuted_options = _refuted_options(workspace, options=options)
    scores: dict[str, int] = {}
    citations: dict[str, list[str]] = {}
    support_counts: dict[str, int] = {}
    specificity: dict[str, int] = {}
    overlap: dict[str, int] = {}
    for entry in memory_by_id.values():
        if allowed_memory_ids is not None and entry.entry_id not in allowed_memory_ids:
            continue
        if entry.kind not in ANSWER_GROUNDING_KINDS:
            continue
        option_id = str(entry.supports_option or "").strip().upper()[:1]
        if not option_id or (option_ids and option_id not in option_ids):
            continue
        if option_id in refuted_options:
            continue
        scores[option_id] = scores.get(option_id, 0) + _confidence_score(entry.confidence)
        support_counts[option_id] = support_counts.get(option_id, 0) + 1
        specificity[option_id] = specificity.get(option_id, 0) + _anchor_specificity(entry)
        overlap[option_id] = overlap.get(option_id, 0) + _claim_option_overlap(entry, options.get(option_id, ""))
        citations.setdefault(option_id, []).append(entry.entry_id)
    option_order = {option_id: index for index, option_id in enumerate(_option_ids(options))}
    ranked = sorted(
        scores,
        key=lambda option_id: (
            -scores[option_id],
            -support_counts.get(option_id, 0),
            -specificity.get(option_id, 0),
            -overlap.get(option_id, 0),
            option_order.get(option_id, 999),
            option_id,
        ),
    )
    return [(option_id, tuple(citations[option_id])) for option_id in ranked]


def _needs_conflict_follow_up(scored_options: Sequence[tuple[str, tuple[str, ...]]], sub_goals: Sequence[Any]) -> bool:
    scored_ids = [option_id for option_id, _citations in scored_options]
    for option_id in scored_ids:
        count = sum(
            1
            for goal in sub_goals
            if goal.intent == "disambiguate" and str(goal.constraint.option_id or "").strip().upper()[:1] == option_id
        )
        if count >= 1:
            return False
    return True


def _answer_or_need_more(
    *,
    choice: str,
    citations: tuple[str, ...],
    workspace: Any,
    confidence: str,
    rounds: int,
    metadata: Mapping[str, Any],
) -> WorkspaceRunResult:
    if _valid_final_citations(workspace, citations):
        return WorkspaceRunResult(
            answer=choice,
            citations=citations,
            confidence=confidence,
            rounds=rounds,
            metadata=metadata,
        )
    return WorkspaceRunResult(
        answer="need_more_evidence",
        citations=(),
        confidence="low",
        rounds=rounds,
        metadata={**dict(metadata), "reason": "missing_visual_citation"},
    )


def _valid_final_citations(workspace: Any, citations: tuple[str, ...]) -> bool:
    if not citations:
        return False
    memory_by_id = {entry.entry_id: entry for entry in workspace.memory_entries()}
    cited_entries = [memory_by_id.get(str(citation)) for citation in citations]
    cited_entries = [entry for entry in cited_entries if entry is not None]
    if not cited_entries:
        return False
    return any(entry.kind in FINAL_GROUNDING_KINDS for entry in cited_entries)


def _anchor_specificity(entry: Any) -> int:
    text_parts = [str(getattr(entry, "claim", "") or "")]
    metadata = getattr(entry, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        text_parts.extend(str(value) for value in metadata.values() if isinstance(value, (str, int, float)))
        if metadata.get("derived_from_verify_cross_match"):
            text_parts.append("cross_match")
    for anchor in getattr(entry, "anchors", ()) or ():
        text_parts.append(str(getattr(anchor, "excerpt", "") or ""))
    text = " ".join(text_parts)
    score = 0
    score += len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text))
    score += len(re.findall(r"\b\d{3,4}\b", text)) * 2
    if str(getattr(entry, "kind", "") or "") == "visual_support":
        score += 2
    if isinstance(metadata, Mapping) and metadata.get("derived_from_verify_cross_match"):
        score += 2
    generic_terms = {"war", "map", "people", "scene", "video", "object", "thing"}
    tokens = {token.lower() for token in re.findall(r"[A-Za-z]{3,}", text)}
    if tokens and tokens <= generic_terms:
        score = max(0, score - 2)
    return score


def _claim_option_overlap(entry: Any, option_text: str) -> int:
    option_tokens = _content_tokens(option_text)
    if not option_tokens:
        return 0
    entry_text = str(getattr(entry, "claim", "") or "")
    for anchor in getattr(entry, "anchors", ()) or ():
        entry_text += " " + str(getattr(anchor, "excerpt", "") or "")
    entry_tokens = _content_tokens(entry_text)
    return len(option_tokens & entry_tokens)


def _content_tokens(text: str) -> set[str]:
    stopwords = {"the", "and", "are", "was", "were", "with", "from", "that", "this", "option", "question"}
    return {token for token in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(token) >= 3 and token not in stopwords}


def _refuted_options(workspace: Any, *, options: Mapping[str, str]) -> set[str]:
    option_ids = set(_option_ids(options))
    refuted: set[str] = set()
    for entry in workspace.memory_entries():
        option_id = str(entry.supports_option or "").strip().upper()[:1]
        if option_id not in option_ids or entry.kind not in CONTRADICTING_KINDS:
            continue
        verdict = ""
        metadata = getattr(entry, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            verdict = str(metadata.get("verdict") or "").strip().lower()
        if verdict and verdict != "contradicted":
            continue
        refuted.add(option_id)
    return refuted


def _elimination_answer(workspace: Any, *, options: Mapping[str, str]) -> tuple[str, tuple[str, ...]] | None:
    option_ids = _option_ids(options)
    if len(option_ids) < 2:
        return None
    eliminated: dict[str, list[str]] = {}
    for entry in workspace.memory_entries():
        option_id = str(entry.supports_option or "").strip().upper()[:1]
        if option_id not in option_ids or entry.kind not in CONTRADICTING_KINDS:
            continue
        verdict = ""
        metadata = getattr(entry, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            verdict = str(metadata.get("verdict") or "").strip().lower()
        if verdict and verdict != "contradicted":
            continue
        eliminated.setdefault(option_id, []).append(entry.entry_id)
    remaining = [option_id for option_id in option_ids if option_id not in eliminated]
    if len(remaining) != 1:
        return None
    if sum(1 for option_id in option_ids if option_id in eliminated) != len(option_ids) - 1:
        return None
    citations: list[str] = []
    for option_id in option_ids:
        citations.extend(eliminated.get(option_id, ()))
    return remaining[0], tuple(citations)


def _option_claim(*, question: str, option_id: str, option_text: str) -> str:
    question_text = _question_stem(question)
    option_body = " ".join(str(option_text or "").split())
    if option_id and option_body:
        return f"{question_text} Option {option_id}: {option_body}."
    return option_body or question_text or "Find answer-relevant local video evidence."


def _question_stem(question: str) -> str:
    text = " ".join(str(question or "").split())
    for marker in (" Options:", " options:", "\nOptions:", "\noptions:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
            break
    return text


def _confidence_score(confidence: Any) -> int:
    normalized = str(confidence or "").strip().lower()
    if normalized == "high":
        return 3
    if normalized == "low":
        return 1
    return 2


def _option_ids(options: Mapping[str, str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for key in sorted(str(item).strip().upper() for item in options if str(item).strip()):
        option_id = key[:1]
        if option_id and option_id not in seen:
            seen.add(option_id)
            ordered.append(option_id)
    return ordered
