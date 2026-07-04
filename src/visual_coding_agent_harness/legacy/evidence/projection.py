"""Project generic evidence support back to answer options."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from ..task.spec import OptionSpec, TaskSpec

ProjectionStatus = Literal["supported", "partial", "ambiguous", "unsupported"]


@dataclass(frozen=True)
class ProjectionEvidence:
    evidence_id: str
    target_ref: str
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    order_index: int | None = None
    confidence: float | None = None
    segment_id: str = ""
    support_status: str = "supported"
    option_label: str | None = None
    modality: str = ""
    source: str = ""


@dataclass(frozen=True)
class ProjectionResult:
    status: ProjectionStatus
    option_label: str | None
    strategy: str
    score: float
    supporting_evidence_ids: Sequence[str] = field(default_factory=tuple)
    candidate_option_labels: Sequence[str] = field(default_factory=tuple)
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "supporting_evidence_ids", tuple(self.supporting_evidence_ids))
        object.__setattr__(self, "candidate_option_labels", tuple(self.candidate_option_labels))


@dataclass(frozen=True)
class _OptionScore:
    option: OptionSpec
    strategy: str
    score: float
    supporting_evidence_ids: tuple[str, ...]
    complete: bool


def project_option_support(task: TaskSpec, evidence: Sequence[ProjectionEvidence]) -> ProjectionResult:
    """Return the best option supported by generic evidence projection."""

    normalized_task = task.normalized()
    supported_evidence = tuple(item for item in evidence if str(item.support_status or "supported") == "supported")
    if not normalized_task.options:
        return ProjectionResult("unsupported", None, "none", 0.0, reason="no_supported_evidence")
    operator = str(getattr(normalized_task, "answer_operator", "select_present") or "select_present")
    if operator == "select_absent":
        return _project_select_absent(normalized_task, supported_evidence)
    if not supported_evidence:
        return ProjectionResult("unsupported", None, "none", 0.0, reason="no_supported_evidence")
    if operator == "causal_bind":
        return _project_causal_bind(normalized_task, supported_evidence)
    if operator == "universal_intersection":
        return _project_universal_intersection(normalized_task, supported_evidence)
    if operator == "ordered_projection":
        return _project_ordered_projection(normalized_task, supported_evidence)
    if operator == "main_arc":
        return _project_main_arc(normalized_task, supported_evidence)

    strategies = _strategies_for_task(normalized_task)
    for strategy in strategies:
        scores = tuple(
            score
            for option in normalized_task.options
            if (score := _score_option(option, supported_evidence, strategy)).score > 0.0
        )
        result = _select_score(scores, strategy=strategy)
        if result.status in {"supported", "ambiguous"}:
            return result

    return ProjectionResult("unsupported", None, "none", 0.0, reason="no_option_projection")


def _project_select_absent(task: TaskSpec, evidence: Sequence[ProjectionEvidence]) -> ProjectionResult:
    evidence_by_target = _evidence_by_target(evidence)
    unsupported: list[OptionSpec] = []
    for option in task.options:
        refs = _option_refs(option)
        if not refs or not any(ref in evidence_by_target for ref in refs):
            unsupported.append(option)
    if len(unsupported) == 1:
        candidate = unsupported[0]
        competitor_missing = [
            option.label
            for option in task.options
            if option.label != candidate.label and not any(ref in evidence_by_target for ref in _option_refs(option))
        ]
        if competitor_missing:
            return ProjectionResult(
                "partial",
                candidate.label,
                "select_absent",
                0.0,
                candidate_option_labels=(candidate.label,),
                reason="complement_incomplete_presence",
            )
        competitor_ids = _unique_ids(
            [
                evidence_id
                for option in task.options
                if option.label != candidate.label
                for evidence_id in _supporting_ids_for_targets(_option_refs(option), evidence_by_target)[:1]
            ]
        )
        return ProjectionResult(
            "supported",
            candidate.label,
            "select_absent",
            1.0,
            supporting_evidence_ids=competitor_ids,
            candidate_option_labels=(candidate.label,),
            reason="complement_resolved",
        )
    if len(unsupported) > 1:
        return ProjectionResult(
            "ambiguous",
            None,
            "select_absent",
            0.0,
            candidate_option_labels=tuple(option.label for option in unsupported),
            reason="multiple_absent_candidates",
        )
    return ProjectionResult("unsupported", None, "select_absent", 0.0, reason="candidate_has_positive_support")


def _project_causal_bind(task: TaskSpec, evidence: Sequence[ProjectionEvidence]) -> ProjectionResult:
    binding_evidence = tuple(item for item in evidence if _is_causal_binding_evidence(item))
    if not binding_evidence:
        reason = "topic_overlap_only" if evidence else "causal_binding_missing"
        return ProjectionResult("unsupported", None, "causal_bind", 0.0, reason=reason)
    evidence_by_target = _evidence_by_target(binding_evidence)
    complete_scores = tuple(
        score
        for option in task.options
        if (score := _score_required_target_set(option, binding_evidence)).complete
    )
    if len(complete_scores) > 1:
        return ProjectionResult(
            "ambiguous",
            None,
            "causal_bind",
            max(score.score for score in complete_scores),
            candidate_option_labels=tuple(score.option.label for score in complete_scores),
            reason="multiple_causes_supported",
        )
    if len(complete_scores) == 1:
        winner = complete_scores[0]
        return ProjectionResult(
            "supported",
            winner.option.label,
            "causal_bind",
            winner.score,
            supporting_evidence_ids=_supporting_ids_for_targets(_option_refs(winner.option), evidence_by_target),
            candidate_option_labels=(winner.option.label,),
            reason="causal_binding_supported",
        )
    return ProjectionResult("unsupported", None, "causal_bind", 0.0, reason="causal_binding_missing")


def _project_universal_intersection(task: TaskSpec, evidence: Sequence[ProjectionEvidence]) -> ProjectionResult:
    groups = tuple(sorted({_evidence_group(item) for item in evidence if _evidence_group(item)}))
    if len(groups) <= 1:
        return ProjectionResult("unsupported", None, "universal_intersection", 0.0, reason="single_group_only")
    complete: list[tuple[OptionSpec, tuple[str, ...]]] = []
    for option in task.options:
        refs = set(_option_refs(option))
        if not refs:
            continue
        supporting = tuple(item for item in evidence if item.target_ref in refs)
        supported_groups = {_evidence_group(item) for item in supporting if _evidence_group(item)}
        if set(groups).issubset(supported_groups):
            complete.append((option, _unique_ids([item.evidence_id for item in supporting])))
    if len(complete) == 1:
        option, supporting_ids = complete[0]
        return ProjectionResult(
            "supported",
            option.label,
            "universal_intersection",
            1.0,
            supporting_evidence_ids=supporting_ids,
            candidate_option_labels=(option.label,),
            reason="universal_complete",
        )
    if len(complete) > 1:
        return ProjectionResult(
            "ambiguous",
            None,
            "universal_intersection",
            1.0,
            candidate_option_labels=tuple(option.label for option, _ids in complete),
            reason="multiple_options_have_complete_support",
        )
    return ProjectionResult("partial", None, "universal_intersection", 0.0, reason="universal_coverage_incomplete")


def _project_ordered_projection(task: TaskSpec, evidence: Sequence[ProjectionEvidence]) -> ProjectionResult:
    sequences = tuple((option, tuple(option.target_sequence)) for option in task.options if option.target_sequence)
    if not sequences:
        return ProjectionResult("unsupported", None, "ordered_projection", 0.0, reason="ordered_item_missing")
    required_targets = _unique_ids([target for _option, sequence in sequences for target in sequence])
    positions = _earliest_positions(evidence)
    missing = tuple(target for target in required_targets if target not in positions)
    if missing:
        return ProjectionResult("partial", None, "ordered_projection", 0.0, reason="ordered_item_missing")
    observed_order = tuple(sorted(required_targets, key=lambda target: positions[target][0]))
    matches = tuple(option for option, sequence in sequences if sequence == observed_order)
    if len(matches) == 1:
        option = matches[0]
        return ProjectionResult(
            "supported",
            option.label,
            "ordered_projection",
            1.0,
            supporting_evidence_ids=tuple(positions[target][1] for target in observed_order),
            candidate_option_labels=(option.label,),
            reason="ordered_projection_supported",
        )
    if len(matches) > 1:
        return ProjectionResult(
            "ambiguous",
            None,
            "ordered_projection",
            1.0,
            candidate_option_labels=tuple(option.label for option in matches),
            reason="multiple_order_options_match",
        )
    return ProjectionResult("unsupported", None, "ordered_projection", 0.0, reason="order_position_ambiguous")


def _project_main_arc(task: TaskSpec, evidence: Sequence[ProjectionEvidence]) -> ProjectionResult:
    scores = tuple(
        score
        for option in task.options
        if (score := _score_theme_coverage(option, evidence)).score > 0.0
    )
    result = _select_score(scores, strategy="theme_coverage")
    if result.status == "supported":
        return ProjectionResult(
            result.status,
            result.option_label,
            result.strategy,
            result.score,
            supporting_evidence_ids=result.supporting_evidence_ids,
            candidate_option_labels=result.candidate_option_labels,
            reason="main_arc_dominant",
        )
    if result.status == "partial":
        return ProjectionResult(
            result.status,
            result.option_label,
            result.strategy,
            result.score,
            supporting_evidence_ids=result.supporting_evidence_ids,
            candidate_option_labels=result.candidate_option_labels,
            reason="insufficient_breadth",
        )
    return result


def _strategies_for_task(task: TaskSpec) -> tuple[str, ...]:
    route = str(task.route or "").casefold()
    has_sequence = any(option.target_sequence for option in task.options)
    has_theme = any(option.theme_targets for option in task.options)
    if has_sequence and any(marker in route for marker in ("order", "sequence", "timeline", "temporal")):
        return ("ordered_sequence", "required_target_set", "theme_coverage", "unique_target")
    if has_theme or any(marker in route for marker in ("main_idea", "gist", "synopsis", "summary")):
        return ("theme_coverage", "required_target_set", "ordered_sequence", "unique_target")
    return ("required_target_set", "ordered_sequence", "theme_coverage", "unique_target")


def _score_option(option: OptionSpec, evidence: Sequence[ProjectionEvidence], strategy: str) -> _OptionScore:
    if strategy == "ordered_sequence":
        return _score_ordered_sequence(option, evidence)
    if strategy == "theme_coverage":
        return _score_theme_coverage(option, evidence)
    if strategy == "unique_target":
        return _score_unique_target(option, evidence)
    return _score_required_target_set(option, evidence)


def _score_required_target_set(option: OptionSpec, evidence: Sequence[ProjectionEvidence]) -> _OptionScore:
    required = tuple(option.required_targets)
    if not required:
        return _empty_score(option, "required_target_set")
    evidence_by_target = _evidence_by_target(evidence)
    matched_ids = _supporting_ids_for_targets(required, evidence_by_target)
    matched_count = len({target for target in required if target in evidence_by_target})
    score = matched_count / max(len(required), 1)
    return _OptionScore(
        option=option,
        strategy="required_target_set",
        score=score,
        supporting_evidence_ids=matched_ids,
        complete=matched_count == len(required),
    )


def _score_ordered_sequence(option: OptionSpec, evidence: Sequence[ProjectionEvidence]) -> _OptionScore:
    sequence = tuple(option.target_sequence)
    if not sequence:
        return _empty_score(option, "ordered_sequence")
    ordered_evidence = sorted(
        (item for item in evidence if item.timestamp_start is not None),
        key=lambda item: (float(item.timestamp_start or 0.0), item.evidence_id),
    )
    cursor = 0
    supporting_ids: list[str] = []
    for target_ref in sequence:
        found = None
        for index in range(cursor, len(ordered_evidence)):
            if ordered_evidence[index].target_ref == target_ref:
                found = index
                break
        if found is None:
            break
        supporting_ids.append(ordered_evidence[found].evidence_id)
        cursor = found + 1
    score = len(supporting_ids) / max(len(sequence), 1)
    return _OptionScore(
        option=option,
        strategy="ordered_sequence",
        score=score,
        supporting_evidence_ids=tuple(supporting_ids),
        complete=len(supporting_ids) == len(sequence),
    )


def _score_theme_coverage(option: OptionSpec, evidence: Sequence[ProjectionEvidence]) -> _OptionScore:
    theme_targets = tuple(option.theme_targets or option.required_targets)
    if not theme_targets:
        return _empty_score(option, "theme_coverage")
    evidence_by_target = _evidence_by_target(evidence)
    supporting_ids = _supporting_ids_for_targets(theme_targets, evidence_by_target)
    matched_targets = tuple(target for target in theme_targets if target in evidence_by_target)
    coverage = len(matched_targets) / max(len(theme_targets), 1)
    distinct_segments = len(
        {
            item.segment_id
            for target in matched_targets
            for item in evidence_by_target.get(target, ())
            if item.segment_id
        }
    )
    breadth_bonus = 0.1 * min(distinct_segments, len(theme_targets))
    target_breadth_bonus = 0.01 * len(theme_targets)
    score = coverage + breadth_bonus + target_breadth_bonus
    return _OptionScore(
        option=option,
        strategy="theme_coverage",
        score=score,
        supporting_evidence_ids=supporting_ids,
        complete=len(matched_targets) == len(theme_targets) and len(theme_targets) >= 2,
    )


def _score_unique_target(option: OptionSpec, evidence: Sequence[ProjectionEvidence]) -> _OptionScore:
    refs = tuple(option.required_targets or option.target_sequence or option.theme_targets)
    if len(refs) != 1:
        return _empty_score(option, "unique_target")
    evidence_by_target = _evidence_by_target(evidence)
    supporting_ids = _supporting_ids_for_targets(refs, evidence_by_target)
    return _OptionScore(
        option=option,
        strategy="unique_target",
        score=1.0 if supporting_ids else 0.0,
        supporting_evidence_ids=supporting_ids,
        complete=bool(supporting_ids),
    )


def _select_score(scores: Sequence[_OptionScore], *, strategy: str) -> ProjectionResult:
    complete_scores = tuple(score for score in scores if score.complete)
    if len(complete_scores) > 1:
        best_score = max(score.score for score in complete_scores)
        return ProjectionResult(
            "ambiguous",
            None,
            strategy,
            best_score,
            candidate_option_labels=tuple(score.option.label for score in complete_scores),
            reason="multiple_options_have_complete_support",
        )
    candidate_scores = complete_scores or tuple(score for score in scores if score.score > 0.0)
    if not candidate_scores:
        return ProjectionResult("unsupported", None, strategy, 0.0)
    best_score = max(score.score for score in candidate_scores)
    best = tuple(score for score in candidate_scores if abs(score.score - best_score) < 1e-9)
    if len(best) > 1:
        return ProjectionResult(
            "ambiguous",
            None,
            strategy,
            best_score,
            candidate_option_labels=tuple(score.option.label for score in best),
            reason="multiple_options_have_equal_support",
        )
    winner = best[0]
    return ProjectionResult(
        "supported" if winner.complete else "partial",
        winner.option.label,
        winner.strategy,
        winner.score,
        supporting_evidence_ids=winner.supporting_evidence_ids,
    )


def _evidence_by_target(evidence: Sequence[ProjectionEvidence]) -> dict[str, tuple[ProjectionEvidence, ...]]:
    grouped: dict[str, list[ProjectionEvidence]] = {}
    for item in evidence:
        grouped.setdefault(str(item.target_ref), []).append(item)
    return {target: tuple(items) for target, items in grouped.items()}


def _supporting_ids_for_targets(
    targets: Sequence[str],
    evidence_by_target: dict[str, tuple[ProjectionEvidence, ...]],
) -> tuple[str, ...]:
    ids: list[str] = []
    seen: set[str] = set()
    for target in targets:
        for item in evidence_by_target.get(target, ()):
            if item.evidence_id in seen:
                continue
            ids.append(item.evidence_id)
            seen.add(item.evidence_id)
            break
    return tuple(ids)


def _empty_score(option: OptionSpec, strategy: str) -> _OptionScore:
    return _OptionScore(option=option, strategy=strategy, score=0.0, supporting_evidence_ids=(), complete=False)


def _option_refs(option: OptionSpec) -> tuple[str, ...]:
    return tuple(option.required_targets or option.target_sequence or option.theme_targets)


def _is_causal_binding_evidence(item: ProjectionEvidence) -> bool:
    source = str(item.source or "").casefold()
    modality = str(item.modality or "").casefold()
    return (
        "bind_asr_claim" in source
        or "claim_binding" in source
        or "causal" in source
        or "explicit_cause" in source
        or modality in {"indexed_transcript", "claim_binding"}
    )


def _evidence_group(item: ProjectionEvidence) -> str:
    text = str(item.segment_id or "").strip()
    if text:
        return text
    if item.timestamp_start is not None:
        return f"t{int(float(item.timestamp_start) // 30)}"
    return ""


def _earliest_positions(evidence: Sequence[ProjectionEvidence]) -> dict[str, tuple[tuple[float, int, str], str]]:
    positions: dict[str, tuple[tuple[float, int, str], str]] = {}
    for index, item in enumerate(evidence):
        if item.order_index is not None:
            key = (float(item.order_index), index, item.evidence_id)
        elif item.timestamp_start is not None:
            key = (float(item.timestamp_start), index, item.evidence_id)
        else:
            key = (float(index), index, item.evidence_id)
        current = positions.get(item.target_ref)
        if current is None or key < current[0]:
            positions[item.target_ref] = (key, item.evidence_id)
    return positions


def _unique_ids(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)
