"""Pure final-answer gate for grounded option evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .contracts import FinalGateDecision, FinalRejectionReason, OptionEvaluation
from .skills.policy_constants import (
    MAIN_IDEA_RECOVERY_TOP_K,
    SkillPolicy,
    SkillPolicyName,
    TRANSCRIPT_MODALITIES,
    VISUAL_MODALITIES,
    get_skill_policy,
)


@dataclass(frozen=True)
class _NormalizedEvidence:
    evidence_id: str
    target_ref: str | None
    relation_ref: str | None
    option_id: str | None
    modality: str
    source: str
    grounding_quality: str
    timestamp_start: float | None
    timestamp_end: float | None
    support_status: str


@dataclass(frozen=True)
class _NormalizedRelation:
    relation_ref: str
    ordered_target_refs: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    support_status: str
    timestamp_order: tuple[float, ...]
    modality: str | None


def evaluate_final_candidate(
    selected_option: str | Any,
    registry: Any,
    evidence_bindings: Sequence[Any],
    relation_bindings: Sequence[Any],
    policy: SkillPolicy | SkillPolicyName | str | None = None,
    *,
    skill_name: SkillPolicyName | str | None = None,
    option_evaluations: Sequence[OptionEvaluation | Any] | None = None,
    central_subjects: Sequence[str] = (),
) -> FinalGateDecision:
    """Evaluate whether a proposed option has answer-grade grounded support."""

    skill_policy = _resolve_policy(policy, skill_name)
    option = _resolve_option(selected_option, registry)
    option_id = _field(option, "option_id", default=str(selected_option))
    target_refs = _option_target_refs(option)
    required_relation_refs = _option_relation_refs(option)
    evidence = tuple(_normalize_evidence(binding) for binding in evidence_bindings)
    relations = tuple(_normalize_relation(binding, evidence) for binding in relation_bindings)
    option_eval_by_id = _option_evaluation_by_id(option_evaluations or ())
    expected_option_ids = _registry_option_ids(registry)

    selected_evidence = tuple(binding for binding in evidence if _applies_to_option(binding, option_id))
    conflicting_ids = tuple(
        binding.evidence_id
        for binding in selected_evidence
        if binding.support_status == "conflicting" and binding.target_ref in target_refs
    )
    if conflicting_ids:
        return _reject(option_id, "conflicting_evidence", supporting_evidence_ids=conflicting_ids)

    if skill_policy.skill_name == "main_idea":
        return _evaluate_main_idea(
            option_id=option_id,
            option=option,
            registry=registry,
            policy=skill_policy,
            target_refs=target_refs,
            evidence=selected_evidence,
            option_evaluation=option_eval_by_id.get(option_id),
            option_eval_by_id=option_eval_by_id,
            expected_option_ids=expected_option_ids,
            central_subjects=central_subjects,
        )

    if skill_policy.skill_name == "mixed_asr_visual_qa":
        return _evaluate_mixed_asr_visual(
            option_id=option_id,
            policy=skill_policy,
            target_refs=target_refs,
            evidence=selected_evidence,
        )

    if skill_policy.skill_name == "mutex_fact_qa":
        return _evaluate_mutex_fact(
            option_id=option_id,
            policy=skill_policy,
            target_refs=target_refs,
            evidence=evidence,
            selected_evidence=selected_evidence,
        )

    missing_target_refs, unsupported_modality_refs, supporting_evidence_ids = _target_support(
        target_refs=target_refs,
        evidence=selected_evidence,
        policy=skill_policy,
        require_timestamp=skill_policy.visual_verification_mandatory,
    )
    if missing_target_refs:
        if unsupported_modality_refs:
            return _reject(
                option_id,
                "unsupported_modality",
                supporting_evidence_ids=supporting_evidence_ids,
                missing_target_refs=missing_target_refs,
            )
        return _reject(
            option_id,
            "missing_target_binding",
            supporting_evidence_ids=supporting_evidence_ids,
            missing_target_refs=missing_target_refs,
        )

    relation_required = bool(required_relation_refs) and (
        skill_policy.relation_required or _option_kind(option) == "sequence"
    )
    if relation_required:
        missing_relation_refs = _missing_relation_refs(
            required_relation_refs=required_relation_refs,
            option_target_refs=target_refs,
            registry=registry,
            relations=relations,
            policy=skill_policy,
        )
        if missing_relation_refs:
            return _reject(
                option_id,
                "missing_relation_binding",
                supporting_evidence_ids=supporting_evidence_ids,
                missing_relation_refs=missing_relation_refs,
            )

    return _accept(option_id, supporting_evidence_ids=supporting_evidence_ids)


def _evaluate_main_idea(
    *,
    option_id: str,
    option: Any,
    registry: Any,
    policy: SkillPolicy,
    target_refs: tuple[str, ...],
    evidence: Sequence[_NormalizedEvidence],
    option_evaluation: OptionEvaluation | Any | None,
    option_eval_by_id: dict[str, OptionEvaluation | Any],
    expected_option_ids: tuple[str, ...],
    central_subjects: Sequence[str],
) -> FinalGateDecision:
    if policy.requires_per_option_coverage:
        missing_option_evaluations = bool(expected_option_ids) and not set(expected_option_ids).issubset(option_eval_by_id)
        single_option_chase = len(option_eval_by_id) <= 1 and len(expected_option_ids) > 1
        if missing_option_evaluations or single_option_chase or option_evaluation is None:
            return _reject(
                option_id,
                "no_per_option_coverage",
                actionable_next_program=_per_option_coverage_program(registry),
            )

    option_kind = _option_kind(option)
    if option_kind is not None and option_kind not in policy.requires_option_kind:
        return _reject(option_id, "verifier_failed")

    if central_subjects and not _subjects_overlap(registry, target_refs, central_subjects):
        return _reject(option_id, "wrong_subject")

    answer_grade_evidence = tuple(binding for binding in evidence if _is_answer_grade_main_idea_evidence(binding))
    missing_target_refs, unsupported_modality_refs, supporting_evidence_ids = _target_support(
        target_refs=target_refs,
        evidence=answer_grade_evidence,
        policy=policy,
        require_timestamp=False,
    )
    supported = tuple(
        binding
        for binding in answer_grade_evidence
        if (
            binding.target_ref in target_refs
            and binding.support_status == "supported"
            and _modality_allowed(binding.modality, policy)
        )
    )
    coverage_breadth = _int_field(option_evaluation, "coverage_breadth", default=len({b.target_ref for b in supported}))
    distinct_segment_count = len({_segment_key(binding) for binding in supported})

    if (
        len(supported) < policy.min_bindings
        or coverage_breadth < policy.min_bindings
        or distinct_segment_count < policy.min_distinct_segments
    ):
        return _reject(
            option_id,
            "insufficient_breadth",
            supporting_evidence_ids=supporting_evidence_ids,
            missing_target_refs=missing_target_refs,
        )

    if missing_target_refs:
        if unsupported_modality_refs:
            return _reject(
                option_id,
                "unsupported_modality",
                supporting_evidence_ids=supporting_evidence_ids,
                missing_target_refs=missing_target_refs,
            )
        return _reject(
            option_id,
            "missing_target_binding",
            supporting_evidence_ids=supporting_evidence_ids,
            missing_target_refs=missing_target_refs,
        )

    rejection_reason = _field(option_evaluation, "rejection_reason")
    if rejection_reason == "wrong_subject":
        return _reject(option_id, "wrong_subject", supporting_evidence_ids=supporting_evidence_ids)
    if rejection_reason == "off_topic":
        return _reject(option_id, "wrong_subject", supporting_evidence_ids=supporting_evidence_ids)
    if rejection_reason in {"insufficient_breadth", "narrower", "wrong_arc"}:
        return _reject(
            option_id,
            "insufficient_breadth",
            supporting_evidence_ids=supporting_evidence_ids,
            missing_target_refs=missing_target_refs,
        )
    if rejection_reason is not None:
        return _reject(option_id, "verifier_failed", supporting_evidence_ids=supporting_evidence_ids)

    return _accept(option_id, supporting_evidence_ids=supporting_evidence_ids)


def _evaluate_mixed_asr_visual(
    *,
    option_id: str,
    policy: SkillPolicy,
    target_refs: tuple[str, ...],
    evidence: Sequence[_NormalizedEvidence],
) -> FinalGateDecision:
    asr = tuple(
        binding
        for binding in evidence
        if binding.target_ref in target_refs
        and binding.support_status == "supported"
        and binding.modality in TRANSCRIPT_MODALITIES
    )
    visual = tuple(
        binding
        for binding in evidence
        if binding.target_ref in target_refs
        and binding.support_status == "supported"
        and binding.modality in VISUAL_MODALITIES
    )
    supporting_ids = _unique_ids([*(binding.evidence_id for binding in asr), *(binding.evidence_id for binding in visual)])
    for asr_binding in asr:
        for visual_binding in visual:
            if asr_binding.target_ref != visual_binding.target_ref:
                continue
            if _timestamps_overlap(asr_binding, visual_binding, policy.overlap_window_seconds or 0.0):
                return _accept(option_id, supporting_evidence_ids=supporting_ids)

    missing_target_refs = tuple(target for target in target_refs if target not in {b.target_ref for b in asr} or target not in {b.target_ref for b in visual})
    return _reject(
        option_id,
        "missing_target_binding",
        supporting_evidence_ids=supporting_ids,
        missing_target_refs=missing_target_refs,
    )


def _evaluate_mutex_fact(
    *,
    option_id: str,
    policy: SkillPolicy,
    target_refs: tuple[str, ...],
    evidence: Sequence[_NormalizedEvidence],
    selected_evidence: Sequence[_NormalizedEvidence],
) -> FinalGateDecision:
    missing_target_refs, unsupported_modality_refs, supporting_evidence_ids = _target_support(
        target_refs=target_refs,
        evidence=selected_evidence,
        policy=policy,
        require_timestamp=False,
    )
    if missing_target_refs:
        if unsupported_modality_refs:
            return _reject(
                option_id,
                "unsupported_modality",
                supporting_evidence_ids=supporting_evidence_ids,
                missing_target_refs=missing_target_refs,
            )
        return _reject(
            option_id,
            "missing_target_binding",
            supporting_evidence_ids=supporting_evidence_ids,
            missing_target_refs=missing_target_refs,
        )

    selected_window_keys = {_segment_key(binding) for binding in selected_evidence if binding.evidence_id in supporting_evidence_ids}
    conflicting_ids = tuple(
        binding.evidence_id
        for binding in evidence
        if (
            binding.option_id not in (None, "", option_id)
            and binding.support_status == "supported"
            and _segment_key(binding) in selected_window_keys
        )
    )
    if conflicting_ids:
        return _reject(option_id, "conflicting_evidence", supporting_evidence_ids=supporting_evidence_ids)
    return _accept(option_id, supporting_evidence_ids=supporting_evidence_ids)


def _target_support(
    *,
    target_refs: tuple[str, ...],
    evidence: Sequence[_NormalizedEvidence],
    policy: SkillPolicy,
    require_timestamp: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    missing_target_refs: list[str] = []
    unsupported_modality_refs: list[str] = []
    supporting_evidence_ids: list[str] = []

    for target_ref in target_refs:
        target_evidence = tuple(
            binding
            for binding in evidence
            if binding.target_ref == target_ref and binding.support_status == "supported"
        )
        allowed = tuple(
            binding
            for binding in target_evidence
            if _modality_allowed(binding.modality, policy) and (not require_timestamp or _has_timestamp(binding))
        )
        if allowed:
            supporting_evidence_ids.extend(binding.evidence_id for binding in allowed)
            continue
        missing_target_refs.append(target_ref)
        if target_evidence:
            unsupported_modality_refs.append(target_ref)

    return (
        tuple(missing_target_refs),
        tuple(unsupported_modality_refs),
        _unique_ids(supporting_evidence_ids),
    )


def _missing_relation_refs(
    *,
    required_relation_refs: tuple[str, ...],
    option_target_refs: tuple[str, ...],
    registry: Any,
    relations: Sequence[_NormalizedRelation],
    policy: SkillPolicy,
) -> tuple[str, ...]:
    missing: list[str] = []
    for relation_ref in required_relation_refs:
        expected_order = _relation_expected_order(registry, relation_ref)
        matching = tuple(
            relation
            for relation in relations
            if (
                relation.relation_ref == relation_ref
                and relation.support_status == "supported"
                and _relation_modality_allowed(relation, policy)
                and _relation_order_compatible(
                    relation.ordered_target_refs,
                    expected_order=expected_order,
                    option_target_refs=option_target_refs,
                )
            )
        )
        if not matching:
            missing.append(relation_ref)
    return tuple(missing)


def _relation_order_compatible(
    ordered_target_refs: tuple[str, ...],
    *,
    expected_order: tuple[str, ...],
    option_target_refs: tuple[str, ...],
) -> bool:
    if not ordered_target_refs:
        return True
    if expected_order:
        return ordered_target_refs == expected_order
    return _is_subsequence(ordered_target_refs, option_target_refs)


def _relation_expected_order(registry: Any, relation_ref: str) -> tuple[str, ...]:
    relation = _registry_relation(registry, relation_ref)
    if relation is None:
        return ()
    ordered = _tuple_field(relation, "ordered_target_refs", "ordered_targets")
    if ordered:
        return ordered
    source = _field(relation, "source_target_ref", "source_target_id")
    target = _field(relation, "target_ref", "destination_target_id", "target_id")
    if source and target:
        return (str(source), str(target))
    return ()


def _registry_relation(registry: Any, relation_ref: str) -> Any | None:
    relations_by_id = _field(registry, "relations_by_id", default={})
    if isinstance(relations_by_id, dict) or hasattr(relations_by_id, "get"):
        return relations_by_id.get(relation_ref)
    return None


def _resolve_policy(policy: SkillPolicy | str | None, skill_name: str | None) -> SkillPolicy:
    if isinstance(policy, SkillPolicy):
        return policy
    policy_name = str(policy or skill_name or "").strip()
    return get_skill_policy(policy_name)


def _resolve_option(selected_option: str | Any, registry: Any) -> Any:
    if not isinstance(selected_option, str):
        return selected_option
    if hasattr(registry, "option_for"):
        return registry.option_for(selected_option)
    options_by_id = _field(registry, "options_by_id", default={})
    if hasattr(options_by_id, "get"):
        option = options_by_id.get(selected_option)
        if option is not None:
            return option
    raise KeyError(f"Unknown option: {selected_option}")


def _option_target_refs(option: Any) -> tuple[str, ...]:
    return _tuple_field(
        option,
        "target_refs",
        "target_sequence",
        "ordered_target_refs",
        "required_target_keys",
    )


def _option_relation_refs(option: Any) -> tuple[str, ...]:
    relations = _field(option, "required_relations", "required_relation_refs", "required_relation_keys", default=())
    relation_refs: list[str] = []
    for relation in relations or ():
        relation_ref = _field(relation, "relation_ref", "relation_id", default=relation)
        if relation_ref is not None:
            relation_refs.append(str(relation_ref))
    return tuple(relation_refs)


def _option_kind(option: Any) -> str | None:
    kind = _field(option, "option_kind")
    text = str(kind or "").strip()
    return text or None


def _normalize_evidence(binding: Any) -> _NormalizedEvidence:
    timestamp = _float_or_none(_field(binding, "timestamp_start", "mention_timestamp_sec"))
    return _NormalizedEvidence(
        evidence_id=str(_field(binding, "evidence_id", "binding_id", default="")),
        target_ref=_string_or_none(_field(binding, "target_ref", "target_id")),
        relation_ref=_string_or_none(_field(binding, "relation_ref", "relation_id")),
        option_id=_string_or_none(_field(binding, "option_id")),
        modality=_normalize_modality(_field(binding, "modality", "claim_modality", "source")),
        source=str(_field(binding, "source", "tool", "obs_id", default="") or "").strip(),
        grounding_quality=_normalize_modality(_field(binding, "grounding_quality", default="")),
        timestamp_start=timestamp,
        timestamp_end=_float_or_none(_field(binding, "timestamp_end", default=timestamp)),
        support_status=_normalize_status(_field(binding, "support_status", "status")),
    )


def _normalize_relation(binding: Any, evidence: Sequence[_NormalizedEvidence]) -> _NormalizedRelation:
    evidence_ids = _tuple_field(binding, "evidence_ids")
    modality = _field(binding, "modality")
    if modality is None and evidence_ids:
        modalities = tuple(ev.modality for ev in evidence if ev.evidence_id in evidence_ids)
        if len(set(modalities)) == 1:
            modality = modalities[0]
    return _NormalizedRelation(
        relation_ref=str(_field(binding, "relation_ref", "relation_id", default="")),
        ordered_target_refs=_tuple_field(binding, "ordered_target_refs"),
        evidence_ids=evidence_ids,
        support_status=_normalize_status(_field(binding, "support_status", "status")),
        timestamp_order=tuple(
            value
            for value in (_float_or_none(item) for item in _tuple_field(binding, "timestamp_order"))
            if value is not None
        ),
        modality=_normalize_modality(modality) if modality is not None else None,
    )


def _option_evaluation_by_id(evaluations: Sequence[OptionEvaluation | Any]) -> dict[str, OptionEvaluation | Any]:
    result: dict[str, OptionEvaluation | Any] = {}
    for evaluation in evaluations:
        option_id = _field(evaluation, "option_id")
        if option_id is not None:
            result[str(option_id)] = evaluation
    return result


def _registry_option_ids(registry: Any) -> tuple[str, ...]:
    options_by_id = _field(registry, "options_by_id", default={})
    if hasattr(options_by_id, "keys"):
        return tuple(str(option_id) for option_id in options_by_id.keys())
    return ()


def _applies_to_option(binding: _NormalizedEvidence, option_id: str) -> bool:
    return binding.option_id in (None, "", option_id)


def _modality_allowed(binding_modality: str | None, policy: SkillPolicy) -> bool:
    return str(binding_modality or "").strip().lower() in policy.allowed_modalities


def _is_answer_grade_main_idea_evidence(binding: _NormalizedEvidence) -> bool:
    marker_values = {
        _normalize_text(binding.modality),
        _normalize_text(binding.source),
        _normalize_text(binding.grounding_quality),
    }
    context_only_markers = {"global_gist", "global_sparse", "query_global_context"}
    return not (marker_values & context_only_markers)


def _relation_modality_allowed(relation: _NormalizedRelation, policy: SkillPolicy) -> bool:
    if relation.modality is None:
        return True
    return _modality_allowed(relation.modality, policy)


def _subjects_overlap(registry: Any, target_refs: tuple[str, ...], central_subjects: Sequence[str]) -> bool:
    central = {_normalize_text(subject) for subject in central_subjects if _normalize_text(subject)}
    if not central:
        return True
    for target_ref in target_refs:
        target = _registry_target(registry, target_ref)
        values = {_normalize_text(value) for value in _target_subject_values(target)}
        if central & values:
            return True
    return False


def _registry_target(registry: Any, target_ref: str) -> Any | None:
    if hasattr(registry, "resolve_target_ref"):
        try:
            return registry.resolve_target_ref(target_ref)
        except KeyError:
            return None
    targets_by_id = _field(registry, "targets_by_id", default={})
    if hasattr(targets_by_id, "get"):
        return targets_by_id.get(target_ref)
    return None


def _target_subject_values(target: Any | None) -> tuple[str, ...]:
    if target is None:
        return ()
    values: list[str] = []
    for field_name in ("subject", "canonical_text", "canonical_claim"):
        value = _field(target, field_name)
        if value is not None:
            values.append(str(value))
    values.extend(_tuple_field(target, "aliases"))
    return tuple(values)


def _normalize_modality(value: Any) -> str:
    raw = getattr(value, "value", value)
    text = str(raw or "").strip().lower()
    aliases = {
        "visual_fact": "visual",
        "narrated_fact": "asr",
        "ocr_fact": "ocr",
        "indexed transcript": "indexed_transcript",
    }
    return aliases.get(text, text)


def _normalize_status(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


def _has_timestamp(binding: _NormalizedEvidence) -> bool:
    return binding.timestamp_start is not None or binding.timestamp_end is not None


def _segment_key(binding: _NormalizedEvidence) -> tuple[Any, ...]:
    if binding.source:
        return (binding.source,)
    if binding.timestamp_start is not None or binding.timestamp_end is not None:
        return (
            round(binding.timestamp_start or 0.0, 3),
            round(binding.timestamp_end or binding.timestamp_start or 0.0, 3),
        )
    return (binding.target_ref,)


def _timestamps_overlap(
    left: _NormalizedEvidence,
    right: _NormalizedEvidence,
    tolerance_seconds: float,
) -> bool:
    if left.timestamp_start is None or right.timestamp_start is None:
        return False
    return abs(left.timestamp_start - right.timestamp_start) <= tolerance_seconds


def _is_subsequence(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    if not needle:
        return True
    position = 0
    for item in haystack:
        if item == needle[position]:
            position += 1
            if position == len(needle):
                return True
    return False


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _tuple_field(obj: Any, *names: str) -> tuple[str, ...]:
    value = _field(obj, *names, default=())
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    return tuple(str(item) for item in value)


def _int_field(obj: Any, *names: str, default: int = 0) -> int:
    value = _field(obj, *names, default=default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


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


def _all_option_target_refs(registry: Any) -> tuple[str, ...]:
    refs: list[str] = []
    option_ref_index = _field(registry, "option_ref_index", default={})
    if hasattr(option_ref_index, "values"):
        for value in option_ref_index.values():
            refs.extend(_coerce_refs(value))
    options_by_id = _field(registry, "options_by_id", default={})
    if hasattr(options_by_id, "values"):
        for option in options_by_id.values():
            refs.extend(_option_target_refs(option))
    if not refs:
        targets_by_id = _field(registry, "targets_by_id", default={})
        if hasattr(targets_by_id, "keys"):
            refs.extend(str(target_ref) for target_ref in targets_by_id.keys())
    return _unique_ids(refs)


def _coerce_refs(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return (str(value),)


def _per_option_coverage_program(registry: Any) -> tuple[dict[str, Any], ...]:
    target_refs = _all_option_target_refs(registry)
    args: dict[str, Any] = {
        "target_refs": list(target_refs),
        "group_by_option": True,
        "top_k": MAIN_IDEA_RECOVERY_TOP_K,
    }
    return ({"tool": "target_coverage", "args": args},)


def _accept(option_id: str, *, supporting_evidence_ids: Sequence[str]) -> FinalGateDecision:
    decision = FinalGateDecision(
        proposed_option=option_id,
        gate_status="accepted",
        supporting_evidence_ids=_unique_ids(supporting_evidence_ids),
    )
    return _attach_feedback_fields(decision)


def _reject(
    option_id: str,
    reason_code: FinalRejectionReason,
    *,
    supporting_evidence_ids: Sequence[str] = (),
    missing_target_refs: Sequence[str] = (),
    missing_relation_refs: Sequence[str] = (),
    actionable_next_program: Sequence[dict[str, Any]] = (),
    do_not_repeat: Sequence[str] = (),
) -> FinalGateDecision:
    decision = FinalGateDecision(
        proposed_option=option_id,
        gate_status="rejected",
        reason_code=reason_code,
        supporting_evidence_ids=_unique_ids(supporting_evidence_ids),
        missing_target_refs=tuple(missing_target_refs),
        missing_relation_refs=tuple(missing_relation_refs),
    )
    return _attach_feedback_fields(
        decision,
        actionable_next_program=actionable_next_program,
        do_not_repeat=do_not_repeat,
    )


def _attach_feedback_fields(
    decision: FinalGateDecision,
    *,
    actionable_next_program: Sequence[dict[str, Any]] = (),
    do_not_repeat: Sequence[str] = (),
) -> FinalGateDecision:
    object.__setattr__(decision, "actionable_next_program", tuple(actionable_next_program))
    object.__setattr__(decision, "do_not_repeat", tuple(do_not_repeat))
    return decision
