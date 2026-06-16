"""Compile validated grounding plans into frozen target registries."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from ...contracts import ClaimModality, ClaimRelation, OptionSpec, TargetRegistry, TargetSpec
from ..question_policy import (
    classify_narration_subroute,
    classify_question_route,
    extract_option_sequence_specs,
    extract_option_target_atoms_for_option,
)
from .contracts import ALLOWED_GROUNDING_ROUTES, GroundingOption, GroundingPlan, GroundingSubject, GroundingTarget
from .discriminators import derive_discriminators_lexical
from .order_hypotheses import OrderedEntity, OrderedSetSpec, OptionOrderHypothesis
from .operators import derive_answer_operator
from .validator import validate_grounding_plan


@dataclass(frozen=True)
class CompiledGroundingPlan:
    registry: TargetRegistry
    target_key_to_id: dict[str, str]
    relation_key_to_id: dict[str, str]
    plan_hash: str
    route: str
    recommended_skill_id: str
    answer_operator: str
    central_subjects: tuple[str, ...]
    acceptable_evidence_sources: tuple[str, ...]
    unresolved_ambiguities: tuple[str, ...]
    raw_options: dict[str, str]
    ordered_sets: tuple[OrderedSetSpec, ...] = ()


def compile_grounding_plan(
    plan: GroundingPlan,
    *,
    raw_options: Mapping[str, str] | None = None,
    skill_ids: tuple[str, ...] = (),
) -> CompiledGroundingPlan:
    normalized_raw_options = _normalize_raw_options(raw_options or {})
    plan = _plan_with_framework_raw_options(plan, normalized_raw_options)
    validation = validate_grounding_plan(plan, raw_options=normalized_raw_options, skill_ids=skill_ids)
    if not validation.is_valid:
        raise ValueError("Invalid GroundingPlan:\n" + validation.feedback())

    plan_hash = _plan_hash(plan, raw_options=normalized_raw_options)
    target_key_to_id = {target.target_key: f"T{index}" for index, target in enumerate(plan.targets, start=1)}
    relation_key_to_id = {relation.relation_key: f"R{index}" for index, relation in enumerate(plan.relations, start=1)}
    subjects_by_key = {subject.subject_key: subject for subject in plan.subjects}
    acceptable_evidence_sources = tuple(plan.acceptable_evidence_sources)

    targets = []
    for target in plan.targets:
        subject = subjects_by_key.get(str(target.subject_key or ""))
        aliases = _unique_strings(tuple(target.aliases))
        targets.append(
            TargetSpec(
                target_id=target_key_to_id[target.target_key],
                canonical_text=target.canonical_claim,
                aliases=aliases,
                search_queries=_unique_strings(tuple(target.search_queries)),
                discriminators=_unique_strings(tuple(target.discriminators)),
                subject=subject.canonical_name if subject is not None else None,
                relation=None,
                modality_hint=_claim_modality(target.claim_modality),
                source="grounding_plan",
                claim_kind=target.claim_kind,
                polarity=target.polarity,
                acceptable_evidence_sources=acceptable_evidence_sources,
            )
        )

    relations = [
        ClaimRelation(
            relation_id=relation_key_to_id[relation.relation_key],
            kind=relation.kind,
            source_target_id=target_key_to_id[relation.source_target_key],
            destination_target_id=target_key_to_id[relation.destination_target_key],
        )
        for relation in plan.relations
    ]

    options = []
    for option in plan.options:
        ordered_keys = tuple(option.ordered_target_keys) or tuple(option.required_target_keys)
        options.append(
            OptionSpec(
                option.option_id,
                target_sequence=tuple(target_key_to_id[key] for key in ordered_keys),
                required_relations=tuple(relation_key_to_id[key] for key in option.required_relation_keys),
                raw_option_text=option.raw_option_text,
                option_kind=option.option_kind,
            )
        )

    registry = TargetRegistry.from_specs(
        targets=targets,
        options=options,
        relations=relations,
        version=f"grounding:v1:{plan_hash[:12]}",
    )
    return CompiledGroundingPlan(
        registry=registry,
        target_key_to_id=target_key_to_id,
        relation_key_to_id=relation_key_to_id,
        plan_hash=plan_hash,
        route=plan.route,
        recommended_skill_id=_skill_id(plan.recommended_skill),
        answer_operator=plan.answer_operator,
        central_subjects=tuple(plan.central_subjects),
        acceptable_evidence_sources=acceptable_evidence_sources,
        unresolved_ambiguities=tuple(plan.unresolved_ambiguities),
        raw_options=normalized_raw_options,
        ordered_sets=tuple(plan.ordered_sets),
    )


def compile_fallback_plan(
    question: str,
    options: Sequence[str],
    route_hint: str = "",
) -> GroundingPlan:
    route = _fallback_route(question, route_hint)
    option_items = _fallback_options(options)
    subject = GroundingSubject(
        subject_key="subject_main",
        canonical_name=_fallback_subject(question),
        aliases=(),
    )
    targets: list[GroundingTarget] = []
    grounding_options: list[GroundingOption] = []
    sequence_plan = _fallback_ordered_sequence_plan(
        question=question,
        option_items=option_items,
        subject_key=subject.subject_key,
        route=route,
    )
    if sequence_plan is not None:
        targets, grounding_options, ordered_sets = sequence_plan
        return GroundingPlan(
            route=route,
            recommended_skill=_fallback_recommended_skill(question, route),
            answer_operator=derive_answer_operator(question, route=route, options=options),
            central_subjects=(subject.canonical_name,),
            subjects=(subject,),
            targets=tuple(targets),
            relations=(),
            options=tuple(grounding_options),
            ordered_sets=tuple(ordered_sets),
            acceptable_evidence_sources=_fallback_evidence_sources(route),
            confidence=0.0,
            unresolved_ambiguities=("grounding_model_unavailable_or_invalid",),
        )
    for option_id, option_text in option_items:
        fallback_discriminators = derive_discriminators_lexical(dict(option_items))
        target_key = f"OPT_{option_id}_claim"
        aliases = tuple(
            atom
            for atom in extract_option_target_atoms_for_option(f"{option_id}. {option_text}", include_synonyms=False)
            if atom
        )
        targets.append(
            GroundingTarget(
                target_key=target_key,
                canonical_claim=option_text,
                subject_key=subject.subject_key,
                claim_kind=_fallback_claim_kind(route),
                claim_modality=_fallback_claim_modality(route),
                aliases=aliases,
                search_queries=aliases[:3],
                discriminators=fallback_discriminators.get(option_id, ()),
                polarity="unknown",
            )
        )
        grounding_options.append(
            GroundingOption(
                option_id=option_id,
                required_target_keys=(target_key,),
                ordered_target_keys=(target_key,),
                required_relation_keys=(),
                raw_option_text=option_text,
                option_kind=_fallback_option_kind(route),
            )
        )
    return GroundingPlan(
        route=route,
        recommended_skill=_fallback_recommended_skill(question, route),
        answer_operator=derive_answer_operator(question, route=route, options=options),
        central_subjects=(subject.canonical_name,),
        subjects=(subject,),
        targets=tuple(targets),
        relations=(),
        options=tuple(grounding_options),
        acceptable_evidence_sources=_fallback_evidence_sources(route),
        confidence=0.0,
        unresolved_ambiguities=("grounding_model_unavailable_or_invalid",),
    )


def _fallback_ordered_sequence_plan(
    *,
    question: str,
    option_items: Sequence[tuple[str, str]],
    subject_key: str,
    route: str,
) -> tuple[list[GroundingTarget], list[GroundingOption], list[OrderedSetSpec]] | None:
    if route != "temporal_order" or len(option_items) < 2:
        return None
    sequence_specs = extract_option_sequence_specs([f"{option_id}. {option_text}" for option_id, option_text in option_items])
    if len(sequence_specs) < 2:
        return None
    canonical_by_ref: dict[str, str] = {}
    for spec in sequence_specs.values():
        if len(spec.ordered_items) < 2:
            return None
        for ref, item in zip(spec.ordered_target_refs, spec.ordered_items):
            canonical_by_ref.setdefault(str(ref), item)
    if len(canonical_by_ref) < 2:
        return None
    ordered_set = _ordered_set_from_sequence_specs(canonical_by_ref=canonical_by_ref, sequence_specs=sequence_specs)

    targets: list[GroundingTarget] = []
    for index, ref in enumerate(sorted(canonical_by_ref, key=_target_ref_sort_key), start=1):
        item = canonical_by_ref[ref]
        targets.append(
            GroundingTarget(
                target_key=f"SEQ_item_{index}",
                canonical_claim=item,
                subject_key=subject_key,
                claim_kind=_fallback_claim_kind(route),
                claim_modality=_fallback_claim_modality(route),
                aliases=(item,),
                search_queries=(item,),
                polarity="affirmed",
            )
        )
    target_key_by_ref = {
        ref: f"SEQ_item_{index}"
        for index, ref in enumerate(sorted(canonical_by_ref, key=_target_ref_sort_key), start=1)
    }

    grounding_options: list[GroundingOption] = []
    for option_id, option_text in option_items:
        spec = sequence_specs.get(option_id)
        if spec is None:
            return None
        ordered_target_keys = tuple(target_key_by_ref.get(ref, "") for ref in spec.ordered_target_refs)
        if len(ordered_target_keys) != len(spec.ordered_target_refs) or any(not key for key in ordered_target_keys):
            return None
        grounding_options.append(
            GroundingOption(
                option_id=option_id,
                required_target_keys=ordered_target_keys,
                ordered_target_keys=ordered_target_keys,
                required_relation_keys=(),
                raw_option_text=option_text,
                option_kind=_fallback_option_kind(route),
            )
        )
    ordered_sets = [ordered_set] if ordered_set is not None else []
    return targets, grounding_options, ordered_sets


def _ordered_set_from_sequence_specs(
    *,
    canonical_by_ref: Mapping[str, str],
    sequence_specs: Mapping[str, Any],
) -> OrderedSetSpec | None:
    canonical_refs = tuple(sorted(canonical_by_ref, key=_target_ref_sort_key))
    canonical_ref_set = set(canonical_refs)
    if not canonical_refs:
        return None
    if not all(set(spec.ordered_target_refs) == canonical_ref_set for spec in sequence_specs.values()):
        return None
    entity_id_by_ref = {ref: f"E{index}" for index, ref in enumerate(canonical_refs, start=1)}
    return OrderedSetSpec(
        set_id="OS1",
        entities=tuple(
            OrderedEntity(entity_id=entity_id_by_ref[ref], canonical_name=canonical_by_ref[ref])
            for ref in canonical_refs
        ),
        hypotheses=tuple(
            OptionOrderHypothesis(
                option_id=option_id,
                ordered_entity_ids=tuple(entity_id_by_ref[ref] for ref in spec.ordered_target_refs),
            )
            for option_id, spec in sorted(sequence_specs.items())
        ),
    )


def _target_ref_sort_key(ref: str) -> tuple[int, str]:
    match = re.match(r"^T(\d+)$", str(ref))
    return (int(match.group(1)), str(ref)) if match else (10**9, str(ref))


def _plan_hash(plan: GroundingPlan, *, raw_options: Mapping[str, str]) -> str:
    payload = dict(plan.to_dict())
    if raw_options:
        payload["raw_options"] = dict(raw_options)
        payload["options"] = [
            {
                **option,
                "raw_option_text": raw_options.get(str(option.get("option_id", "")).upper(), ""),
            }
            for option in payload.get("options", [])
            if isinstance(option, dict)
        ]
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _claim_modality(value: str) -> ClaimModality:
    return {
        "visual": ClaimModality.VISUAL_FACT,
        "asr": ClaimModality.NARRATED_FACT,
        "ocr": ClaimModality.OCR_FACT,
        "mixed": ClaimModality.MIXED,
        "unknown": ClaimModality.UNKNOWN,
    }.get(str(value or "").strip().lower(), ClaimModality.UNKNOWN)


def _unique_strings(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return tuple(result)


def _normalize_raw_options(raw_options: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for option_id, option_text in raw_options.items():
        key = str(option_id or "").strip().upper()[:1]
        if key:
            normalized[key] = " ".join(str(option_text or "").split()).strip()
    return normalized


def _skill_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    name = text.split("@", 1)[0].strip()
    return name


def _plan_with_framework_raw_options(
    plan: GroundingPlan,
    raw_options: Mapping[str, str],
) -> GroundingPlan:
    if not raw_options:
        return plan
    normalized_options: list[GroundingOption] = []
    for option in plan.options:
        raw_option_text = raw_options.get(option.option_id, option.raw_option_text)
        normalized_options.append(replace(option, raw_option_text=raw_option_text))
    return replace(plan, options=tuple(normalized_options))


def _fallback_options(options: Sequence[str]) -> tuple[tuple[str, str], ...]:
    resolved: list[tuple[str, str]] = []
    for index, option in enumerate(options):
        match = re.match(r"\s*([A-H])[\).:-]?\s*(.*?)\s*$", str(option or ""), flags=re.IGNORECASE | re.DOTALL)
        if match:
            option_id = match.group(1).upper()
            option_text = " ".join(match.group(2).split()).strip()
        else:
            option_id = chr(ord("A") + index)
            option_text = " ".join(str(option or "").split()).strip()
        if not option_text:
            option_text = option_id
        resolved.append((option_id, option_text))
    return tuple(resolved)


def _fallback_subject(question: str) -> str:
    text = re.sub(r"\bOptions\s*:\s*.*", "", str(question or ""), flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\bQuestion\s*:\s*", "", text, flags=re.IGNORECASE)
    text = " ".join(text.split()).strip()
    return text[:160] or "video question"


def _fallback_recommended_skill(question: str, route: str) -> str:
    if route == "gist_global":
        return "main_idea"
    if route == "temporal_order":
        return "narration_timeline_qa" if classify_narration_subroute(question) == "narration_timeline" else "visual_timeline_qa"
    if route == "mixed_asr_visual":
        return "mixed_asr_visual_qa"
    return "grounded_factual_qa"


def _fallback_claim_kind(route: str) -> str:
    if route == "temporal_order":
        return "ordered_item"
    if route == "gist_global":
        return "topic"
    return "narrated_fact"


def _fallback_claim_modality(route: str) -> str:
    if route in {"temporal_order", "gist_global", "mixed_asr_visual"}:
        return "mixed"
    return "unknown"


def _fallback_option_kind(route: str) -> str:
    if route == "temporal_order":
        return "sequence"
    if route == "gist_global":
        return "topic_focus"
    if route == "mixed_asr_visual":
        return "mixed_fact"
    return "narrated_fact"


def _fallback_evidence_sources(route: str) -> tuple[str, ...]:
    if route == "gist_global":
        return ("asr", "visual", "global")
    if route == "temporal_order":
        return ("visual", "asr", "indexed_transcript")
    if route == "mixed_asr_visual":
        return ("mixed", "asr", "visual")
    return ("visual", "asr", "ocr")


def _fallback_route(question: str, route_hint: str) -> str:
    hinted = str(route_hint or "").strip()
    if hinted in ALLOWED_GROUNDING_ROUTES:
        return hinted
    classified = classify_question_route(question)
    if classified in ALLOWED_GROUNDING_ROUTES:
        return classified
    return "needle_local"
