"""Compile validated grounding plans into frozen target registries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ...contracts import ClaimModality, ClaimRelation, OptionSpec, TargetRegistry, TargetSpec
from .contracts import GroundingPlan
from .validator import validate_grounding_plan


@dataclass(frozen=True)
class CompiledGroundingPlan:
    registry: TargetRegistry
    target_key_to_id: dict[str, str]
    relation_key_to_id: dict[str, str]
    plan_hash: str


def compile_grounding_plan(plan: GroundingPlan) -> CompiledGroundingPlan:
    validation = validate_grounding_plan(plan)
    if not validation.is_valid:
        raise ValueError("Invalid GroundingPlan:\n" + validation.feedback())

    plan_hash = _plan_hash(plan)
    target_key_to_id = {target.target_key: f"T{index}" for index, target in enumerate(plan.targets, start=1)}
    relation_key_to_id = {relation.relation_key: f"R{index}" for index, relation in enumerate(plan.relations, start=1)}
    subjects_by_key = {subject.subject_key: subject for subject in plan.subjects}

    targets = []
    for target in plan.targets:
        subject = subjects_by_key.get(str(target.subject_key or ""))
        aliases = _unique_strings([*target.aliases, *target.search_queries])
        targets.append(
            TargetSpec(
                target_id=target_key_to_id[target.target_key],
                canonical_text=target.canonical_claim,
                aliases=aliases,
                subject=subject.canonical_name if subject is not None else None,
                relation=target.claim_kind,
                modality_hint=_claim_modality(target.claim_modality),
                source="grounding_plan",
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
    )


def _plan_hash(plan: GroundingPlan) -> str:
    canonical = json.dumps(plan.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
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
