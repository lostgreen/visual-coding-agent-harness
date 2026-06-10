"""Compile validated grounding plans into frozen target registries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Mapping

from ...contracts import ClaimModality, ClaimRelation, OptionSpec, TargetRegistry, TargetSpec
from .contracts import GroundingOption, GroundingPlan
from .validator import validate_grounding_plan


@dataclass(frozen=True)
class CompiledGroundingPlan:
    registry: TargetRegistry
    target_key_to_id: dict[str, str]
    relation_key_to_id: dict[str, str]
    plan_hash: str
    route: str
    recommended_skill_id: str
    acceptable_evidence_sources: tuple[str, ...]
    unresolved_ambiguities: tuple[str, ...]
    raw_options: dict[str, str]


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
        aliases = _unique_strings([*target.aliases, *target.search_queries])
        targets.append(
            TargetSpec(
                target_id=target_key_to_id[target.target_key],
                canonical_text=target.canonical_claim,
                aliases=aliases,
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
        acceptable_evidence_sources=acceptable_evidence_sources,
        unresolved_ambiguities=tuple(plan.unresolved_ambiguities),
        raw_options=normalized_raw_options,
    )


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
