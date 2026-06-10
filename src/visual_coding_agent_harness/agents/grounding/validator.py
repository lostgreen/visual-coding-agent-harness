"""Structural validation for planner-owned grounding plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import (
    ALLOWED_CLAIM_KINDS,
    ALLOWED_EVIDENCE_SOURCES,
    ALLOWED_GROUNDING_MODALITIES,
    ALLOWED_GROUNDING_POLARITIES,
    ALLOWED_GROUNDING_ROUTES,
    ALLOWED_OPTION_KINDS,
    ALLOWED_RELATION_KINDS,
    GroundingPlan,
)

_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_:-]{0,63}$")
_OPTION_RE = re.compile(r"^[A-H]$")


@dataclass(frozen=True)
class GroundingValidationFinding:
    path: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class GroundingValidationResult:
    is_valid: bool
    findings: tuple[GroundingValidationFinding, ...] = ()

    def feedback(self) -> str:
        if not self.findings:
            return "No validation errors."
        return "\n".join(f"- {finding.path}: {finding.message}" for finding in self.findings)


def validate_grounding_plan(
    plan: GroundingPlan,
    *,
    option_ids: Sequence[str] = (),
    raw_options: Mapping[str, str] | None = None,
    skill_ids: Sequence[str] = (),
    max_targets: int = 24,
    max_aliases_per_target: int = 12,
) -> GroundingValidationResult:
    findings: list[GroundingValidationFinding] = []
    _require_nonempty(plan.route, "route", findings)
    _require_nonempty(plan.recommended_skill, "recommended_skill", findings)
    route = _normalized_value(plan.route)
    if route and route not in ALLOWED_GROUNDING_ROUTES:
        findings.append(GroundingValidationFinding("route", f"invalid route: {plan.route}"))
    allowed_skills = {_skill_name(skill_id) for skill_id in skill_ids if _skill_name(skill_id)}
    recommended_skill = _skill_name(plan.recommended_skill)
    if allowed_skills and recommended_skill not in allowed_skills:
        findings.append(
            GroundingValidationFinding("recommended_skill", f"unknown skill: {plan.recommended_skill}")
        )
    for index, source in enumerate(plan.acceptable_evidence_sources):
        normalized_source = _normalized_value(source)
        if normalized_source not in ALLOWED_EVIDENCE_SOURCES:
            findings.append(
                GroundingValidationFinding(
                    f"acceptable_evidence_sources[{index}]",
                    f"invalid evidence source: {source}",
                )
            )

    central_subjects = tuple(_normalize_space(subject) for subject in plan.central_subjects if _normalize_space(subject))
    if not central_subjects:
        findings.append(GroundingValidationFinding("central_subjects", "at least one central subject is required"))

    subject_keys = _unique_keys(
        (subject.subject_key for subject in plan.subjects),
        path="subjects",
        findings=findings,
    )
    for index, subject in enumerate(plan.subjects):
        path = f"subjects[{index}]"
        _require_key(subject.subject_key, f"{path}.subject_key", findings)
        _require_nonempty(subject.canonical_name, f"{path}.canonical_name", findings)

    if not plan.targets:
        findings.append(GroundingValidationFinding("targets", "at least one target is required"))
    if len(plan.targets) > max_targets:
        findings.append(GroundingValidationFinding("targets", f"too many targets: {len(plan.targets)} > {max_targets}"))
    target_keys = _unique_keys(
        (target.target_key for target in plan.targets),
        path="targets",
        findings=findings,
    )
    for index, target in enumerate(plan.targets):
        path = f"targets[{index}]"
        _require_key(target.target_key, f"{path}.target_key", findings)
        _require_nonempty(target.canonical_claim, f"{path}.canonical_claim", findings)
        if _normalized_value(target.claim_kind) not in ALLOWED_CLAIM_KINDS:
            findings.append(GroundingValidationFinding(f"{path}.claim_kind", f"invalid claim kind: {target.claim_kind}"))
        if _normalized_value(target.claim_modality) not in ALLOWED_GROUNDING_MODALITIES:
            findings.append(
                GroundingValidationFinding(
                    f"{path}.claim_modality",
                    f"invalid claim modality: {target.claim_modality}",
                )
            )
        if _normalized_value(target.polarity) not in ALLOWED_GROUNDING_POLARITIES:
            findings.append(GroundingValidationFinding(f"{path}.polarity", f"invalid polarity: {target.polarity}"))
        if target.subject_key and target.subject_key not in subject_keys:
            findings.append(GroundingValidationFinding(f"{path}.subject_key", f"unknown subject: {target.subject_key}"))
        if len(target.aliases) > max_aliases_per_target:
            findings.append(
                GroundingValidationFinding(
                    f"{path}.aliases",
                    f"too many aliases: {len(target.aliases)} > {max_aliases_per_target}",
                )
            )
        if target.target_key.upper() in {"A", "B", "C", "D", "E", "F", "G", "H"}:
            findings.append(GroundingValidationFinding(f"{path}.target_key", "option letter cannot be a target key"))

    target_subject_surfaces: list[str] = []
    for target in plan.targets:
        canonical = _normalize_space(target.canonical_claim)
        if canonical:
            target_subject_surfaces.append(canonical)
        for alias in target.aliases:
            alias_text = _normalize_space(alias)
            if alias_text:
                target_subject_surfaces.append(alias_text)
    lowered_surfaces = tuple(surface.lower() for surface in target_subject_surfaces)
    for index, central_subject in enumerate(central_subjects):
        needle = central_subject.lower()
        if not any(needle == surface or needle in surface or surface in needle for surface in lowered_surfaces):
            findings.append(
                GroundingValidationFinding(
                    f"central_subjects[{index}]",
                    "central subject must appear in (or contain) at least one target canonical claim or alias",
                )
            )

    relation_keys = _unique_keys(
        (relation.relation_key for relation in plan.relations),
        path="relations",
        findings=findings,
    )
    for index, relation in enumerate(plan.relations):
        path = f"relations[{index}]"
        _require_key(relation.relation_key, f"{path}.relation_key", findings)
        if _normalized_value(relation.kind) not in ALLOWED_RELATION_KINDS:
            findings.append(GroundingValidationFinding(f"{path}.kind", f"invalid relation kind: {relation.kind}"))
        if relation.source_target_key not in target_keys:
            findings.append(
                GroundingValidationFinding(
                    f"{path}.source_target_key",
                    f"unknown target: {relation.source_target_key}",
                )
            )
        if relation.destination_target_key not in target_keys:
            findings.append(
                GroundingValidationFinding(
                    f"{path}.destination_target_key",
                    f"unknown target: {relation.destination_target_key}",
                )
            )

    normalized_raw_options = _normalize_raw_options(raw_options or {})
    normalized_option_ids = (
        tuple(normalized_raw_options)
        if normalized_raw_options
        else tuple(_option_id(option) for option in option_ids if _option_id(option))
    )
    plan_option_ids = _unique_keys(
        (option.option_id for option in plan.options),
        path="options",
        findings=findings,
    )
    if normalized_option_ids:
        missing = [option_id for option_id in normalized_option_ids if option_id not in plan_option_ids]
        if missing:
            findings.append(GroundingValidationFinding("options", "missing option(s): " + ", ".join(missing)))
        extra = [option_id for option_id in plan_option_ids if option_id not in normalized_option_ids]
        if extra:
            findings.append(GroundingValidationFinding("options", "extra option(s): " + ", ".join(sorted(extra))))
    for index, option in enumerate(plan.options):
        path = f"options[{index}]"
        if not _OPTION_RE.fullmatch(str(option.option_id)):
            findings.append(GroundingValidationFinding(f"{path}.option_id", f"invalid option id: {option.option_id}"))
        option_kind = _normalized_value(option.option_kind)
        if not option_kind:
            findings.append(GroundingValidationFinding(f"{path}.option_kind", "required"))
        elif option_kind not in ALLOWED_OPTION_KINDS:
            findings.append(GroundingValidationFinding(f"{path}.option_kind", f"invalid option kind: {option.option_kind}"))
        # NOTE: option.raw_option_text is authoritatively overwritten by
        # compiler._plan_with_framework_raw_options(), so the framework, not the LLM,
        # is the single source of truth for option surface text. We deliberately do
        # not enforce string equality here -- doing so makes bootstrap brittle to
        # benign LLM normalization (smart quotes, double spaces, etc.) and produces
        # zero planner turns on otherwise structurally valid plans.
        if not option.required_target_keys and not option.ordered_target_keys and not option.required_relation_keys:
            findings.append(GroundingValidationFinding(path, "option has no target or relation requirements"))
        for field_name, keys, known_keys, label in (
            ("required_target_keys", option.required_target_keys, target_keys, "target"),
            ("ordered_target_keys", option.ordered_target_keys, target_keys, "target"),
            ("required_relation_keys", option.required_relation_keys, relation_keys, "relation"),
        ):
            if len(keys) != len(set(keys)):
                findings.append(GroundingValidationFinding(f"{path}.{field_name}", "duplicate refs are not allowed"))
            for key in keys:
                if key not in known_keys:
                    findings.append(GroundingValidationFinding(f"{path}.{field_name}", f"unknown {label}: {key}"))
        option_target_keys = set(option.required_target_keys) | set(option.ordered_target_keys)
        for relation_key in option.required_relation_keys:
            relation = next((item for item in plan.relations if item.relation_key == relation_key), None)
            if relation is None:
                continue
            if relation.source_target_key not in option_target_keys or relation.destination_target_key not in option_target_keys:
                findings.append(
                    GroundingValidationFinding(
                        f"{path}.required_relation_keys",
                        f"relation {relation_key} endpoints must be included in option targets",
                    )
                )

    return GroundingValidationResult(is_valid=not findings, findings=tuple(findings))


def _unique_keys(
    values: Sequence[str],
    *,
    path: str,
    findings: list[GroundingValidationFinding],
) -> set[str]:
    keys: set[str] = set()
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        if key in keys:
            findings.append(GroundingValidationFinding(path, f"duplicate key: {key}"))
        keys.add(key)
    return keys


def _require_key(value: str, path: str, findings: list[GroundingValidationFinding]) -> None:
    text = str(value or "").strip()
    if not text:
        findings.append(GroundingValidationFinding(path, "required"))
    elif not _KEY_RE.fullmatch(text):
        findings.append(GroundingValidationFinding(path, f"invalid key: {text}"))


def _require_nonempty(value: str, path: str, findings: list[GroundingValidationFinding]) -> None:
    if not str(value or "").strip():
        findings.append(GroundingValidationFinding(path, "required"))


def _option_id(value: object) -> str:
    match = re.match(r"\s*([A-H])(?:[.)]\s*|\s+|$)", str(value), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _normalize_raw_options(raw_options: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for option_id, option_text in raw_options.items():
        resolved_id = _option_id(option_id) or str(option_id).strip().upper()[:1]
        if not _OPTION_RE.fullmatch(resolved_id):
            continue
        normalized[resolved_id] = _normalize_space(option_text)
    return normalized


def _normalize_space(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalized_value(value: object) -> str:
    return _normalize_space(value).lower()


def _skill_name(value: object) -> str:
    return _normalize_space(value).split("@", 1)[0].strip()
