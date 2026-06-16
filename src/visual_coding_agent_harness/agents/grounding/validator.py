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
    GroundingTarget,
)

_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_:-]{0,63}$")
_OPTION_RE = re.compile(r"^[A-H]$")
_PREDECIDED_CLAIM_RE = re.compile(
    r"^\s*(?:"
    r"(?:the\s+)?(?:main\s+idea|answer|correct\s+answer)\s+(?:of\s+the\s+video\s+)?(?:is|was|would\s+be)\b|"
    r"option\s+[A-H]\s+(?:is|was)\s+(?:correct|the\s+answer)\b"
    r")",
    flags=re.IGNORECASE,
)
_ORDER_POSITION_WORD = (
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final|"
    r"1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)"
)
_ORDER_VERB_CONTEXT = (
    r"(?:present(?:ed|s|ing)?|list(?:ed|s|ing)?|show(?:n|s|ed|ing)?|appear(?:s|ed|ing)?|"
    r"introduc(?:ed|es|ing)?|plac(?:ed|es|ing)?|rank(?:ed|s|ing)?|order(?:ed|s|ing)?)"
)
_ORDER_NOUN_CONTEXT = r"(?:sequence|order|timeline|list)"
_TEMPORAL_TARGET_POSITION_RE = re.compile(
    rf"\b{_ORDER_VERB_CONTEXT}\b[^.:\n]{{0,64}}\b{_ORDER_POSITION_WORD}\b|"
    rf"\b{_ORDER_POSITION_WORD}\b[^.:\n]{{0,64}}\b(?:in|within|of|from)?\s*(?:the\s+)?{_ORDER_NOUN_CONTEXT}\b",
    flags=re.IGNORECASE,
)
_TEMPORAL_TARGET_POSITION_MESSAGE = (
    "temporal-order targets must be order-neutral items/events; put first/second/third/fourth ordering "
    "in option ordered_target_keys or relations, not target text"
)


@dataclass(frozen=True)
class GroundingValidationFinding:
    path: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class GroundingValidationResult:
    is_valid: bool
    findings: tuple[GroundingValidationFinding, ...] = ()

    def feedback(self, *, max_findings: int | None = None) -> str:
        if not self.findings:
            return "No validation errors."
        findings = self.findings[:max(0, max_findings)] if max_findings is not None else self.findings
        suffix = ""
        if max_findings is not None and len(self.findings) > len(findings):
            suffix = f"\n- ... {len(self.findings) - len(findings)} more validation finding(s) omitted"
        return "\n".join(f"- {finding.path}: {finding.message}" for finding in findings) + suffix


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
            GroundingValidationFinding("recommended_skill", f"unknown skill: {plan.recommended_skill}", "warning")
        )
    for index, source in enumerate(plan.acceptable_evidence_sources):
        normalized_source = _normalized_value(source)
        if normalized_source not in ALLOWED_EVIDENCE_SOURCES:
            findings.append(
                GroundingValidationFinding(
                    f"acceptable_evidence_sources[{index}]",
                    f"invalid evidence source: {source}",
                    "warning",
                )
            )

    central_subjects = tuple(_normalize_space(subject) for subject in plan.central_subjects if _normalize_space(subject))
    if not central_subjects:
        findings.append(
            GroundingValidationFinding("central_subjects", "at least one central subject is required", "warning")
        )

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
        if _predecides_answer(target.canonical_claim):
            findings.append(
                GroundingValidationFinding(
                    f"{path}.canonical_claim",
                    "canonical claim appears to pre-decide an option instead of naming a checkable video fact",
                    "warning",
                )
            )
        if route == "temporal_order":
            for surface_path, surface in _target_text_surfaces(path, target):
                if _temporal_target_mentions_option_position(surface):
                    findings.append(
                        GroundingValidationFinding(
                            surface_path,
                            _TEMPORAL_TARGET_POSITION_MESSAGE,
                        )
                    )
        if _normalized_value(target.claim_kind) not in ALLOWED_CLAIM_KINDS:
            findings.append(
                GroundingValidationFinding(f"{path}.claim_kind", f"invalid claim kind: {target.claim_kind}", "warning")
            )
        if _normalized_value(target.claim_modality) not in ALLOWED_GROUNDING_MODALITIES:
            findings.append(
                GroundingValidationFinding(
                    f"{path}.claim_modality",
                    f"invalid claim modality: {target.claim_modality}",
                    "warning",
                )
            )
        if _normalized_value(target.polarity) not in ALLOWED_GROUNDING_POLARITIES:
            findings.append(GroundingValidationFinding(f"{path}.polarity", f"invalid polarity: {target.polarity}", "warning"))
        if target.subject_key and target.subject_key not in subject_keys:
            findings.append(GroundingValidationFinding(f"{path}.subject_key", f"unknown subject: {target.subject_key}"))
        if len(target.aliases) > max_aliases_per_target:
            findings.append(
                GroundingValidationFinding(
                    f"{path}.aliases",
                    f"too many aliases: {len(target.aliases)} > {max_aliases_per_target}",
                    "warning",
                )
            )
        if target.target_key.upper() in {"A", "B", "C", "D", "E", "F", "G", "H"}:
            findings.append(GroundingValidationFinding(f"{path}.target_key", "option letter cannot be a target key"))
        for discriminator in target.discriminators:
            if _generic_discriminator(discriminator):
                findings.append(
                    GroundingValidationFinding(
                        f"{path}.discriminators",
                        f"generic discriminator: {discriminator}",
                        "warning",
                    )
                )

    if plan.targets and all(not tuple(target.discriminators) for target in plan.targets):
        findings.append(
            GroundingValidationFinding(
                "targets.discriminators",
                "all targets have zero discriminators; add short option-unique phrases when available",
                "warning",
            )
        )
    discriminator_owner: dict[str, str] = {}
    for index, target in enumerate(plan.targets):
        for discriminator in target.discriminators:
            normalized = _normalized_discriminator(discriminator)
            if not normalized:
                continue
            prior = discriminator_owner.get(normalized)
            if prior is not None and prior != target.target_key:
                findings.append(
                    GroundingValidationFinding(
                        f"targets[{index}].discriminators",
                        f"overlapping discriminator shared with {prior}: {discriminator}",
                        "warning",
                    )
                )
            discriminator_owner.setdefault(normalized, target.target_key)

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
                    "warning",
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

    return GroundingValidationResult(
        is_valid=not any(finding.severity != "warning" for finding in findings),
        findings=tuple(findings),
    )


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


def _predecides_answer(value: object) -> bool:
    return bool(_PREDECIDED_CLAIM_RE.search(_normalize_space(value)))


def _target_text_surfaces(path: str, target: GroundingTarget) -> tuple[tuple[str, object], ...]:
    surfaces: list[tuple[str, object]] = [(f"{path}.canonical_claim", target.canonical_claim)]
    for index, alias in enumerate(target.aliases):
        surfaces.append((f"{path}.aliases[{index}]", alias))
    for index, query in enumerate(target.search_queries):
        surfaces.append((f"{path}.search_queries[{index}]", query))
    for index, discriminator in enumerate(target.discriminators):
        surfaces.append((f"{path}.discriminators[{index}]", discriminator))
    return tuple(surfaces)


def _temporal_target_mentions_option_position(value: object) -> bool:
    return bool(_TEMPORAL_TARGET_POSITION_RE.search(_normalize_space(value)))


def _skill_name(value: object) -> str:
    return _normalize_space(value).split("@", 1)[0].strip()


def _normalized_discriminator(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _normalized_value(value)).strip()


def _generic_discriminator(value: object) -> bool:
    tokens = _normalized_discriminator(value).split()
    if not tokens:
        return False
    generic_terms = {"video", "shown", "scene", "thing", "event", "object", "generic", "option"}
    return all(token in generic_terms for token in tokens)
