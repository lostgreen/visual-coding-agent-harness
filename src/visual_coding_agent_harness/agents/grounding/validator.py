"""Structural validation for planner-owned grounding plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .contracts import GroundingPlan

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
    max_targets: int = 24,
    max_aliases_per_target: int = 12,
) -> GroundingValidationResult:
    findings: list[GroundingValidationFinding] = []
    _require_nonempty(plan.route, "route", findings)
    _require_nonempty(plan.recommended_skill, "recommended_skill", findings)

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

    relation_keys = _unique_keys(
        (relation.relation_key for relation in plan.relations),
        path="relations",
        findings=findings,
    )
    for index, relation in enumerate(plan.relations):
        path = f"relations[{index}]"
        _require_key(relation.relation_key, f"{path}.relation_key", findings)
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

    normalized_option_ids = tuple(_option_id(option) for option in option_ids if _option_id(option))
    plan_option_ids = _unique_keys(
        (option.option_id for option in plan.options),
        path="options",
        findings=findings,
    )
    if normalized_option_ids:
        missing = [option_id for option_id in normalized_option_ids if option_id not in plan_option_ids]
        if missing:
            findings.append(GroundingValidationFinding("options", "missing option(s): " + ", ".join(missing)))
    for index, option in enumerate(plan.options):
        path = f"options[{index}]"
        if not _OPTION_RE.fullmatch(str(option.option_id)):
            findings.append(GroundingValidationFinding(f"{path}.option_id", f"invalid option id: {option.option_id}"))
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
