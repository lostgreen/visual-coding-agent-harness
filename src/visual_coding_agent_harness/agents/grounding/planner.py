"""Planner-model entry point for generic question grounding."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from ...backends.base import BackendRequest, VisionLanguageBackend
from ..skills.specs import builtin_skill_registry
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
from .validator import GroundingValidationResult, validate_grounding_plan

GROUNDING_MAX_NEW_TOKENS = 4096


@dataclass(frozen=True)
class GroundingPlannerResult:
    plan: GroundingPlan | None
    validation: GroundingValidationResult
    raw_text: str
    attempts: int
    fallback_reason: str = ""


def ground_question_with_model(
    backend: VisionLanguageBackend,
    *,
    question: str,
    options: Sequence[str],
    route_hint: str = "",
    max_retries: int = 1,
) -> GroundingPlannerResult:
    option_ids = tuple(_option_id(option) for option in options if _option_id(option))
    raw_options = _raw_options_by_id(options)
    skill_ids = tuple(skill.name for skill in builtin_skill_registry().list())
    validation = GroundingValidationResult(is_valid=False)
    raw_text = ""
    feedback = ""
    fallback_reason = ""
    attempts = 0
    for attempt in range(max(0, int(max_retries)) + 1):
        attempts = attempt + 1
        response = backend.generate(
            BackendRequest(
                task="ground_question",
                prompt=_grounding_prompt(
                    question=question,
                    options=options,
                    route_hint=route_hint,
                    feedback=feedback,
                    skill_ids=skill_ids,
                ),
                max_new_tokens=GROUNDING_MAX_NEW_TOKENS,
                temperature=0.0,
            )
        )
        raw_text = response.text
        try:
            plan = GroundingPlan.from_mapping(json.loads(_extract_json_object(raw_text)))
        except (json.JSONDecodeError, ValueError):
            validation = GroundingValidationResult(is_valid=False)
            fallback_reason = "grounding_parse_failed"
            feedback = "Previous response was not a valid JSON object matching the GroundingPlan schema."
            continue
        validation = validate_grounding_plan(
            plan,
            option_ids=option_ids,
            raw_options=raw_options,
            skill_ids=skill_ids,
        )
        if validation.is_valid:
            return GroundingPlannerResult(plan=plan, validation=validation, raw_text=raw_text, attempts=attempts)
        fallback_reason = "grounding_validation_failed"
        feedback = validation.feedback()
    return GroundingPlannerResult(
        plan=None,
        validation=validation,
        raw_text=raw_text,
        attempts=attempts,
        fallback_reason=fallback_reason or "grounding_validation_failed",
    )


def _grounding_prompt(
    *,
    question: str,
    options: Sequence[str],
    route_hint: str,
    feedback: str,
    skill_ids: Sequence[str],
) -> str:
    option_text = "\n".join(str(option) for option in options)
    feedback_block = f"\nValidation feedback from the previous attempt:\n{feedback}\n" if feedback else ""
    routes = ", ".join(sorted(ALLOWED_GROUNDING_ROUTES))
    evidence_sources = ", ".join(sorted(ALLOWED_EVIDENCE_SOURCES))
    option_kinds = ", ".join(sorted(ALLOWED_OPTION_KINDS))
    claim_kinds = ", ".join(sorted(ALLOWED_CLAIM_KINDS))
    modalities = ", ".join(sorted(ALLOWED_GROUNDING_MODALITIES))
    polarities = ", ".join(sorted(ALLOWED_GROUNDING_POLARITIES))
    relation_kinds = ", ".join(sorted(ALLOWED_RELATION_KINDS))
    skills = ", ".join(str(skill_id) for skill_id in skill_ids)
    return (
        "Create a GroundingPlan for a long-video question. Do not answer the question and do not choose an option.\n"
        "Return ONLY one JSON object. Do not wrap it in markdown fences. Do not include explanations or schema notes.\n"
        "Identify only the minimal subjects, claims, events, states, and relations needed to distinguish the options.\n"
        "Preserve semantic differences between initial states, later transitions, attributes, and ordered events.\n"
        "Output strict JSON with keys: route, recommended_skill, central_subjects, subjects, targets, relations, options, "
        "acceptable_evidence_sources, confidence, unresolved_ambiguities.\n"
        f"route must be one of: {routes}.\n"
        f"recommended_skill must be one of: {skills}.\n"
        f"acceptable_evidence_sources values must be from: {evidence_sources}.\n"
        f"claim_kind must be one of: {claim_kinds}.\n"
        f"claim_modality must be one of: {modalities}. Use asr for narration/transcript claims; do not use narrated.\n"
        f"polarity must be one of: {polarities}. Use affirmed for positive claims; do not use positive or neutral.\n"
        f"relation.kind must be one of: {relation_kinds}. Every relation object requires relation_key, kind, "
        "source_target_key, and destination_target_key.\n"
        "subjects must be objects with keys \"subject_key\", \"canonical_name\", and \"aliases\"; do not output bare strings.\n"
        "central_subjects is a non-empty list of high-level subject strings that appear in, contain, or are contained by "
        "a target canonical_claim or alias.\n"
        "Use domain-neutral temporary keys such as subject_main, event_alpha, relation_1; the framework will assign T/R IDs.\n"
        "Use task-specific, option-faithful canonical claims, aliases, and search queries that preserve the words needed for retrieval.\n"
        "Each target requires target_key, canonical_claim, subject_key, claim_kind, claim_modality, aliases, "
        "search_queries, polarity. Each option requires option_id, required_target_keys, ordered_target_keys, "
        f"required_relation_keys, raw_option_text, option_kind. option_kind must be one of: {option_kinds}.\n"
        "Keep aliases and search_queries short: 1-3 concise strings per target are enough.\n"
        "Do not assert that any claim is true; describe what evidence would need to be checked.\n"
        "Use domain-neutral temporary keys only; do not rely on memorized examples.\n"
        f"{feedback_block}\n"
        f"Route hint: {route_hint or '(none)'}\n"
        f"Question:\n{question}\n"
        f"Options:\n{option_text}\n"
    )


def _extract_json_object(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    for candidate in _balanced_json_object_candidates(raw):
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate
    raise ValueError("No JSON object found")


def _balanced_json_object_candidates(raw: str) -> tuple[str, ...]:
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char != "}" or depth == 0:
            continue
        depth -= 1
        if depth == 0 and start is not None:
            candidates.append(raw[start : index + 1])
            start = None
    return tuple(candidates)


def _option_id(value: object) -> str:
    match = re.match(r"\s*([A-H])(?:[.)]\s*|\s+|$)", str(value), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _raw_options_by_id(options: Sequence[str]) -> dict[str, str]:
    raw_options: dict[str, str] = {}
    for option in options:
        option_id = _option_id(option)
        if not option_id:
            continue
        raw_options[option_id] = _strip_option_id(option)
    return raw_options


def _strip_option_id(option: object) -> str:
    return re.sub(r"^\s*[A-H][\).:-]\s*", "", str(option or ""), count=1, flags=re.IGNORECASE).strip()
