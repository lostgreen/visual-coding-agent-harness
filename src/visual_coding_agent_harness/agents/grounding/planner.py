"""Planner-model entry point for generic question grounding."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from ...backends.base import BackendRequest, VisionLanguageBackend
from .contracts import GroundingPlan
from .validator import GroundingValidationResult, validate_grounding_plan


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
    validation = GroundingValidationResult(is_valid=False)
    raw_text = ""
    feedback = ""
    attempts = 0
    for attempt in range(max(0, int(max_retries)) + 1):
        attempts = attempt + 1
        response = backend.generate(
            BackendRequest(
                task="ground_question",
                prompt=_grounding_prompt(question=question, options=options, route_hint=route_hint, feedback=feedback),
                max_new_tokens=1200,
                temperature=0.0,
            )
        )
        raw_text = response.text
        try:
            plan = GroundingPlan.from_mapping(json.loads(_extract_json_object(raw_text)))
        except (json.JSONDecodeError, ValueError):
            validation = GroundingValidationResult(is_valid=False)
            feedback = "Previous response was not a valid JSON object matching the GroundingPlan schema."
            continue
        validation = validate_grounding_plan(plan, option_ids=option_ids)
        if validation.is_valid:
            return GroundingPlannerResult(plan=plan, validation=validation, raw_text=raw_text, attempts=attempts)
        feedback = validation.feedback()
    return GroundingPlannerResult(
        plan=None,
        validation=validation,
        raw_text=raw_text,
        attempts=attempts,
        fallback_reason="grounding_validation_failed",
    )


def _grounding_prompt(
    *,
    question: str,
    options: Sequence[str],
    route_hint: str,
    feedback: str,
) -> str:
    option_text = "\n".join(str(option) for option in options)
    feedback_block = f"\nValidation feedback from the previous attempt:\n{feedback}\n" if feedback else ""
    return (
        "Create a GroundingPlan for a long-video question. Do not answer the question and do not choose an option.\n"
        "Identify only the minimal subjects, claims, events, states, and relations needed to distinguish the options.\n"
        "Preserve semantic differences between initial states, later transitions, attributes, and ordered events.\n"
        "Output strict JSON with keys: route, recommended_skill, subjects, targets, relations, options, "
        "acceptable_evidence_sources, confidence, unresolved_ambiguities.\n"
        "Use temporary keys such as subject_main, event_alpha, relation_1; the framework will assign T/R IDs.\n"
        "Each target requires target_key, canonical_claim, subject_key, claim_kind, claim_modality, aliases, "
        "search_queries, polarity. Each option requires option_id, required_target_keys, ordered_target_keys, "
        "required_relation_keys, raw_option_text.\n"
        "Do not assert that any claim is true; describe what evidence would need to be checked.\n"
        "Use domain-neutral wording in the plan; do not rely on memorized examples.\n"
        f"{feedback_block}\n"
        f"Route hint: {route_hint or '(none)'}\n"
        f"Question:\n{question}\n"
        f"Options:\n{option_text}\n"
    )


def _extract_json_object(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return match.group(0)


def _option_id(value: object) -> str:
    match = re.match(r"\s*([A-H])(?:[.)]\s*|\s+|$)", str(value), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""
