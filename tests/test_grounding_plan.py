from __future__ import annotations

import json
from dataclasses import replace

from visual_coding_agent_harness.agents.grounding.compiler import compile_grounding_plan
from visual_coding_agent_harness.agents.grounding.contracts import (
    GroundingOption,
    GroundingPlan,
    GroundingRelation,
    GroundingSubject,
    GroundingTarget,
)
from visual_coding_agent_harness.agents.grounding.planner import ground_question_with_model
from visual_coding_agent_harness.agents.grounding.validator import validate_grounding_plan
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.contracts import ClaimModality


class ScriptedGroundingBackend(VisionLanguageBackend):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if not self.responses:
            return BackendResponse(text="{}")
        return BackendResponse(text=self.responses.pop(0))


def _valid_plan() -> GroundingPlan:
    return GroundingPlan(
        route="temporal_order",
        recommended_skill="timeline_ordering",
        subjects=(GroundingSubject(subject_key="subject", canonical_name="Subject X", aliases=("X",)),),
        targets=(
            GroundingTarget(
                target_key="event_alpha",
                canonical_claim="Event Alpha occurs",
                subject_key="subject",
                claim_kind="visible_event",
                claim_modality="visual",
                aliases=("Alpha event",),
                search_queries=("Event Alpha",),
                polarity="affirmed",
            ),
            GroundingTarget(
                target_key="event_beta",
                canonical_claim="Event Beta occurs",
                subject_key="subject",
                claim_kind="visible_event",
                claim_modality="visual",
                aliases=("Beta event",),
                search_queries=("Event Beta",),
                polarity="affirmed",
            ),
        ),
        relations=(
            GroundingRelation(
                relation_key="alpha_before_beta",
                kind="before",
                source_target_key="event_alpha",
                destination_target_key="event_beta",
            ),
        ),
        options=(
            GroundingOption(
                option_id="A",
                ordered_target_keys=("event_alpha", "event_beta"),
                required_relation_keys=("alpha_before_beta",),
                raw_option_text="Alpha then Beta",
            ),
            GroundingOption(
                option_id="B",
                ordered_target_keys=("event_beta", "event_alpha"),
                raw_option_text="Beta then Alpha",
            ),
        ),
        acceptable_evidence_sources=("visual", "asr"),
        confidence=0.74,
        unresolved_ambiguities=(),
    )


def test_validate_grounding_plan_accepts_structural_plan() -> None:
    result = validate_grounding_plan(_valid_plan(), option_ids=("A", "B"))

    assert result.is_valid
    assert result.findings == ()


def test_validate_grounding_plan_rejects_unknown_target_ref() -> None:
    bad_plan = replace(
        _valid_plan(),
        options=(
            GroundingOption(
                option_id="A",
                ordered_target_keys=("event_alpha", "missing_event"),
                raw_option_text="Alpha then missing",
            ),
        ),
    )

    result = validate_grounding_plan(bad_plan, option_ids=("A",))

    assert not result.is_valid
    assert any("unknown target" in finding.message for finding in result.findings)


def test_validate_grounding_plan_rejects_invalid_enums_and_policy_values() -> None:
    bad_plan = replace(
        _valid_plan(),
        route="dream_route",
        recommended_skill="missing_skill",
        targets=(
            replace(
                _valid_plan().targets[0],
                claim_kind="whatever",
                claim_modality="narration",
                polarity="maybe",
            ),
        ),
        relations=(
            GroundingRelation(
                relation_key="bad_relation",
                kind="earlier_than",
                source_target_key="event_alpha",
                destination_target_key="event_alpha",
            ),
        ),
        options=(
            GroundingOption(
                option_id="A",
                required_target_keys=("event_alpha",),
                required_relation_keys=("bad_relation",),
                raw_option_text="Alpha then Beta",
            ),
        ),
        acceptable_evidence_sources=("dream",),
    )

    result = validate_grounding_plan(
        bad_plan,
        raw_options={"A": "Alpha then Beta"},
        skill_ids=("timeline_ordering",),
    )

    assert not result.is_valid
    messages = "\n".join(f"{finding.path}: {finding.message}" for finding in result.findings)
    assert "route" in messages
    assert "recommended_skill" in messages
    assert "claim_kind" in messages
    assert "claim_modality" in messages
    assert "polarity" in messages
    assert "relations[0].kind" in messages
    assert "acceptable_evidence_sources" in messages


def test_validate_grounding_plan_requires_exact_raw_option_set_and_text() -> None:
    bad_plan = replace(
        _valid_plan(),
        options=(
            GroundingOption(
                option_id="A",
                ordered_target_keys=("event_alpha", "event_beta"),
                raw_option_text="model rewritten option",
            ),
            GroundingOption(
                option_id="C",
                ordered_target_keys=("event_alpha",),
                raw_option_text="extra option",
            ),
        ),
    )

    result = validate_grounding_plan(
        bad_plan,
        raw_options={"A": "Alpha then Beta", "B": "Beta then Alpha"},
    )

    assert not result.is_valid
    messages = "\n".join(f"{finding.path}: {finding.message}" for finding in result.findings)
    assert "missing option(s): B" in messages
    assert "extra option(s): C" in messages
    assert "raw_option_text must match framework raw option text" in messages


def test_compile_grounding_plan_assigns_stable_registry_ids_and_hash() -> None:
    compiled = compile_grounding_plan(_valid_plan())

    assert compiled.target_key_to_id == {"event_alpha": "T1", "event_beta": "T2"}
    assert compiled.relation_key_to_id == {"alpha_before_beta": "R1"}
    assert compiled.registry.resolve_target_ref("T1").canonical_text == "Event Alpha occurs"
    assert compiled.registry.resolve_target_ref("T1").aliases == ("Alpha event", "Event Alpha")
    assert compiled.registry.resolve_target_ref("T1").modality_hint == ClaimModality.VISUAL_FACT
    assert compiled.registry.option_for("A").target_sequence == ("T1", "T2")
    assert compiled.registry.option_for("A").required_relations == ("R1",)
    assert compiled.registry.version.startswith("grounding:v1:")

    compiled_again = compile_grounding_plan(_valid_plan())
    assert compiled.plan_hash == compiled_again.plan_hash
    assert compiled.registry.version == compiled_again.registry.version


def test_compile_grounding_plan_preserves_runtime_policy_and_target_metadata() -> None:
    plan = replace(
        _valid_plan(),
        unresolved_ambiguities=("needs transcript confirmation",),
        options=(
            replace(_valid_plan().options[0], raw_option_text="model rewrite A"),
            replace(_valid_plan().options[1], raw_option_text="model rewrite B"),
        ),
    )

    compiled = compile_grounding_plan(
        plan,
        raw_options={"A": "Alpha then Beta", "B": "Beta then Alpha"},
    )

    assert compiled.route == "temporal_order"
    assert compiled.recommended_skill_id == "timeline_ordering"
    assert compiled.acceptable_evidence_sources == ("visual", "asr")
    assert compiled.unresolved_ambiguities == ("needs transcript confirmation",)
    assert compiled.raw_options == {"A": "Alpha then Beta", "B": "Beta then Alpha"}
    target = compiled.registry.resolve_target_ref("T1")
    assert target.claim_kind == "visible_event"
    assert target.polarity == "affirmed"
    assert target.acceptable_evidence_sources == ("visual", "asr")
    assert target.relation is None


def test_ground_question_retries_once_then_uses_valid_plan() -> None:
    backend = ScriptedGroundingBackend(
        [
            json.dumps({"route": "temporal_order", "targets": []}),
            json.dumps(_valid_plan().to_dict()),
        ]
    )

    result = ground_question_with_model(
        backend,
        question="Question: Which event happens first?",
        options=("A. Alpha then Beta", "B. Beta then Alpha"),
    )

    assert result.plan is not None
    assert result.validation.is_valid
    assert result.attempts == 2
    assert [request.task for request in backend.requests] == ["ground_question", "ground_question"]
    assert "Validation feedback" in backend.requests[1].prompt


def test_grounding_prompt_keeps_task_specific_claim_text_and_neutral_keys() -> None:
    backend = ScriptedGroundingBackend([json.dumps(_valid_plan().to_dict())])

    ground_question_with_model(
        backend,
        question="Question: Which named event happens first?",
        options=("A. Alpha then Beta", "B. Beta then Alpha"),
    )

    prompt = backend.requests[0].prompt
    assert "Use task-specific, option-faithful canonical claims" in prompt
    assert "Use domain-neutral temporary keys only" in prompt
    assert "Use domain-neutral wording in the plan" not in prompt


def test_ground_question_falls_back_unstructured_after_retry() -> None:
    backend = ScriptedGroundingBackend(
        [
            json.dumps({"route": "temporal_order", "targets": []}),
            json.dumps({"route": "temporal_order", "targets": []}),
        ]
    )

    result = ground_question_with_model(
        backend,
        question="Question: Which event happens first?",
        options=("A. Alpha then Beta", "B. Beta then Alpha"),
    )

    assert result.plan is None
    assert not result.validation.is_valid
    assert result.fallback_reason == "grounding_validation_failed"
    assert result.attempts == 2
