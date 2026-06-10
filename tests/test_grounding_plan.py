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
        central_subjects=("Subject X",),
        subjects=(GroundingSubject(subject_key="subject", canonical_name="Subject X", aliases=("X",)),),
        targets=(
            GroundingTarget(
                target_key="event_alpha",
                canonical_claim="Event Alpha occurs",
                subject_key="subject",
                claim_kind="visible_event",
                claim_modality="visual",
                aliases=("Alpha event", "Subject X"),
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
                option_kind="sequence",
            ),
            GroundingOption(
                option_id="B",
                ordered_target_keys=("event_beta", "event_alpha"),
                raw_option_text="Beta then Alpha",
                option_kind="sequence",
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
                option_kind="sequence",
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
                option_kind="dream_kind",
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
    assert "option_kind" in messages
    warning_paths = {finding.path for finding in result.findings if finding.severity == "warning"}
    assert "recommended_skill" in warning_paths
    assert "targets[0].claim_kind" in warning_paths
    assert "targets[0].claim_modality" in warning_paths
    assert "targets[0].polarity" in warning_paths


def test_validate_grounding_plan_requires_planner_owned_central_subjects_and_option_kind() -> None:
    bad_plan = replace(
        _valid_plan(),
        central_subjects=("Subject not in any target",),
        options=(
            replace(_valid_plan().options[0], option_kind=None),
            replace(_valid_plan().options[1], option_kind="sequence"),
        ),
    )

    result = validate_grounding_plan(
        bad_plan,
        raw_options={"A": "Alpha then Beta", "B": "Beta then Alpha"},
    )

    assert not result.is_valid
    messages = "\n".join(f"{finding.path}: {finding.message}" for finding in result.findings)
    assert "central_subjects[0]" in messages
    assert "option_kind" in messages


def test_validate_grounding_plan_requires_exact_option_id_set_but_tolerates_paraphrased_raw_text() -> None:
    # The compiler authoritatively overwrites raw_option_text with the framework value,
    # so the validator must not reject a plan solely because the LLM normalized
    # whitespace / quotes / punctuation in raw_option_text. Otherwise option text
    # containing literal quotes or double spaces hard-blocks bootstrap and produces
    # zero planner turns. Option *id* set is still enforced.
    bad_plan = replace(
        _valid_plan(),
        options=(
            GroundingOption(
                option_id="A",
                ordered_target_keys=("event_alpha", "event_beta"),
                raw_option_text="model rewritten option",  # paraphrased -- must be tolerated
                option_kind="sequence",
            ),
            GroundingOption(
                option_id="C",  # wrong id -- must still be flagged
                ordered_target_keys=("event_alpha",),
                raw_option_text="extra option",
                option_kind="sequence",
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
    assert "raw_option_text must match" not in messages


def test_validate_grounding_plan_accepts_paraphrased_raw_option_text_with_correct_ids() -> None:
    paraphrased = replace(
        _valid_plan(),
        options=(
            replace(_valid_plan().options[0], raw_option_text="Alpha  then  Beta"),
            replace(_valid_plan().options[1], raw_option_text="\u201cBeta\u201d then Alpha"),
        ),
    )

    result = validate_grounding_plan(
        paraphrased,
        raw_options={"A": "Alpha then Beta", "B": '"Beta" then Alpha'},
    )

    assert result.is_valid, result.feedback()


def test_validate_grounding_plan_accepts_central_subject_substring_match() -> None:
    plan = replace(
        _valid_plan(),
        central_subjects=("subject x",),
    )
    result = validate_grounding_plan(plan, option_ids=("A", "B"))
    assert result.is_valid, result.feedback()


def test_validate_grounding_plan_warnings_do_not_invalidate_plan() -> None:
    warn_only = replace(
        _valid_plan(),
        recommended_skill="unknown_skill",
        central_subjects=("not present in targets",),
        acceptable_evidence_sources=("dream",),
    )

    result = validate_grounding_plan(
        warn_only,
        raw_options={"A": "Alpha then Beta", "B": "Beta then Alpha"},
        skill_ids=("timeline_ordering",),
    )

    assert result.is_valid
    assert result.findings
    assert {finding.severity for finding in result.findings} == {"warning"}


def test_compile_grounding_plan_assigns_stable_registry_ids_and_hash() -> None:
    compiled = compile_grounding_plan(_valid_plan())

    assert compiled.target_key_to_id == {"event_alpha": "T1", "event_beta": "T2"}
    assert compiled.relation_key_to_id == {"alpha_before_beta": "R1"}
    assert compiled.registry.resolve_target_ref("T1").canonical_text == "Event Alpha occurs"
    assert compiled.registry.resolve_target_ref("T1").aliases == ("Alpha event", "Subject X", "Event Alpha")
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
    assert compiled.central_subjects == ("Subject X",)
    assert compiled.acceptable_evidence_sources == ("visual", "asr")
    assert compiled.unresolved_ambiguities == ("needs transcript confirmation",)
    assert compiled.raw_options == {"A": "Alpha then Beta", "B": "Beta then Alpha"}
    assert compiled.registry.option_for("A").raw_option_text == "Alpha then Beta"
    assert compiled.registry.option_for("A").option_kind == "sequence"
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
    assert "central_subjects" in prompt
    assert "option_kind must be one of" in prompt
    assert "Return ONLY one JSON object" in prompt
    assert "Do not wrap it in markdown fences" in prompt
    assert "recommended_skill must be one of" in prompt
    assert "timeline_ordering" in prompt
    assert "route must be one of" in prompt
    assert "claim_kind must be one of" in prompt
    assert "claim_modality must be one of" in prompt
    assert "polarity must be one of" in prompt
    assert "relation.kind must be one of" in prompt
    assert "subjects must be objects" in prompt
    assert '"subject_key"' in prompt
    assert "Use task-specific, option-faithful canonical claims" in prompt
    assert "Use domain-neutral temporary keys only" in prompt
    assert "Use domain-neutral wording in the plan" not in prompt
    assert backend.requests[0].max_new_tokens >= 2400


def test_ground_question_accepts_string_subject_shorthand() -> None:
    payload = _valid_plan().to_dict()
    payload["subjects"] = ["subject"]
    backend = ScriptedGroundingBackend([json.dumps(payload)])

    result = ground_question_with_model(
        backend,
        question="Question: Which event happens first?",
        options=("A. Alpha then Beta", "B. Beta then Alpha"),
    )

    assert result.plan is not None
    assert result.validation.is_valid
    assert result.plan.subjects[0].subject_key == "subject"
    assert result.plan.subjects[0].canonical_name == "subject"


def test_ground_question_accepts_subject_object_map() -> None:
    payload = _valid_plan().to_dict()
    payload["subjects"] = {
        "subject": {
            "canonical_name": "Subject X",
            "aliases": ["X"],
        }
    }
    backend = ScriptedGroundingBackend([json.dumps(payload)])

    result = ground_question_with_model(
        backend,
        question="Question: Which event happens first?",
        options=("A. Alpha then Beta", "B. Beta then Alpha"),
    )

    assert result.plan is not None
    assert result.validation.is_valid
    assert result.plan.subjects[0].subject_key == "subject"
    assert result.plan.subjects[0].canonical_name == "Subject X"


def test_ground_question_canonicalizes_common_grounding_enum_synonyms() -> None:
    payload = _valid_plan().to_dict()
    payload["targets"][0]["claim_kind"] = "event"
    payload["targets"][0]["claim_modality"] = "narrated"
    payload["targets"][0]["polarity"] = "positive"
    payload["targets"][1]["claim_kind"] = "fact"
    payload["targets"][1]["polarity"] = "neutral"
    backend = ScriptedGroundingBackend([json.dumps(payload)])

    result = ground_question_with_model(
        backend,
        question="Question: Which event happens first?",
        options=("A. Alpha then Beta", "B. Beta then Alpha"),
    )

    assert result.plan is not None
    assert result.validation.is_valid
    assert result.plan.targets[0].claim_kind == "visible_event"
    assert result.plan.targets[0].claim_modality == "asr"
    assert result.plan.targets[0].polarity == "affirmed"
    assert result.plan.targets[1].claim_kind == "narrated_fact"
    assert result.plan.targets[1].polarity == "unknown"


def test_ground_question_falls_back_unstructured_after_retry() -> None:
    backend = ScriptedGroundingBackend(
        [
            json.dumps({"route": "temporal_order", "targets": []}),
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
    assert result.attempts == 3


def test_ground_question_reports_parse_failure_after_retry() -> None:
    backend = ScriptedGroundingBackend(["not json", "still not json", "nope"])

    result = ground_question_with_model(
        backend,
        question="Question: Which event happens first?",
        options=("A. Alpha then Beta", "B. Beta then Alpha"),
    )

    assert result.plan is None
    assert not result.validation.is_valid
    assert result.fallback_reason == "grounding_parse_failed"
    assert result.attempts == 3


def test_ground_question_retry_feedback_limits_validation_findings() -> None:
    backend = ScriptedGroundingBackend(
        [
            json.dumps(
                {
                    "route": "bad_route",
                    "recommended_skill": "bad_skill",
                    "central_subjects": [],
                    "subjects": [],
                    "targets": [],
                    "relations": [],
                    "options": [],
                    "acceptable_evidence_sources": ["bad_source"],
                }
            ),
            json.dumps(_valid_plan().to_dict()),
        ]
    )

    result = ground_question_with_model(
        backend,
        question="Question: Which event happens first?",
        options=("A. Alpha then Beta", "B. Beta then Alpha"),
    )

    assert result.plan is not None
    retry_prompt = backend.requests[1].prompt
    feedback = retry_prompt.split("Validation feedback from the previous attempt:", 1)[1]
    assert feedback.count("\n- ") <= 6
    assert "more validation finding(s) omitted" in feedback


def test_ground_question_extracts_final_json_object_from_prose() -> None:
    payload = json.dumps(_valid_plan().to_dict())
    backend = ScriptedGroundingBackend(
        [
            "I will follow the schema {route, targets}.\n```json\n"
            + payload
            + "\n```\nNo answer selected."
        ]
    )

    result = ground_question_with_model(
        backend,
        question="Question: Which event happens first?",
        options=("A. Alpha then Beta", "B. Beta then Alpha"),
    )

    assert result.plan is not None
    assert result.validation.is_valid
    assert result.attempts == 1
