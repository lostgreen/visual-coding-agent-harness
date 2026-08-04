from __future__ import annotations

from vcah.interactive_agents import WorkspaceReasoner, _frozen_reasoner_prompt
from vcah.phase5 import (
    Phase5Protocol,
    blind_prior_prompt,
    inspection_mode_policy_errors,
)
from vcah.runtime_metrics import agent_run_metrics
from vcah.workspace import ObservationLog, WorkingDocument, render_frozen_working_view


def test_blind_arm_receives_no_visual_or_caption_tools() -> None:
    protocol = Phase5Protocol(
        controller_mode="frozen_baseline",
        controller_evidence_visibility="none",
        measurement_control="blind_prior",
    )

    assert protocol.arm == "blind_prior"
    assert protocol.allowed_inspection_modes == frozenset()
    prompt = blind_prior_prompt("What changes?")
    assert prompt.count("What changes?") == 1
    assert "Options:" not in prompt
    assert "Working view" not in prompt


def test_caption_only_arm_rejects_visual_tasks() -> None:
    protocol = Phase5Protocol(
        controller_mode="frozen_baseline",
        controller_evidence_visibility="full",
        measurement_control="caption_only",
    )
    tasks = (
        {"inspection_mode": "search_caption"},
        {"inspection_mode": "window"},
    )

    errors = inspection_mode_policy_errors(
        tasks,
        allowed_modes=protocol.allowed_inspection_modes,
    )

    assert [row["inspection_mode"] for row in errors] == ["window"]


def test_frozen_prompt_excludes_mger_control_plane() -> None:
    prompt = _frozen_reasoner_prompt(
        {
            "question": "What changes?",
            "options": {},
            "mechanical_status": {},
            "workspace_overview": {},
        }
    )

    assert "obligation" not in prompt.casefold()
    assert "closure" not in prompt.casefold()
    assert "supporting_claim_ids" in prompt


def test_frozen_working_view_excludes_mger_state(tmp_path) -> None:
    document = WorkingDocument.with_question_premise("What changes?")
    observations = ObservationLog(tmp_path / "observations.jsonl")

    view = render_frozen_working_view(document, observations)

    assert "EVIDENCE OBLIGATIONS" not in view
    assert "TEMPORAL SCOPES" not in view
    assert "OBSERVATION CUE STATES" not in view
    assert "OBSERVATION CATALOG" in view


def test_frozen_reasoner_retains_historical_single_json_repair(tmp_path) -> None:
    class FakeAPI:
        model = "fake"
        last_response_metadata = {}

        def __init__(self) -> None:
            self.outputs = iter(("not json", '{"action":"answer","answer":"fact"}'))

        def chat(self, prompt, *, max_tokens):
            return next(self.outputs)

    reasoner = WorkspaceReasoner(
        FakeAPI(),
        trace_path=tmp_path / "trace.jsonl",
        controller_mode="frozen_baseline",
    )

    decision = reasoner.decide(question="What changes?", options={})
    metadata = reasoner.consume_decision_metadata()

    assert decision.action == "answer"
    assert decision.answer == "fact"
    assert metadata["internal_control_retry_count"] == 1
    assert metadata["format_repaired"] is True


def test_observed_case_and_malformed_decision_metrics() -> None:
    trace = (
        {"type": "reasoner_decision_attempt", "schema_valid": False},
        {"type": "reasoner_decision_attempt", "schema_valid": True},
        {"type": "reasoner_decision", "semantic_committed": True, "round": 1},
    )
    observations = (
        {
            "attempt_id": "attempt_1",
            "modality": "visual",
            "frame_times": [1.0, 2.0, 3.0],
            "sampling_config": {},
            "interpretation_items": [],
        },
    )

    metrics = agent_run_metrics(
        trace,
        observations,
        answer_present=True,
        reference_valid=False,
    )

    assert metrics["observed_case_rate"] == 1.0
    assert metrics["conditional_visual_frames"] == 3
    assert metrics["malformed_decision_rate"] == 0.5


def test_conditional_frames_are_none_without_visual_observation() -> None:
    metrics = agent_run_metrics(
        (),
        (),
        answer_present=True,
        reference_valid=False,
    )

    assert metrics["observed_case_rate"] == 0.0
    assert metrics["conditional_visual_frames"] is None
