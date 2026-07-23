from __future__ import annotations

import json
from pathlib import Path

import pytest

from vcah.investigator import ObservationAttempt
from vcah.workspace import ObservationLog, WorkingDocument, render_working_view, stable_attempt_id


def test_attempt_identity_is_prompt_independent_but_material_sensitive() -> None:
    common = {
        "source_video_ids": ("video-a",),
        "frame_times": (1.0, 2.0, 3.0),
        "sampling_fps": 1.0,
        "modality": "visual",
    }

    first = stable_attempt_id(**common)
    second = stable_attempt_id(**common)
    denser = stable_attempt_id(**{**common, "sampling_fps": 2.0})
    shifted = stable_attempt_id(**{**common, "frame_times": (1.0, 2.0, 3.5)})

    assert first == second
    assert first != denser
    assert first != shifted


def test_observation_log_groups_interpretations_without_truncating_raw_output(tmp_path: Path) -> None:
    attempt_id = stable_attempt_id(
        source_video_ids=("video-a",),
        frame_times=(1.0, 2.0),
        sampling_fps=1.0,
    )
    raw_first = '{"summary":"' + ("visible detail " * 500) + '"}'
    log = ObservationLog(tmp_path / "observation_log.jsonl")
    common = {
        "attempt_id": attempt_id,
        "task_id": "inspect-1",
        "requested_range": (1.0, 2.0),
        "inspected_ranges": ((1.0, 2.0),),
        "attached_frame_times": (1.0, 2.0),
        "sampling_config": {"fps": 1.0},
        "images_requested": 2,
        "images_attached": 2,
        "frame_refs": ("frame-1.jpg", "frame-2.jpg"),
        "source_video_ids": ("video-a",),
    }
    log.append_attempt(
        ObservationAttempt(**common, prompt_digest="prompt-one", raw_output=raw_first),
        round_id=1,
    )
    log.append_attempt(
        ObservationAttempt(
            **common,
            prompt_digest="prompt-two",
            raw_output='{"summary":"a second interpretation"}',
        ),
        round_id=2,
    )

    assert log.attempt_ids == (attempt_id,)
    assert log.catalog()[0]["interpretation_count"] == 2
    assert [row["raw_output"] for row in log.read(attempt_ids=(attempt_id,))] == [
        raw_first,
        '{"summary":"a second interpretation"}',
    ]
    persisted = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    assert len(persisted) == 2
    assert persisted[0]["raw_output"] == raw_first


def test_observation_log_rejects_attempt_id_that_does_not_match_material(tmp_path: Path) -> None:
    log = ObservationLog(tmp_path / "observation_log.jsonl")

    with pytest.raises(ValueError, match="does not match inspected material"):
        log.append_attempt(
            ObservationAttempt(
                attempt_id="attempt_forged",
                inspected_ranges=((1.0, 2.0),),
                sampling_config={"fps": 1.0},
            ),
            round_id=1,
        )


def test_observation_log_creation_never_overwrites_existing_content(tmp_path: Path) -> None:
    path = tmp_path / "observation_log.jsonl"
    path.write_text("immutable-row\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        ObservationLog(path)

    assert path.read_text(encoding="utf-8") == "immutable-row\n"


def test_working_document_applies_ops_transactionally_and_validates_answer_references() -> None:
    attempt_a = "attempt_a"
    attempt_b = "attempt_b"
    document = WorkingDocument.with_question_premise("Which option is supported?")

    rejected = document.apply_ops(
        (
            {
                "op": "add_claim",
                "claim_id": "claim_bad",
                "text": "An unsupported observation claim.",
                "source": "observation",
                "cites": ("attempt_unknown",),
            },
        ),
        observation_ids=(attempt_a, attempt_b),
    )

    assert not rejected.accepted
    assert document.revision == 0
    assert "claim_bad" not in document.claims

    forged_premise = document.apply_ops(
        (
            {
                "op": "add_claim",
                "claim_id": "claim_forged_premise",
                "text": "A model-generated statement presented as question text.",
                "source": "premise",
            },
        ),
        observation_ids=(attempt_a, attempt_b),
    )
    assert not forged_premise.accepted
    assert forged_premise.errors == ("op[1]: premise_source_is_framework_managed",)
    assert set(document.claims) == {"premise:question"}

    premise_edit = document.apply_ops(
        ({"op": "set_status", "claim_id": "premise:question", "status": "retracted"},),
        observation_ids=(attempt_a, attempt_b),
    )
    assert not premise_edit.accepted
    assert premise_edit.errors == ("op[1]: premise_is_framework_managed",)

    accepted = document.apply_ops(
        (
            {
                "op": "add_claim",
                "claim_id": "claim_visible_a",
                "text": "The first observation shows the subject enter.",
                "source": "observation",
                "cites": (attempt_a,),
                "confidence": "high",
            },
            {
                "op": "add_claim",
                "claim_id": "claim_visible_b",
                "text": "The second observation shows the subject leave.",
                "source": "observation",
                "cites": (attempt_b,),
            },
            {
                "op": "add_claim",
                "claim_id": "claim_sequence",
                "text": "The subject enters before leaving.",
                "source": "derived",
                "derived_from": ("claim_visible_a", "claim_visible_b"),
            },
            {
                "op": "link_conflict",
                "claim_id": "claim_visible_a",
                "other_claim_id": "claim_visible_b",
            },
            {
                "op": "note_interval",
                "time_range": (1.0, 4.0),
                "label": "entry and exit",
                "claim_ids": ("claim_sequence",),
            },
            {
                "op": "update_entity",
                "entity_id": "subject_1",
                "description": "person in a red coat",
                "aliases": ("the subject",),
            },
        ),
        observation_ids=(attempt_a, attempt_b),
    )

    assert accepted.accepted
    assert accepted.applied_count == 6
    assert document.revision == 1
    assert document.claims["claim_visible_b"].claim_id in document.claims["claim_visible_a"].conflicts_with
    validation = document.validate_answer(
        ("claim_sequence",),
        observation_ids=(attempt_a, attempt_b),
    )
    assert validation.passed
    assert validation.cited_attempt_ids == (attempt_a, attempt_b)

    redundant_status_path = document.apply_ops(
        ({"op": "set_status", "claim_id": "claim_visible_a", "status": "superseded"},),
        observation_ids=(attempt_a, attempt_b),
    )
    assert not redundant_status_path.accepted
    assert redundant_status_path.errors == ("op[1]: invalid_claim_status:superseded",)


@pytest.mark.parametrize(
    ("source", "status", "confidence", "expected_reason"),
    (
        ("hypothesis", "active", "high", "supporting_claim_hypothetical"),
        ("observation", "contested", "high", "supporting_claim_uncertain"),
        ("observation", "active", "low", "supporting_claim_uncertain"),
    ),
)
def test_answer_validation_rejects_non_direct_support(
    source: str,
    status: str,
    confidence: str,
    expected_reason: str,
) -> None:
    document = WorkingDocument.with_question_premise("Which option is supported?")
    result = document.apply_ops(
        (
            {
                "op": "add_claim",
                "claim_id": "weak_support",
                "text": "The option might be true.",
                "source": source,
                "status": status,
                "confidence": confidence,
                "cites": ("attempt_a",),
            },
        ),
        observation_ids=("attempt_a",),
    )
    assert result.accepted

    validation = document.validate_answer(("weak_support",), observation_ids=("attempt_a",))

    assert not validation.passed
    assert validation.reason == expected_reason


def test_working_view_exposes_mechanical_coverage_and_requested_raw_observations(tmp_path: Path) -> None:
    attempt_id = stable_attempt_id(
        frame_times=(5.0, 6.0, 7.0),
        sampling_fps=1.0,
    )
    log = ObservationLog(tmp_path / "observation_log.jsonl")
    log.append_attempt(
        ObservationAttempt(
            attempt_id=attempt_id,
            task_id="inspect-visible",
            requested_range=(5.0, 7.0),
            inspected_ranges=((5.0, 7.0),),
            attached_frame_times=(5.0, 6.0, 7.0),
            sampling_config={"fps": 1.0},
            frame_refs=("f1", "f2", "f3"),
            raw_output='{"summary":"subject raises a cup"}',
        ),
        round_id=1,
    )
    document = WorkingDocument.with_question_premise("What does the subject do?")
    requested = log.read(attempt_ids=(attempt_id,))

    view = render_working_view(document, log, requested_observations=requested)

    assert "COVERAGE LEDGER" in view
    assert "OBSERVATION CATALOG" in view
    assert "REQUESTED OBSERVATION PREVIEWS" in view
    assert attempt_id in view
    assert "subject raises a cup" in view


def test_working_view_bounds_requested_observation_and_keeps_pointer(tmp_path: Path) -> None:
    attempt_id = stable_attempt_id(frame_times=(5.0,), sampling_fps=1.0)
    raw_output = ("direct observation " * 1800) + "RAW_END_SENTINEL"
    log = ObservationLog(tmp_path / "observation_log.jsonl")
    log.append_attempt(
        ObservationAttempt(
            attempt_id=attempt_id,
            inspected_ranges=((5.0, 5.1),),
            attached_frame_times=(5.0,),
            sampling_config={"fps": 1.0},
            frame_refs=("f1",),
            raw_output=raw_output,
        ),
        round_id=1,
    )

    view = render_working_view(
        WorkingDocument.with_question_premise("What is visible?"),
        log,
        requested_observations=log.read(attempt_ids=(attempt_id,)),
    )

    assert "RAW_END_SENTINEL" not in view
    assert "REQUESTED OBSERVATION PREVIEWS" in view
    assert f"raw_pointer={log.path}" in view


def test_claim_entities_and_interval_roles_round_trip(tmp_path: Path) -> None:
    attempt_id = stable_attempt_id(frame_times=(5.0,), sampling_fps=1.0)
    document = WorkingDocument.with_question_premise("Who appears and where?")
    result = document.apply_ops(
        (
            {
                "op": "update_entity",
                "entity_id": "person_1",
                "description": "The player character",
                "aliases": ["player"],
            },
            {
                "op": "add_claim",
                "claim_id": "located_player",
                "text": "The player stands by the shrine.",
                "source": "observation",
                "cites": [attempt_id],
                "entity_ids": ["person_1"],
                "metadata": {"located_by": {"passage_id": "caption:p1"}},
            },
            {
                "op": "note_interval",
                "start_sec": 5.0,
                "end_sec": 7.0,
                "label": "counted_event",
                "claim_ids": ["located_player"],
                "role": "candidate",
                "metadata": {"event_key": "shrine_visit", "status": "candidate"},
            },
        ),
        observation_ids=(attempt_id,),
    )
    path = tmp_path / "working_document.json"
    document.save(path)
    restored = WorkingDocument.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    assert result.accepted
    assert restored.claims["located_player"].entity_ids == ("person_1",)
    assert restored.claims["located_player"].metadata["located_by"]["passage_id"] == "caption:p1"
    assert restored.timeline[0].role == "candidate"
    assert restored.timeline[0].metadata["event_key"] == "shrine_visit"


def test_candidate_search_attempt_cannot_directly_support_answer() -> None:
    attempt_id = stable_attempt_id(
        source_video_ids=("video-a",),
        frame_refs=("caption-search://candidate",),
        modality="caption_search",
    )
    document = WorkingDocument.with_question_premise("What item appears?")
    applied = document.apply_ops(
        (
            {
                "op": "add_claim",
                "claim_id": "caption_guess",
                "text": "The caption suggests an item.",
                "source": "observation",
                "cites": [attempt_id],
            },
        ),
        observation_ids=(attempt_id,),
    )

    validation = document.validate_answer(
        ("caption_guess",),
        observation_ids=(attempt_id,),
        supporting_observation_ids=(),
    )

    assert applied.accepted
    assert not validation.passed
    assert validation.reason == "supporting_claims_cite_candidate_or_negative"
