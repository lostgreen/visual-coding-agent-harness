from __future__ import annotations

from pathlib import Path

from vcah.evidence_state import InterpretationItem
from vcah.investigator import ObservationAttempt
from vcah.workspace import ObservationLog, WorkingDocument, stable_attempt_id


def _logged_observation(tmp_path: Path) -> tuple[ObservationLog, dict[str, object]]:
    attempt_id = stable_attempt_id(
        source_video_ids=("video-a",),
        frame_times=(5.0,),
        sampling_fps=2.0,
        modality="visual",
    )
    log = ObservationLog(tmp_path / "observation_log.jsonl")
    row = log.append_attempt(
        ObservationAttempt(
            attempt_id=attempt_id,
            task_id="inspect",
            requested_range=(5.0, 5.0),
            inspected_ranges=((5.0, 5.0),),
            attached_frame_times=(5.0,),
            sampling_config={"fps": 2.0, "modality": "visual"},
            images_requested=1,
            images_attached=1,
            parse_status="parsed",
            frame_refs=("frame-5.jpg",),
            raw_output='{"summary":"A blue button is visible."}',
            source_video_ids=("video-a",),
            interpretation_items=(
                InterpretationItem(
                    item_id="item_blue_button",
                    time_anchor=(5.0, 5.0),
                    text="A blue button is visible.",
                    item_kind="ui_description",
                ),
            ),
        ),
        round_id=1,
    )
    return log, dict(row)


def test_observation_claim_requires_exact_interpretation_item_triple(tmp_path: Path) -> None:
    log, row = _logged_observation(tmp_path)
    document = WorkingDocument.with_question_premise("What color is the button?")

    rejected = document.apply_ops(
        (
            {
                "op": "add_claim",
                "claim_id": "claim_missing_item",
                "text": "The button is blue.",
                "source": "observation",
                "cites": [row["attempt_id"]],
                "time_anchor": [5.0, 5.0],
            },
        ),
        observation_ids=log.attempt_ids,
        observation_rows=log.rows,
        require_item_provenance=True,
    )
    assert not rejected.accepted
    assert any("observation_claim_requires_interpretation_id" in error for error in rejected.errors)

    accepted = document.apply_ops(
        (
            {
                "op": "add_claim",
                "claim_id": "claim_blue",
                "text": "The button is blue.",
                "source": "observation",
                "cites": [row["attempt_id"]],
                "interpretation_id": row["interpretation_id"],
                "interpretation_item_id": "item_blue_button",
                "time_anchor": [5.0, 5.0],
                "confidence": "high",
            },
        ),
        observation_ids=log.attempt_ids,
        observation_rows=log.rows,
        require_item_provenance=True,
    )
    assert accepted.accepted
    validation = document.validate_answer(
        ("claim_blue",),
        observation_ids=log.attempt_ids,
        observation_rows=log.rows,
        require_item_provenance=True,
    )
    assert validation.passed
    assert document.provenance_summary(log.rows)["observation_claim_item_binding_rate"] == 1.0


def test_claim_item_time_anchor_must_overlap_bound_item(tmp_path: Path) -> None:
    log, row = _logged_observation(tmp_path)
    document = WorkingDocument.with_question_premise("What color is the button?")
    result = document.apply_ops(
        (
            {
                "op": "add_claim",
                "claim_id": "claim_shifted",
                "text": "The button is blue.",
                "source": "observation",
                "cites": [row["attempt_id"]],
                "interpretation_id": row["interpretation_id"],
                "interpretation_item_id": "item_blue_button",
                "time_anchor": [9.0, 9.0],
            },
        ),
        observation_ids=log.attempt_ids,
        observation_rows=log.rows,
        require_item_provenance=True,
    )
    assert not result.accepted
    assert any("observation_claim_time_anchor_mismatch" in error for error in result.errors)
