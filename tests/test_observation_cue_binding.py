from __future__ import annotations

from pathlib import Path

from vcah.evidence_state import InterpretationItem
from vcah.investigator import ObservationAttempt
from vcah.workspace import ObservationLog, WorkingDocument, stable_attempt_id


def _attempt(
    *,
    fps: float,
    items: tuple[InterpretationItem, ...],
) -> ObservationAttempt:
    attempt_id = stable_attempt_id(
        source_video_ids=("video-a",),
        frame_times=(5.0,),
        sampling_fps=fps,
        modality="visual",
    )
    return ObservationAttempt(
        attempt_id=attempt_id,
        requested_range=(4.0, 6.0),
        inspected_ranges=((5.0, 5.0),),
        attached_frame_times=(5.0,),
        sampling_config={"fps": fps, "modality": "visual"},
        images_requested=1,
        images_attached=1,
        parse_status="parsed",
        frame_refs=(f"frame-{fps}.jpg",),
        raw_output='{"summary":"Visible frame."}',
        source_video_ids=("video-a",),
        interpretation_items=items,
    )


def test_only_exact_sampled_point_item_becomes_observation_cue(tmp_path: Path) -> None:
    log = ObservationLog(tmp_path / "observation_log.jsonl")
    row = log.append_attempt(
        _attempt(
            fps=1.0,
            items=(
                InterpretationItem("item_exact", (5.0, 5.0), "Exact sampled moment."),
                InterpretationItem("item_invented", (5.123, 5.123), "Invented precision."),
                InterpretationItem("item_interval", (4.5, 5.5), "A temporal interval."),
            ),
        ),
        round_id=1,
    )

    assert len(row["observation_cues"]) == 1
    cue = row["observation_cues"][0]
    assert cue["item_id"] == "item_exact"
    assert cue["virtual_time"] == 5.0
    assert cue["source_frame_ref"] == "frame-1.0.jpg"


def test_cue_status_requires_a_real_verification_item(tmp_path: Path) -> None:
    log = ObservationLog(tmp_path / "observation_log.jsonl")
    parent = log.append_attempt(
        _attempt(
            fps=1.0,
            items=(InterpretationItem("item_parent", (5.0, 5.0), "Candidate cue."),),
        ),
        round_id=1,
    )
    verification = log.append_attempt(
        _attempt(
            fps=2.0,
            items=(InterpretationItem("item_verify", (5.0, 5.0), "The cue is confirmed."),),
        ),
        round_id=2,
    )
    cue_id = parent["observation_cues"][0]["cue_id"]
    document = WorkingDocument.with_question_premise("What happens?")

    accepted = document.apply_ops(
        (
            {
                "op": "set_cue_status",
                "cue_id": cue_id,
                "status": "verified",
                "verification_attempt_id": verification["attempt_id"],
                "verification_interpretation_id": verification["interpretation_id"],
                "verification_item_id": "item_verify",
            },
        ),
        observation_ids=log.attempt_ids,
        observation_rows=log.rows,
    )
    assert accepted.accepted
    assert document.cue_summary(log.rows)["verified_cue_count"] == 1

    rejected = WorkingDocument.with_question_premise("What happens?").apply_ops(
        (
            {
                "op": "set_cue_status",
                "cue_id": cue_id,
                "status": "verified",
                "verification_attempt_id": verification["attempt_id"],
                "verification_interpretation_id": verification["interpretation_id"],
                "verification_item_id": "item_missing",
            },
        ),
        observation_ids=log.attempt_ids,
        observation_rows=log.rows,
    )
    assert not rejected.accepted
    assert any("cue_verification_item_unknown" in error for error in rejected.errors)
