from __future__ import annotations

from pathlib import Path

from vcah.multiround import _apply_episode_binding_completion
from vcah.types import EvidenceRecord
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    manifest = VirtualVideoManifest(
        workspace_id="episode-chain",
        segments=(
            VirtualVideoSegment("seg_1", "source", "source.mp4", 0.0, 5.0, 0.0, 5.0),
            VirtualVideoSegment("seg_2", "source", "source.mp4", 5.0, 10.0, 5.0, 10.0),
        ),
    )
    case = VirtualVideoCase(
        case_id="episode-chain",
        question="In the last instance, what color clothes did the second person who ultimately overtook the recorder wear?",
        options={"A": "red clothes", "B": "green clothes"},
        gold="B",
        target_segment_id="seg_1",
        target_virtual_interval=(0.0, 10.0),
    )
    return VirtualVideoWorkspace(
        workspace_id=manifest.workspace_id,
        root_dir=tmp_path,
        manifest=manifest,
        case=case,
        frame_manifest=tmp_path / "frame_manifest.jsonl",
        asr_virtual_cues=tmp_path / "asr_virtual_cues.json",
        cold_index_dir=tmp_path / "cold_index",
    )


def _event_record(evidence_id: str, start: float, end: float, event_key: str, participants: list[dict[str, str]]) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        beat_id="",
        start_sec=start,
        end_sec=end,
        modality="visual",
        pointer=f"virtual://{evidence_id}",
        verbatim="An overtaking episode is visible.",
        frame_refs=(f"{evidence_id}.jpg",),
        source_lineage=({"segment_id": "seg_1" if start < 5 else "seg_2", "source_video_id": "source"},),
        operation_metadata={
            "structured_parse_status": "parsed",
            "events": [{
                "event_key": event_key,
                "event_class": "scene",
                "counting_unit": "episode",
                "participants": participants,
                "participant_ids": ["recorder", *(row["participant_id"] for row in participants)],
                "start_sec": start,
                "end_sec": end,
                "supports_question_event": True,
            }],
        },
    )


def _association_record(source_event_key: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="ev_finish",
        beat_id="",
        start_sec=8.0,
        end_sec=9.0,
        modality="visual",
        pointer="virtual://finish",
        verbatim="The identified participant wears green clothing at the finish.",
        frame_refs=("finish.jpg",),
        source_lineage=({"segment_id": "seg_2", "source_video_id": "source"},),
        operation_metadata={
            "structured_parse_status": "parsed",
            "entities": [{
                "entity_observation_id": "finish:racer_two",
                "local_id": "racer_two",
                "entity_hypothesis_id": "racer_two",
                "role": "finish_participant",
                "association_confidence": 0.9,
                "countable": True,
                "attributes": {"clothing_color": "green"},
            }],
            "entity_associations": [{
                "association_id": "assoc_racer_two",
                "source_participant_id": "racer_two",
                "source_event_key": source_event_key,
                "source_episode_id": source_event_key,
                "source_event_role": "overtaker",
                "target_entity_observation_id": "finish:racer_two",
                "entity_hypothesis_id": "racer_two",
                "status": "supported",
                "confidence": 0.9,
                "shared_attributes": {"helmet": "black", "jacket": "green"},
                "distinguishing_attributes": {"clothing_color": "green"},
            }],
        },
    )


def test_episode_binding_requires_last_episode_then_episode_local_ordinal_and_attribute(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    evidence = (
        _event_record("ev_first", 1.0, 2.0, "episode-one", [{"participant_id": "racer_one", "role": "overtaker"}]),
        _event_record(
            "ev_last",
            6.0,
            7.0,
            "episode-two",
            [
                {"participant_id": "racer_one", "role": "overtaker"},
                {"participant_id": "racer_two", "role": "overtaker"},
            ],
        ),
        _association_record("episode-two"),
    )

    completion = _apply_episode_binding_completion(
        {
            "range_coverage_complete": True,
            "enumeration_required": True,
            "enumeration_complete": True,
            "ready_for_answer": True,
        },
        workspace,
        evidence,
        {"requires_event_participant_link": True, "requires_temporal_sequence": True},
    )

    assert completion["temporal_max_selection"]["status"] == "resolved"
    assert completion["selected_last_episode_id"] == "event_candidate_002"
    assert completion["target_participant_selection"]["participant_id"] == "racer_two"
    assert completion["target_entity_binding"]["entity_id"] == "racer_two"
    assert completion["target_attribute_facts"][0]["attribute_value"] == "green"


def test_episode_binding_rejects_association_from_another_episode(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    evidence = (
        _event_record("ev_first", 1.0, 2.0, "episode-one", [{"participant_id": "racer_one", "role": "overtaker"}]),
        _event_record(
            "ev_last",
            6.0,
            7.0,
            "episode-two",
            [
                {"participant_id": "racer_one", "role": "overtaker"},
                {"participant_id": "racer_two", "role": "overtaker"},
            ],
        ),
        _association_record("episode-one"),
    )

    completion = _apply_episode_binding_completion(
        {
            "range_coverage_complete": True,
            "enumeration_required": True,
            "enumeration_complete": True,
            "ready_for_answer": True,
        },
        workspace,
        evidence,
        {"requires_event_participant_link": True, "requires_temporal_sequence": True},
    )

    assert completion["temporal_max_selection"]["status"] == "resolved"
    assert completion["event_participant_link_ready"] is False
    assert completion["target_entity_binding"]["status"] == "incomplete"
