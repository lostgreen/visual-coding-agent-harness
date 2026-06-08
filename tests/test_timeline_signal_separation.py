from pathlib import Path

from visual_coding_agent_harness.agents.iterative_agent import _confirmed_timeline_rows
from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.tools.navigation import build_video_navigation_registry
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_locate_ordered_list_rows_are_text_candidates_not_confirmed_timeline(tmp_path: Path):
    video_map = VideoMap(
        video_path="/videos/bernini.mp4",
        duration_sec=600.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0002",
                start_sec=300.0,
                end_sec=600.0,
                asr_sentences=[
                    {
                        "start_sec": 530.0,
                        "end_sec": 546.0,
                        "text": (
                            '"Aeneas, Anchises, and Ascanius fleeing Troy", '
                            '"David", "The rape of Persephone", and "Apollo and Daphne".'
                        ),
                    }
                ],
            )
        ],
    )
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_signal_locate")
    registry = build_video_navigation_registry(video_map, workspace=workspace)

    ProgramInterpreter(registry=registry, workspace=workspace).run(
        [
            {
                "tool": "locate_targets_in_segment",
                "args": {
                    "segment_id": "seg_0002",
                    "targets": [
                        "Aeneas, Anchises, and Ascanius fleeing Troy",
                        "David",
                        "The rape of Persephone",
                        "Apollo and Daphne",
                    ],
                },
            }
        ]
    )

    observation = workspace.read_observations(tool_name="locate_targets_in_segment")[0]
    rows = observation.raw_output["ordered_list_timeline_rows"]
    assert rows
    assert {row["confidence_signal"] for row in rows} == {"text_inferred"}
    assert all(row["requires_visual_verification"] for row in rows)
    assert workspace.read_timeline_sorted() == []
    assert _confirmed_timeline_rows(workspace.read_timeline_sorted()) == []
    candidate_path = workspace.root / "timeline_candidates.md"
    assert candidate_path.exists()
    assert "text_inferred" in candidate_path.read_text(encoding="utf-8")


def test_visual_verifier_rows_are_confirmed_timeline(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_signal_verify")
    observation = workspace.write_observation(
        tool_name="verify_segment_anchors",
        claim="David is visually confirmed at 380 seconds.",
        confidence=0.92,
        raw_output={
            "timeline_rows": [
                {
                    "entity": "David",
                    "observed_at_sec": 380.0,
                    "window": [370.0, 390.0],
                    "confidence_signal": "visually_confirmed",
                    "claim": "David is visually confirmed.",
                }
            ]
        },
    )

    workspace.append_timeline_from_observation(observation)

    timeline = workspace.read_timeline_sorted()
    assert len(timeline) == 1
    assert timeline[0]["confidence_signal"] == "visually_confirmed"
    assert [row["entity"] for row in _confirmed_timeline_rows(timeline)] == ["David"]
