import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.contracts import ClaimRelation, OptionSpec, TargetRegistry, TargetSpec
from visual_coding_agent_harness.legacy.interpreter import ProgramInterpreter
from visual_coding_agent_harness.tools.navigation import build_video_navigation_registry
from visual_coding_agent_harness.video.map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class OrderedListExtractionTest(unittest.TestCase):
    def test_ordered_list_uses_transcript_span_order_not_input_target_order(self):
        video_map = VideoMap(
            video_path="/videos/sequence.mp4",
            duration_sec=120.0,
            segments=[
                VideoMapSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=120.0,
                    asr_sentences=[
                        {
                            "start_sec": 20.0,
                            "end_sec": 28.0,
                            "text": 'The chapter lists "Alpha", "Beta", and "Gamma" in order.',
                        }
                    ],
                )
            ],
        )
        registry = build_video_navigation_registry(video_map)

        located = registry.execute(
            "locate_targets_in_segment",
            {"segment_id": "seg_0001", "targets": ["Gamma", "Alpha", "Beta"]},
        )

        ordered = _ordered_candidate(located)
        self.assertEqual(ordered["ordered_targets"], ["Alpha", "Beta", "Gamma"])
        self.assertEqual(
            [hit["target"] for hit in ordered["ordered_target_hits"]],
            ["Alpha", "Beta", "Gamma"],
        )
        self.assertEqual(
            [hit["source_span_start"] for hit in ordered["ordered_target_hits"]],
            sorted(hit["source_span_start"] for hit in ordered["ordered_target_hits"]),
        )

    def test_forward_reference_not_merged_into_later_enumeration(self):
        video_map = VideoMap(
            video_path="/videos/sequence.mp4",
            duration_sec=120.0,
            segments=[
                VideoMapSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=120.0,
                    asr_sentences=[
                        {
                            "start_sec": 10.0,
                            "end_sec": 15.0,
                            "text": 'The narrator says we will see "Gamma" later.',
                        },
                        {
                            "start_sec": 20.0,
                            "end_sec": 28.0,
                            "text": 'The chapter lists "Alpha" and "Beta" in order.',
                        },
                    ],
                )
            ],
        )
        registry = build_video_navigation_registry(video_map)

        located = registry.execute(
            "locate_targets_in_segment",
            {"segment_id": "seg_0001", "targets": ["Alpha", "Beta", "Gamma"]},
        )

        gamma_hits = [
            candidate
            for candidate in located["candidates"]
            if candidate.get("target") == "Gamma" and candidate.get("match_type") != "ordered_list_mention"
        ]
        self.assertTrue(gamma_hits)
        self.assertTrue(all(hit.get("forward_reference") for hit in gamma_hits))
        ordered = _ordered_candidate(located)
        self.assertEqual(ordered["ordered_targets"], ["Alpha", "Beta"])
        self.assertEqual(ordered["route_kind"], "partial_ordered_list")
        self.assertNotIn("ordered_list_transcript_complete", [action.get("route_kind") for action in located["recommended_next_actions"]])

    def test_no_synthetic_observed_timestamps_for_text_spans(self):
        video_map = VideoMap(
            video_path="/videos/sequence.mp4",
            duration_sec=120.0,
            segments=[
                VideoMapSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=120.0,
                    asr_sentences=[
                        {
                            "start_sec": 30.0,
                            "end_sec": 42.0,
                            "text": 'The chapter lists "Alpha", "Beta", and "Gamma" in order.',
                        }
                    ],
                )
            ],
        )
        registry = build_video_navigation_registry(video_map)

        located = registry.execute(
            "locate_targets_in_segment",
            {"segment_id": "seg_0001", "targets": ["Alpha", "Beta", "Gamma"]},
        )

        self.assertEqual(located["ordered_list_timeline_rows"], [])
        ordered = _ordered_candidate(located)
        self.assertEqual(ordered["text_span_window"], [30.0, 42.0])
        self.assertTrue(all(hit.get("timestamp_start") is None for hit in ordered["ordered_target_hits"]))
        self.assertTrue(all(hit.get("timestamp_end") is None for hit in ordered["ordered_target_hits"]))

    def test_partial_ordered_list_not_promoted_as_supported_relation(self):
        video_map = VideoMap(
            video_path="/videos/sequence.mp4",
            duration_sec=120.0,
            segments=[
                VideoMapSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=120.0,
                    asr_text='The chapter lists "Alpha" and "Beta" in order.',
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="partial_ordered_list")
            workspace.target_registry = TargetRegistry.from_specs(
                targets=[TargetSpec("T1", "Alpha"), TargetSpec("T2", "Beta"), TargetSpec("T3", "Gamma")],
                options=[OptionSpec("A", target_sequence=("T1", "T2", "T3"), required_relations=("R1", "R2"))],
                relations=[ClaimRelation("R1", "before", "T1", "T2"), ClaimRelation("R2", "before", "T2", "T3")],
            )
            registry = build_video_navigation_registry(video_map, workspace=workspace)
            located = registry.execute(
                "locate_targets_in_segment",
                {"segment_id": "seg_0001", "target_refs": ["T1", "T2", "T3"]},
            )
            detail = registry.execute(
                "read_segment_detail",
                {"segment_id": "seg_0001", "target_refs": ["T1", "T2", "T3"], "promote_answer_evidence": True},
            )

        ordered = _ordered_candidate(located)
        self.assertEqual(ordered["route_kind"], "partial_ordered_list")
        self.assertEqual(located["answer_evidence_rows"], [])
        self.assertFalse(
            any(binding.get("status") == "supported" for binding in detail["relation_bindings"])
        )


def _ordered_candidate(payload):
    ordered = [
        candidate
        for candidate in payload["candidates"]
        if candidate.get("match_type") == "ordered_list_mention"
    ]
    if len(ordered) != 1:
        raise AssertionError(f"expected one ordered candidate, found {len(ordered)}")
    return ordered[0]
