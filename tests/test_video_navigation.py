import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.tools.exploration import build_video_exploration_registry
from visual_coding_agent_harness.tools.navigation import build_video_navigation_registry
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment, VideoMapStore
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class NavigationBackend(VisionLanguageBackend):
    def __init__(self):
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=f"{request.task} observation")


def demo_video_map() -> VideoMap:
    return VideoMap(
        video_path="/videos/demo.mp4",
        duration_sec=120.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=40.0,
                low_fps_caption="A person introduces the trip.",
                asr_text="Welcome to the travel diary.",
                entities=["person", "mountain"],
            ),
            VideoMapSegment(
                segment_id="seg_0002",
                start_sec=40.0,
                end_sec=80.0,
                low_fps_caption="A close view of a blue aircraft in a museum.",
                ocr_text="AVIATION HISTORY",
                entities=["aircraft", "museum"],
            ),
            VideoMapSegment(
                segment_id="seg_0003",
                start_sec=80.0,
                end_sec=120.0,
                low_fps_caption="Closing credits and crowd shots.",
                asr_text="Thanks for watching.",
            ),
        ],
    )


class VideoNavigationTest(unittest.TestCase):
    def test_video_map_searches_caption_asr_ocr_and_entities(self):
        video_map = demo_video_map()

        results = video_map.search("aviation aircraft", top_k=2)

        self.assertEqual(results[0].segment.segment_id, "seg_0002")
        self.assertGreater(results[0].score, 0)
        self.assertIn("low_fps_caption", results[0].matched_fields)
        self.assertIn("ocr_text", results[0].matched_fields)
        self.assertIn("entities", results[0].matched_fields)

    def test_navigation_registry_exposes_video_workspace_tools(self):
        registry = build_video_navigation_registry(demo_video_map())

        listing = registry.execute("video_ls", {})
        search = registry.execute("search_segments", {"query": "aviation aircraft", "top_k": 1})
        segment = registry.execute("read_segment", {"segment_id": "seg_0002"})
        window = registry.execute("expand_window", {"segment_id": "seg_0002", "before_sec": 15.0, "after_sec": 50.0})
        zoom = registry.execute("zoom", {"segment_id": "seg_0002", "target_granularity_sec": 20.0})

        self.assertIn("3 segments", listing["claim"])
        self.assertEqual(search["regions"][0]["segment_id"], "seg_0002")
        self.assertIn("AVIATION HISTORY", segment["claim"])
        self.assertEqual(window["regions"][0]["start_sec"], 25.0)
        self.assertEqual(window["regions"][0]["end_sec"], 120.0)
        self.assertIn("Materialized", zoom["claim"])
        self.assertEqual([child["segment_id"] for child in zoom["regions"][0]["child_segments"]], ["seg_0002_z01", "seg_0002_z02"])
        zoomed_listing = registry.execute("video_ls", {"query": "aviation aircraft", "top_k": 5})
        self.assertIn("seg_0002_z01", [candidate["segment_id"] for candidate in zoomed_listing["candidates"]])

    def test_video_ls_returns_map_first_overview_candidates_and_next_steps(self):
        registry = build_video_navigation_registry(demo_video_map())

        listing = registry.execute("video_ls", {"query": "aviation aircraft", "max_segments": 2})

        self.assertIn("map-first", listing["claim"])
        self.assertEqual(listing["coverage"]["segment_count"], 3)
        self.assertEqual(listing["coverage"]["field_counts"]["ocr_text"], 1)
        self.assertEqual(len(listing["outline"]), 2)
        self.assertEqual(listing["candidates"][0]["segment_id"], "seg_0002")
        self.assertIn("entities", listing["candidates"][0]["matched_fields"])
        self.assertIn("relevance_reason", listing["candidates"][0])
        next_tools = [step["tool"] for step in listing["recommended_next_tools"]]
        self.assertIn("read_segment", next_tools)
        self.assertIn("inspect_segment", next_tools)
        self.assertIn("zoom", next_tools)

    def test_target_coverage_returns_matrix_with_candidates_and_missing_targets(self):
        registry = build_video_navigation_registry(demo_video_map())

        coverage = registry.execute(
            "target_coverage",
            {"targets": ["blue aircraft", "travel diary", "Persephone"], "top_k": 2},
        )

        self.assertEqual(coverage["coverage"][0]["target"], "blue aircraft")
        self.assertEqual(coverage["coverage"][0]["status"], "candidate")
        self.assertEqual(coverage["coverage"][0]["candidates"][0]["segment_id"], "seg_0002")
        self.assertIn("low_fps_caption", coverage["coverage"][0]["candidates"][0]["matched_fields"])
        self.assertEqual(coverage["coverage"][0]["candidates"][0]["source"], "low_fps_caption")
        self.assertEqual(coverage["coverage"][0]["candidates"][0]["directness"], "direct_mention")
        self.assertIn("blue aircraft", coverage["coverage"][0]["candidates"][0]["snippet"])
        self.assertEqual(coverage["coverage"][1]["candidates"][0]["segment_id"], "seg_0001")
        self.assertEqual(coverage["coverage"][2]["status"], "missing")
        self.assertIn("T1 blue aircraft", coverage["claim"])

    def test_read_segment_detail_returns_full_segment_fields_and_target_hits(self):
        registry = build_video_navigation_registry(demo_video_map())

        detail = registry.execute(
            "read_segment_detail",
            {"segment_id": "seg_0002", "targets": ["blue aircraft", "travel diary"]},
        )

        self.assertEqual(detail["segment_id"], "seg_0002")
        self.assertIn("A close view of a blue aircraft", detail["visual_caption"])
        self.assertIn("AVIATION HISTORY", detail["ocr_text"])
        self.assertEqual(detail["target_hits"][0]["target"], "blue aircraft")
        self.assertTrue(detail["target_hits"][0]["matched"])
        self.assertFalse(detail["target_hits"][1]["matched"])

    def test_read_segment_detail_inherits_target_coverage_targets_from_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="detail_targets")
            registry = build_video_navigation_registry(demo_video_map(), workspace=workspace)

            ProgramInterpreter(registry=registry, workspace=workspace).run(
                [
                    {
                        "tool": "target_coverage",
                        "args": {"targets": ["blue aircraft", "travel diary", "Persephone"], "top_k": 2},
                    }
                ]
            )
            detail = registry.execute("read_segment_detail", {"segment_id": "seg_0002"})

        self.assertEqual(detail["segment_id"], "seg_0002")
        self.assertIn("visual_caption", detail)
        self.assertIn("asr_summary", detail)
        self.assertEqual([hit["target"] for hit in detail["target_hits"]], ["blue aircraft", "travel diary", "Persephone"])
        self.assertEqual(detail["target_matches"][0]["target"], "blue aircraft")
        self.assertIn("travel diary", detail["unmatched_targets"])
        self.assertIn("Persephone", detail["unmatched_targets"])
        self.assertEqual(detail["recommended_next_tools"][0]["tool"], "vision_read")
        self.assertEqual(detail["recommended_next_tools"][0]["args"]["segment_id"], "seg_0002")

    def test_read_segment_detail_navigation_summary_surfaces_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="detail_digest")
            registry = build_video_navigation_registry(demo_video_map(), workspace=workspace)
            ProgramInterpreter(registry=registry, workspace=workspace).run(
                [
                    {
                        "tool": "read_segment_detail",
                        "args": {"segment_id": "seg_0002", "targets": ["blue aircraft"]},
                    }
                ]
            )

            compact = workspace.compact_ledger_text()

        self.assertIn("read_segment_detail", compact)
        self.assertIn("blue aircraft", compact)
        self.assertIn("A close view of a blue aircraft", compact)
        self.assertIn("AVIATION HISTORY", compact)

    def test_locate_targets_in_segment_returns_text_anchors_without_evidence(self):
        video_map = VideoMap(
            video_path="/videos/bernini.mp4",
            duration_sec=600.0,
            segments=[
                VideoMapSegment(
                    segment_id="seg_0002",
                    start_sec=300.0,
                    end_sec=600.0,
                    low_fps_caption="Bernini sculptures are discussed.",
                    asr_text=(
                        "Apollo appears in a mythological aside. The narration later lists "
                        "Aeneas, Anchises, and Ascanius fleeing Troy, David, The Rape of Persephone, "
                        "and Apollo and Daphne."
                    ),
                    asr_sentences=[
                        {
                            "start_sec": 410.0,
                            "end_sec": 416.0,
                            "text": "Apollo appears in a mythological aside.",
                        },
                        {
                            "start_sec": 430.0,
                            "end_sec": 448.0,
                            "text": (
                                "The narration later lists Bernini's Borghese sculptures: Aeneas, Anchises, and Ascanius fleeing Troy, "
                                "David, The Rape of Persephone, and Apollo and Daphne."
                            ),
                        },
                    ],
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="locate_targets")
            registry = build_video_navigation_registry(video_map, workspace=workspace)

            result = ProgramInterpreter(registry=registry, workspace=workspace).run(
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

        raw = observation.raw_output
        self.assertEqual(result.observation_ids, ["obs_0001"])
        self.assertEqual([candidate["target"] for candidate in raw["candidates"]], [
            "Aeneas, Anchises, and Ascanius fleeing Troy",
            "David",
            "The rape of Persephone",
            "Apollo and Daphne",
        ])
        self.assertTrue(all(candidate["source"] == "asr_sentence" for candidate in raw["candidates"]))
        self.assertTrue(all(candidate["start_sec"] == 430.0 for candidate in raw["candidates"]))
        self.assertEqual(len(raw["anchors_for_vlm"]), 1)
        self.assertEqual(raw["anchors_for_vlm"][0]["targets"], [
            "Aeneas, Anchises, and Ascanius fleeing Troy",
            "David",
            "The rape of Persephone",
            "Apollo and Daphne",
        ])
        self.assertLess(raw["anchors_for_vlm"][0]["start_sec"], 430.0)
        self.assertGreater(raw["anchors_for_vlm"][0]["end_sec"], 448.0)
        self.assertIn("verify_segment_anchors", raw["limitations"])
        self.assertEqual(raw["recommended_next_tools"][0]["tool"], "verify_segment_anchors")
        self.assertEqual(workspace.evidence_table_row_count(), 0)

    def test_locate_targets_keeps_multiple_matches_per_target(self):
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
                            "start_sec": 320.0,
                            "end_sec": 326.0,
                            "text": "The narrator previews Apollo and Daphne before the main Borghese sequence.",
                        },
                        {
                            "start_sec": 470.0,
                            "end_sec": 482.0,
                            "text": "The Borghese sculptures conclude with Apollo and Daphne shown in detail.",
                        },
                    ],
                )
            ],
        )
        registry = build_video_navigation_registry(video_map)

        located = registry.execute(
            "locate_targets_in_segment",
            {"segment_id": "seg_0002", "targets": ["Apollo and Daphne"]},
        )

        apollo_candidates = [candidate for candidate in located["candidates"] if candidate["target"] == "Apollo and Daphne"]
        self.assertEqual([candidate["start_sec"] for candidate in apollo_candidates], [320.0, 470.0])
        self.assertEqual(len(located["anchors_for_vlm"]), 2)

    def test_locate_targets_requires_context_for_common_single_token_names(self):
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
                            "start_sec": 330.0,
                            "end_sec": 336.0,
                            "text": "The lecture compares Michelangelo's David with Renaissance ideals.",
                        },
                        {
                            "start_sec": 430.0,
                            "end_sec": 438.0,
                            "text": "Bernini's David statue in the Borghese collection is shown next.",
                        },
                    ],
                )
            ],
        )
        registry = build_video_navigation_registry(video_map)

        located = registry.execute(
            "locate_targets_in_segment",
            {"segment_id": "seg_0002", "targets": ["David"]},
        )

        self.assertEqual(len(located["candidates"]), 1)
        self.assertEqual(located["candidates"][0]["start_sec"], 430.0)
        self.assertEqual(located["candidates"][0]["match_type"], "contextual_single_name")

    def test_search_segments_returns_modality_channels_and_evidence_snippets(self):
        registry = build_video_navigation_registry(demo_video_map())

        result = registry.execute("search_segments", {"query": "aviation aircraft", "top_k": 1})

        self.assertEqual(result["regions"][0]["segment_id"], "seg_0002")
        channels = {match["modality"]: match for match in result["regions"][0]["matches"]}
        self.assertIn("caption", channels)
        self.assertIn("ocr", channels)
        self.assertIn("entities", channels)
        self.assertIn("blue aircraft", channels["caption"]["evidence"])
        self.assertEqual(result["modalities"]["caption"][0]["segment_id"], "seg_0002")

    def test_search_segments_accepts_literature_style_modality_aliases(self):
        registry = build_video_navigation_registry(demo_video_map())

        result = registry.execute("search_segments", {"query": "welcome", "modalities": ["asr"], "top_k": 1})

        self.assertEqual(result["regions"][0]["segment_id"], "seg_0001")
        self.assertEqual(result["regions"][0]["matched_fields"], ["asr_text"])

    def test_video_map_from_scene_index_indexes_dual_source_asr_and_tags(self):
        scene_index = SceneIndex(
            video_path="/videos/goya.mp4",
            duration_sec=90.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=45.0,
                    low_fps_caption="A gallery wall is shown.",
                    visual_caption="Paintings hang in a museum gallery.",
                    asr_summary="The narration describes Goya's humble birth background and social class transition.",
                    asr_sentences=(
                        {
                            "cue_id": "cue-1",
                            "start_sec": 3.0,
                            "end_sec": 8.0,
                            "text": "The narration describes Goya's humble birth background.",
                        },
                    ),
                    map_summary="Goya museum gallery and early life context.",
                    entities=("Goya",),
                    topic_tags=("biography",),
                    stage_tags=("early life",),
                )
            ],
        )

        video_map = VideoMap.from_scene_index(scene_index)
        segment = video_map.get("seg_0001")
        asr_results = video_map.search("humble birth background", modalities=["asr"], top_k=1)
        entity_results = video_map.search("early life biography", modalities=["entities"], top_k=1)

        self.assertEqual(segment.low_fps_caption, "Paintings hang in a museum gallery.")
        self.assertEqual(segment.asr_text, scene_index.segments[0].asr_summary)
        self.assertEqual(segment.asr_sentences, scene_index.segments[0].asr_sentences)
        self.assertIn("Goya", segment.entities)
        self.assertIn("biography", segment.entities)
        self.assertIn("early life", segment.entities)
        self.assertEqual(asr_results[0].segment.segment_id, "seg_0001")
        self.assertEqual(asr_results[0].matched_fields, ["asr_text"])
        self.assertEqual(asr_results[0].matches[0]["modality"], "asr")
        self.assertEqual(entity_results[0].segment.segment_id, "seg_0001")

    def test_scene_index_summary_renders_target_asr_mentions(self):
        scene_index = SceneIndex(
            video_path="/videos/bernini.mp4",
            duration_sec=600.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0002",
                    start_sec=300.0,
                    end_sec=600.0,
                    map_summary="Bernini discusses Borghese sculptures.",
                    asr_sentences=(
                        {
                            "start_sec": 430.0,
                            "end_sec": 448.0,
                            "text": "The narration lists David and Apollo and Daphne.",
                        },
                    ),
                )
            ],
        )

        summary = scene_index.summary(target_hints=["David", "Apollo and Daphne", "Persephone"])

        self.assertIn("seg_0002 [300.0-600.0s] Bernini discusses Borghese sculptures.", summary)
        self.assertIn("asr mentions: David @ ~430.0s, Apollo and Daphne @ ~430.0s", summary)
        self.assertNotIn("Persephone", summary)

    def test_navigation_registry_reads_updated_video_map_store(self):
        store = VideoMapStore(demo_video_map())
        registry = build_video_navigation_registry(store)

        before = registry.execute("video_ls", {"query": "runway", "top_k": 1})
        store.update_segment("seg_0003", low_fps_caption="A runway aircraft landing sequence.")
        after = registry.execute("video_ls", {"query": "runway", "top_k": 1})

        self.assertNotEqual(before["candidates"][0]["segment_id"], "seg_0003")
        self.assertEqual(after["candidates"][0]["segment_id"], "seg_0003")
        self.assertEqual(after["coverage"]["field_counts"]["low_fps_caption"], 3)

    def test_exploration_registry_combines_navigation_and_segment_vlm_tools(self):
        backend = NavigationBackend()
        registry = build_video_exploration_registry(video_map=demo_video_map(), backend=backend)

        listing = registry.execute("video_ls", {})
        caption = registry.execute(
            "caption_segment",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0002",
                "start_sec": 40.0,
                "end_sec": 80.0,
                "question": "What is visible?",
            },
        )
        inspection = registry.execute(
            "inspect_segment",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0002",
                "start_sec": 40.0,
                "end_sec": 80.0,
                "question": "Which evidence is visible?",
            },
        )

        self.assertIn("3 segments", listing["claim"])
        self.assertEqual(caption["claim"], "caption_segment observation")
        self.assertEqual(inspection["claim"], "inspect_segment observation")
        self.assertEqual(backend.requests[0].metadata["segment_id"], "seg_0002")
        self.assertEqual(backend.requests[1].task, "inspect_segment")

        verification = registry.execute(
            "verify_ledger_answer",
            {
                "answer": "The video shows an aircraft museum.",
                "ledger_text": "- `obs_0001` claim: A close view of a blue aircraft in a museum.",
            },
        )
        self.assertIn("Ledger support", verification["claim"])

    def test_exploration_tools_can_caption_segments_and_update_video_ls(self):
        backend = NavigationBackend()
        store = VideoMapStore(
            VideoMap(
                video_path="/videos/demo.mp4",
                duration_sec=80.0,
                segments=[
                    VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=40.0),
                    VideoMapSegment(segment_id="seg_0002", start_sec=40.0, end_sec=80.0),
                ],
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="enrich")
            registry = build_video_exploration_registry(video_map=store, backend=backend, workspace=workspace)

            enriched = registry.execute(
                "caption_segments",
                {
                    "segment_ids": ["seg_0002"],
                    "question": "Create a concise search caption.",
                    "nframes": 4,
                },
            )
            listing = registry.execute("video_ls", {"query": "caption_segment", "top_k": 1})

            self.assertIn("Captioned 1 segment", enriched["claim"])
            self.assertEqual(listing["candidates"][0]["segment_id"], "seg_0002")
            self.assertEqual(store.current.get("seg_0002").low_fps_caption, "caption_segment observation")

    def test_exploration_caption_segments_uses_physical_clip_when_enabled(self):
        backend = NavigationBackend()
        store = VideoMapStore(
            VideoMap(
                video_path="/videos/demo.mp4",
                duration_sec=80.0,
                segments=[VideoMapSegment(segment_id="seg_0002", start_sec=40.0, end_sec=80.0)],
            )
        )
        extracted = []

        def fake_clip_extractor(video_path, output_path, start_sec, end_sec):
            extracted.append((video_path, output_path, start_sec, end_sec))
            Path(output_path).write_text("fake clip", encoding="utf-8")
            return output_path

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="enrich_clip")
            registry = build_video_exploration_registry(
                video_map=store,
                backend=backend,
                workspace=workspace,
                extract_clips=True,
                clip_extractor=fake_clip_extractor,
            )

            result = registry.execute("caption_segments", {"segment_ids": ["seg_0002"]})

        self.assertEqual(len(extracted), 1)
        self.assertEqual((extracted[0][2], extracted[0][3]), (40.0, 80.0))
        self.assertEqual(backend.requests[0].media_path, extracted[0][1])
        self.assertEqual(backend.requests[0].metadata["source_video_path"], "/videos/demo.mp4")
        self.assertEqual(result["input_artifacts"], [extracted[0][1]])

    def test_exploration_tools_can_ingest_asr_ocr_and_entities(self):
        store = VideoMapStore(demo_video_map())
        registry = build_video_exploration_registry(video_map=store, backend=NavigationBackend())

        result = registry.execute(
            "ingest_segment_metadata",
            {
                "segment_id": "seg_0001",
                "asr_text": "pilot explains the runway approach",
                "ocr_text": "RUNWAY 27",
                "entities": ["pilot", "runway"],
            },
        )
        search = registry.execute("search_segments", {"query": "runway", "top_k": 1})

        self.assertIn("Updated seg_0001", result["claim"])
        self.assertEqual(search["regions"][0]["segment_id"], "seg_0001")


if __name__ == "__main__":
    unittest.main()
