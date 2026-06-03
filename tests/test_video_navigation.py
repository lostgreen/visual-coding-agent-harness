import unittest

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.tools.exploration import build_video_exploration_registry
from visual_coding_agent_harness.tools.navigation import build_video_navigation_registry
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment


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

        self.assertIn("3 segments", listing["claim"])
        self.assertEqual(search["regions"][0]["segment_id"], "seg_0002")
        self.assertIn("AVIATION HISTORY", segment["claim"])
        self.assertEqual(window["regions"][0]["start_sec"], 25.0)
        self.assertEqual(window["regions"][0]["end_sec"], 120.0)

    def test_video_ls_returns_map_first_overview_candidates_and_next_steps(self):
        registry = build_video_navigation_registry(demo_video_map())

        listing = registry.execute("video_ls", {"query": "aviation aircraft", "max_segments": 2})

        self.assertIn("map-first", listing["claim"])
        self.assertEqual(listing["coverage"]["segment_count"], 3)
        self.assertEqual(listing["coverage"]["field_counts"]["ocr_text"], 1)
        self.assertEqual(len(listing["outline"]), 2)
        self.assertEqual(listing["candidates"][0]["segment_id"], "seg_0002")
        self.assertIn("entities", listing["candidates"][0]["matched_fields"])
        next_tools = [step["tool"] for step in listing["recommended_next_tools"]]
        self.assertIn("read_segment", next_tools)
        self.assertIn("caption_segment", next_tools)

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

        self.assertIn("3 segments", listing["claim"])
        self.assertEqual(caption["claim"], "caption_segment observation")
        self.assertEqual(backend.requests[0].metadata["segment_id"], "seg_0002")


if __name__ == "__main__":
    unittest.main()
