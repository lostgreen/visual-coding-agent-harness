import json
from pathlib import Path

from visual_coding_agent_harness.agents.workspace_agent import WorkspaceVisualAgent
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.tools.workspace_v2 import build_workspace_v2_registry
from visual_coding_agent_harness.video.map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class ScriptedWorkspaceV2Backend(VisionLanguageBackend):
    def __init__(
        self,
        plan_responses: list[str],
        commit_responses: list[str],
        tool_responses: list[BackendResponse] | None = None,
    ) -> None:
        self.plan_responses = plan_responses
        self.commit_responses = commit_responses
        self.tool_responses = tool_responses or []
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "workspace_commit":
            return BackendResponse(text=self.commit_responses.pop(0))
        if request.task != "workspace_plan":
            return self.tool_responses.pop(0)
        return BackendResponse(text=self.plan_responses.pop(0))


def _video_map() -> VideoMap:
    return VideoMap(
        video_path="/videos/demo.mp4",
        duration_sec=60.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=60.0,
                low_fps_caption="A Central Europe map with a shield.",
                asr_text="Austria-Hungary was therefore seen as a good buffer between Russia and Western Europe.",
            )
        ],
    )


def test_search_asr_cue_must_be_committed_before_final(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "asr_evidence_pinning")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"explore","args":{"query":"buffer Russia","modalities":["asr"],"top_k":1}}',
            (
                '{"tool":"verify_window","args":{"candidate_key":"obs_0001:cand_0001",'
                '"checks":[{"target_id":"buffer",'
                '"claim":"Narration says Austria-Hungary was a buffer between Russia and Western Europe.",'
                '"polarity":"support"}]}}'
            ),
            '{"tool":"answer","args":{"text":"D","citations":["mem_0002"],"confidence":"high"}}',
        ],
        commit_responses=[
            json.dumps(
                {
                    "tool": "commit_observation",
                    "args": {
                        "observation_id": "obs_0001",
                        "writes": {
                            "pinned_anchors": [
                                {
                                    "anchor_id": "anch_explore_obs_0001_cand_0001",
                                    "kind": "retrieval_hit",
                                    "source_kind": "retrieval_hit",
                                    "excerpt": "Matched query terms buffer, russia in asr indexes.",
                                    "segment_id": "seg_0001",
                                    "start_sec": 0.0,
                                    "end_sec": 20.0,
                                }
                            ],
                            "memory": [
                                {
                                    "kind": "retrieval_candidate",
                                    "claim": "ASR retrieval found a local candidate for the buffer relation.",
                                    "anchor_ids": ["anch_explore_obs_0001_cand_0001"],
                                    "confidence": "medium",
                                }
                            ],
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "tool": "commit_observation",
                    "args": {
                        "observation_id": "obs_0002",
                        "writes": {
                            "pinned_anchors": [
                                {
                                    "anchor_id": "clip_anch_seg_0001_00000000_00020000",
                                    "kind": "asr",
                                    "source_kind": "audio_fact",
                                    "excerpt": (
                                        "Austria-Hungary was therefore seen as a good buffer between "
                                        "Russia and Western Europe."
                                    ),
                                    "segment_id": "seg_0001",
                                    "start_sec": 0.0,
                                    "end_sec": 20.0,
                                }
                            ],
                            "memory": [
                                {
                                    "kind": "answer_support",
                                    "claim": "ASR narration verifies Austria-Hungary is the buffer.",
                                    "anchor_ids": ["clip_anch_seg_0001_00000000_00020000"],
                                    "supports_option": "D",
                                    "confidence": "high",
                                    "target_id": "buffer",
                                }
                            ],
                        },
                    },
                }
            ),
        ],
        tool_responses=[
            BackendResponse(
                text=(
                    "ASR narration says Austria-Hungary was seen as a good buffer between "
                    "Russia and Western Europe."
                ),
                raw={
                    "facts": [
                        {
                            "text": (
                                "Austria-Hungary was therefore seen as a good buffer between "
                                "Russia and Western Europe."
                            ),
                            "source_kind": "audio_fact",
                            "confidence": 0.93,
                        }
                    ],
                    "verification_results": [
                        {
                            "target_id": "buffer",
                            "claim": (
                                "Narration says Austria-Hungary was a buffer between Russia and "
                                "Western Europe."
                            ),
                            "verdict": "supported",
                            "anchor_ids": ["clip_anch_seg_0001_00000000_00020000"],
                            "source_kind": "audio_fact",
                            "confidence": 0.93,
                            "rationale": "The ASR sentence explicitly states the buffer relation.",
                        }
                    ],
                },
            )
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=3)

    result = agent.run("Why is Austria-Hungary shown between Russia and Western Europe?")

    assert result.answer == "D"
    assert result.citations == ("mem_0002",)
    assert workspace.observation_status("obs_0001") == "committed"
    assert workspace.observation_status("obs_0002") == "committed"
    candidate = workspace.get_memory("mem_0001")
    assert candidate is not None
    assert candidate.kind == "retrieval_candidate"
    memory = workspace.get_memory("mem_0002")
    assert memory is not None
    assert memory.kind == "answer_support"
    assert memory.supports_option == "D"
    assert memory.metadata.get("source_tool") == "verify_window"
    assert [request.task for request in backend.requests] == [
        "workspace_plan",
        "workspace_commit",
        "workspace_plan",
        "vision_read",
        "workspace_commit",
        "workspace_plan",
    ]
