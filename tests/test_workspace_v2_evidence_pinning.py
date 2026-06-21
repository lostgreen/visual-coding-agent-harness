import json
from pathlib import Path

from visual_coding_agent_harness.agents.workspace_agent import WorkspaceVisualAgent
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.tools.workspace_v2 import build_workspace_v2_registry
from visual_coding_agent_harness.video.map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class ScriptedWorkspaceV2Backend(VisionLanguageBackend):
    def __init__(self, plan_responses: list[str], commit_responses: list[str]) -> None:
        self.plan_responses = plan_responses
        self.commit_responses = commit_responses
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "workspace_commit":
            return BackendResponse(text=self.commit_responses.pop(0))
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
            '{"tool":"search","args":{"query":"buffer Russia","modality":"asr"}}',
            '{"tool":"answer","args":{"text":"D","citations":["mem_0001"],"confidence":"high"}}',
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
                                    "anchor_id": "anch_search_seg_0001_001",
                                    "kind": "asr",
                                    "source_kind": "audio_fact",
                                    "excerpt": "Austria-Hungary was therefore seen as a good buffer between Russia and Western Europe.",
                                }
                            ],
                            "memory": [
                                {
                                    "kind": "answer_support",
                                    "claim": "ASR narration explains Austria-Hungary is the buffer.",
                                    "anchor_ids": ["anch_search_seg_0001_001"],
                                    "supports_option": "D",
                                    "confidence": "high",
                                }
                            ],
                        },
                    },
                }
            ),
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=2)

    result = agent.run("Why is Austria-Hungary shown between Russia and Western Europe?")

    assert result.answer == "D"
    assert result.citations == ("mem_0001",)
    assert workspace.observation_status("obs_0001") == "committed"
    memory = workspace.get_memory("mem_0001")
    assert memory is not None
    assert memory.kind == "answer_support"
    assert memory.supports_option == "D"
    assert [request.task for request in backend.requests] == [
        "workspace_plan",
        "workspace_commit",
        "workspace_plan",
    ]
