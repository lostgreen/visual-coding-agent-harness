from pathlib import Path

import pytest

from visual_coding_agent_harness.agents.workspace_agent import WorkspaceVisualAgent
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.registry import ToolRegistry, ToolRuntimeSpec, tool
from visual_coding_agent_harness.tools.workspace_primitives import build_workspace_primitives_registry
from visual_coding_agent_harness.tools.workspace_v2 import build_workspace_v2_registry
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class ScriptedWorkspaceBackend(VisionLanguageBackend):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if not self.responses:
            return BackendResponse(text='{"tool":"answer","args":{"text":"done","citations":[],"confidence":"low"}}')
        return BackendResponse(text=self.responses.pop(0))


class ScriptedWorkspaceV2Backend(VisionLanguageBackend):
    def __init__(self, plan_responses: list[str], commit_responses: list[str]) -> None:
        self.plan_responses = plan_responses
        self.commit_responses = commit_responses
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "vision_read":
            return BackendResponse(text="Austria-Hungary was seen as a buffer between Russia and Western Europe.")
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
                asr_text="Austria-Hungary was seen as a buffer between Russia and Western Europe.",
            )
        ],
    )


def _workspace_registry() -> ToolRegistry:
    @tool(name="read_clip", description="Read a clip.")
    def read_clip() -> dict[str, object]:
        return {
            "claim": "Austria-Hungary was therefore seen as a good buffer between Russia and Western Europe.",
            "confidence": 0.9,
            "raw_output": {
                "facts": [
                    {
                        "text": "Austria-Hungary was therefore seen as a good buffer between Russia and Western Europe.",
                        "source_kind": "audio_fact",
                    }
                ]
            },
        }

    registry = ToolRegistry()
    registry.register(ToolRuntimeSpec(tool_spec=read_clip, commit_required=True))
    registry.extend(build_workspace_primitives_registry())
    return registry


def test_workspace_agent_runs_plan_act_commit_before_answer(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_commit")
    registry = ToolRegistry()
    registry.register(_workspace_registry().get_runtime_spec("read_clip"))
    registry.extend(build_workspace_primitives_registry(workspace=workspace))
    backend = ScriptedWorkspaceBackend(
        [
            '{"tool":"read_clip","args":{}}',
            """
            {
              "tool": "commit_observation",
              "args": {
                "observation_id": "obs_0001",
                "writes": {
                  "pinned_anchors": [{
                    "anchor_id": "anch_asr_206",
                    "kind": "asr",
                    "source_kind": "audio_fact",
                    "excerpt": "Austria-Hungary was therefore seen as a good buffer between Russia and Western Europe."
                  }],
                  "memory": [{
                    "kind": "answer_support",
                    "claim": "The narration says Austria-Hungary was a buffer.",
                    "supports_option": "D",
                    "anchor_ids": ["anch_asr_206"],
                    "evidence_obs_ids": ["obs_0001"],
                    "confidence": "high"
                  }]
                }
              }
            }
            """,
            '{"tool":"answer","args":{"text":"D","citations":["mem_0001"],"confidence":"high"}}',
        ]
    )
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=2)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    assert result.answer == "D"
    assert result.citations == ("mem_0001",)
    assert workspace.observation_status("obs_0001") == "committed"
    assert workspace.memory_entries()[0].entry_id == "mem_0001"
    assert [request.task for request in backend.requests] == [
        "workspace_plan",
        "workspace_commit",
        "workspace_plan",
    ]


def test_workspace_agent_rejects_exploration_tool_during_commit_phase(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_bad_commit")
    registry = ToolRegistry()
    registry.register(_workspace_registry().get_runtime_spec("read_clip"))
    registry.extend(build_workspace_primitives_registry(workspace=workspace))
    backend = ScriptedWorkspaceBackend(
        [
            '{"tool":"read_clip","args":{}}',
            '{"tool":"read_clip","args":{}}',
        ]
    )
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    with pytest.raises(ValueError, match="commit phase only accepts disposition tools"):
        agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")


def test_workspace_agent_retries_commit_and_validates_final_answer(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_retry_commit")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"},"focus":["buffer"]}}',
            '{"tool":"answer","args":{"text":"D","citations":["mem_0001"],"confidence":"high"}}',
        ],
        commit_responses=[
            """
            {
              "tool": "commit_observation",
              "args": {
                "observation_id": "obs_0001",
                "writes": {
                  "pinned_anchors": [{"anchor_id":"anch_bad","kind":"asr","source_kind":"audio_fact","excerpt":"not in observation"}],
                  "memory": [{"kind":"answer_support","claim":"bad","anchor_ids":["anch_bad"],"confidence":"high"}]
                }
              }
            }
            """,
            """
            {
              "tool": "commit_observation",
              "args": {
                "observation_id": "obs_0001",
                "writes": {
                  "pinned_anchors": [{
                    "anchor_id":"anch_asr_206",
                    "kind":"asr",
                    "source_kind":"audio_fact",
                    "excerpt":"buffer between Russia and Western Europe"
                  }],
                  "memory": [{
                    "kind":"answer_support",
                    "claim":"Narration says Austria-Hungary was a buffer.",
                    "supports_option":"D",
                    "anchor_ids":["anch_asr_206"],
                    "confidence":"high"
                  }]
                }
              }
            }
            """,
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=2)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    assert result.answer == "D"
    assert result.citations == ("mem_0001",)
    assert workspace.observation_status("obs_0001") == "committed"
    commit_prompts = [request.prompt for request in backend.requests if request.task == "workspace_commit"]
    assert len(commit_prompts) == 2
    assert "excerpt must appear" in commit_prompts[1]


def test_workspace_agent_force_acknowledges_after_commit_retry_exhaustion(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_force_ack")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"},"focus":["buffer"]}}',
        ],
        commit_responses=[
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001","writes":{"pinned_anchors":[{"anchor_id":"anch_bad","kind":"asr","source_kind":"audio_fact","excerpt":"missing"}]}}}',
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001","writes":{"pinned_anchors":[{"anchor_id":"anch_bad","kind":"asr","source_kind":"audio_fact","excerpt":"missing"}]}}}',
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001","writes":{"pinned_anchors":[{"anchor_id":"anch_bad","kind":"asr","source_kind":"audio_fact","excerpt":"missing"}]}}}',
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    assert result.metadata == {"reason": "max_rounds_reached"}
    assert workspace.observation_status("obs_0001") == "acknowledged"
