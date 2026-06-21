from pathlib import Path

import pytest

from visual_coding_agent_harness.agents.workspace_agent import (
    PLAN_SYSTEM_PROMPT,
    WorkspaceVisualAgent,
    compose_commit_prompt,
    compose_final_prompt,
    compose_plan_prompt,
    _parse_action,
)
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.evals.videomme.workspace_round_log import export_workspace_round_log
from visual_coding_agent_harness.core.registry import ToolRegistry, ToolRuntimeSpec, tool
from visual_coding_agent_harness.tools.workspace_primitives import build_workspace_primitives_registry
from visual_coding_agent_harness.tools.workspace_v2 import build_workspace_v2_registry
from visual_coding_agent_harness.video.index import TimelineBeat
from visual_coding_agent_harness.video.map import IndexRefiner, VideoMap, VideoMapSegment, VideoMapStore
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
        if request.task == "refine_segment_index":
            return BackendResponse(
                text=(
                    '{"children":[{"start_sec":5,"end_sec":20,'
                    '"summary":"Fresh refined map view with Austria-Hungary between Russia and Western Europe.",'
                    '"beats":[{"start_sec":5,"end_sec":20,"summary":"Austria-Hungary is shown as the buffer.",'
                    '"modality_hints":["visual"]}],'
                    '"entity_hints":["Austria-Hungary","Russia","Western Europe"]}]}'
                )
            )
        if request.task == "workspace_commit":
            return BackendResponse(text=self.commit_responses.pop(0))
        if request.task == "workspace_final" and not self.plan_responses:
            return BackendResponse(text='{"tool":"answer","args":{"text":"D","citations":[],"confidence":"low"}}')
        return BackendResponse(text=self.plan_responses.pop(0))


class EmptyCommitRequiredBackend(VisionLanguageBackend):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "workspace_final" and not self.responses:
            return BackendResponse(text='{"tool":"answer","args":{"text":"D","citations":[],"confidence":"low"}}')
        return BackendResponse(text=self.responses.pop(0))


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
                timeline_beats=(
                    TimelineBeat(
                        beat_id="seg_0001_b01",
                        start_sec=0.0,
                        end_sec=60.0,
                        summary="A map introduces Austria-Hungary as a buffer region.",
                        modality_hints=("visual", "asr"),
                    ),
                ),
            )
        ],
    )


def _assert_forced_final_metadata(result: object) -> None:
    metadata = getattr(result, "metadata", None)
    assert metadata is not None
    assert metadata["reason"] == "max_rounds_reached"
    assert metadata["forced_final"] is True
    assert "validated" in metadata


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


def _empty_commit_required_registry(workspace: EvidenceWorkspace) -> ToolRegistry:
    @tool(name="read_empty", description="Return an unanchored observation.")
    def read_empty() -> dict[str, object]:
        return {
            "claim": "No concrete fact was extracted.",
            "confidence": 0.1,
            "facts": [],
            "produced_anchors": [],
        }

    registry = ToolRegistry()
    registry.register(ToolRuntimeSpec(tool_spec=read_empty, commit_required=True))
    registry.extend(build_workspace_primitives_registry(workspace=workspace))
    return registry


def test_workspace_agent_parse_action_accepts_trailing_json_object() -> None:
    action = _parse_action(
        '{"tool":"verify","args":{"answer":"D","citations":["mem_0001"]}}\n'
        '{"tool":"answer","args":{"text":"D","citations":["mem_0001"],"confidence":"high"}}'
    )

    assert action == {"tool": "verify", "args": {"answer": "D", "citations": ["mem_0001"]}}


def test_workspace_agent_parse_action_accepts_fenced_json_with_tail_text() -> None:
    action = _parse_action(
        '```json\n{"tool":"read_clip","args":{"scope":{},"focus":["overall evidence"]}}\n```\nDone.'
    )

    assert action == {"tool": "read_clip", "args": {"scope": {}, "focus": ["overall evidence"]}}


def test_workspace_agent_parse_action_accepts_nested_action_json() -> None:
    action = _parse_action(
        '{"analysis":"need one more check",'
        '"action":{"tool":"verify","args":{"claim":"D","against":{"citations":["mem_0001"]}}}}'
    )

    assert action == {"tool": "verify", "args": {"claim": "D", "against": {"citations": ["mem_0001"]}}}


def test_workspace_agent_recovers_from_rejected_plan_tool(tmp_path: Path) -> None:
    @tool(name="bad_tool", description="Always reject.")
    def bad_tool() -> dict[str, object]:
        raise ValueError("bad_tool_failed: invalid planner args")

    workspace = EvidenceWorkspace.create(tmp_path, "workspace_tool_rejected")
    registry = ToolRegistry()
    registry.register(ToolRuntimeSpec(tool_spec=bad_tool))
    backend = ScriptedWorkspaceBackend(
        [
            '{"tool":"bad_tool","args":{}}',
            '{"tool":"answer","args":{"text":"done","citations":[],"confidence":"low"}}',
        ]
    )
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=2)

    result = agent.run("Question: demo")

    assert result.answer == "done"
    assert result.rounds == 2
    trace_events = workspace._read_jsonl_dicts("trace.jsonl")
    rejected = [event for event in trace_events if event["type"] == "workspace_tool_rejected"]
    assert rejected[0]["payload"]["tool"] == "bad_tool"
    assert "bad_tool_failed" in rejected[0]["payload"]["error"]


def test_workspace_agent_duplicate_guard_persists_across_rounds(tmp_path: Path) -> None:
    @tool(name="probe", description="Probe once.")
    def probe(target: str) -> dict[str, object]:
        return {"claim": f"found {target}", "confidence": 0.9}

    workspace = EvidenceWorkspace.create(tmp_path, "workspace_cross_round_duplicate_guard")
    registry = ToolRegistry()
    registry.register(
        ToolRuntimeSpec(
            tool_spec=probe,
            semantic_key_builder=lambda _ctx, request: f"probe:{request.arguments['target']}",
        )
    )
    backend = ScriptedWorkspaceBackend(
        [
            '{"tool":"probe","args":{"target":"same"}}',
            '{"tool":"probe","args":{"target":"same"}}',
            '{"tool":"answer","args":{"text":"done","citations":[],"confidence":"low"}}',
        ]
    )
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=3)

    result = agent.run("Question: demo")

    assert result.answer == "done"
    assert result.rounds == 3
    observations = workspace.read_observations()
    assert len(observations) == 1
    trace_events = workspace._read_jsonl_dicts("trace.jsonl")
    rejected = [event for event in trace_events if event["type"] == "workspace_tool_rejected"]
    assert rejected[0]["payload"]["error"] == "duplicate_tool_call: probe repeats semantic key probe:same."


def test_workspace_agent_forces_answer_at_max_rounds(tmp_path: Path) -> None:
    @tool(name="probe", description="Gather one clue.")
    def probe() -> dict[str, object]:
        return {"claim": "The only clue points to option B.", "confidence": 0.7}

    workspace = EvidenceWorkspace.create(tmp_path, "workspace_forced_final")
    registry = ToolRegistry()
    registry.register(ToolRuntimeSpec(tool_spec=probe))
    backend = ScriptedWorkspaceBackend(
        [
            '{"tool":"probe","args":{}}',
            '{"tool":"answer","args":{"text":"B","citations":[],"confidence":"low"}}',
        ]
    )
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Question: demo")

    assert result.answer == "B"
    assert result.rounds == 1
    assert [request.task for request in backend.requests] == ["workspace_plan", "workspace_final"]
    trace_events = workspace._read_jsonl_dicts("trace.jsonl")
    assert any(event["type"] == "workspace_final_model_io" for event in trace_events)


def test_workspace_agent_forced_answer_validation_rejection_returns_unvalidated_answer(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_forced_final_unvalidated")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"list","args":{"kind":"memory"}}',
            '{"tool":"answer","args":{"text":"C","citations":[],"confidence":"low"}}',
        ],
        commit_responses=[],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Question: demo")

    assert result.answer == "C"
    assert result.confidence == "low"
    assert result.citations == ()
    assert result.metadata is not None
    assert result.metadata["status"] == "low_confidence_final"
    assert result.metadata["forced_final"] is True
    assert result.metadata["validated"] is False
    assert result.metadata["attempted_answer"] == "C"
    assert result.metadata["attempted_citations"] == []
    assert "answer_validation_failed" in str(result.metadata["validation_error"])
    trace_events = workspace._read_jsonl_dicts("trace.jsonl")
    assert any(event["type"] == "workspace_forced_answer_unvalidated" for event in trace_events)


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


def test_workspace_agent_runs_read_segment_index_refine_verify_commit_answer(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_read_segment_progressive")
    sampled = []

    def fake_frame_sampler(video_path: str, start_sec: float, end_sec: float, max_frames: int) -> list[str]:
        sampled.append((video_path, start_sec, end_sec, max_frames))
        return ["/frames/demo/00005.jpg"]

    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"read_segment","args":{"segment_id":"seg_0001","mode":"index"}}',
            '{"tool":"read_segment","args":{"segment_id":"seg_0001","mode":"refine","sub_window":{"start_sec":5,"end_sec":20},"resolution":"medium","focus":["buffer"]}}',
            '{"tool":"read_segment","args":{"segment_id":"seg_0001","mode":"verify","sub_window":{"start_sec":5,"end_sec":20},"evidence_mode":"visual","focus":["buffer"]}}',
            '{"tool":"answer","args":{"text":"D","citations":["mem_0001"],"confidence":"high"}}',
        ],
        commit_responses=[
            """
            {
              "tool": "commit_observation",
              "args": {
                "observation_id": "obs_0003",
                "writes": {
                  "pinned_anchors": [{
                    "anchor_id": "clip_anch_seg_0001_00005000_00020000",
                    "kind": "visual",
                    "source_kind": "visual_fact",
                    "excerpt": "Austria-Hungary was seen as a buffer between Russia and Western Europe."
                  }],
                  "memory": [{
                    "kind": "answer_support",
                    "claim": "Austria-Hungary is identified as the buffer.",
                    "supports_option": "D",
                    "anchor_ids": ["clip_anch_seg_0001_00005000_00020000"],
                    "evidence_obs_ids": ["obs_0003"],
                    "confidence": "high"
                  }]
                }
              }
            }
            """
        ],
    )
    video_map = _video_map()
    registry = build_workspace_v2_registry(
        video_map=video_map,
        backend=backend,
        workspace=workspace,
        index_refiner=IndexRefiner(backend=backend, frame_sampler=fake_frame_sampler),
    )
    agent = WorkspaceVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        max_rounds=4,
        video_map=video_map,
    )

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    assert not hasattr(agent, "runtime_host")
    assert result.answer == "D"
    assert workspace.observation_status("obs_0001") == "acknowledged"
    assert workspace.observation_status("obs_0002") == "acknowledged"
    assert workspace.observation_status("obs_0003") == "committed"
    assert workspace.memory_entries()[0].kind == "answer_support"
    assert sampled == [("/videos/demo.mp4", 5.0, 20.0, 15)]
    assert [request.task for request in backend.requests] == [
        "workspace_plan",
        "workspace_plan",
        "refine_segment_index",
        "workspace_plan",
        "vision_read",
        "workspace_commit",
        "workspace_plan",
    ]


def test_plan_prompt_renders_latest_refinement_from_video_map_store(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_latest_refinement_prompt")
    backend = ScriptedWorkspaceV2Backend(plan_responses=[], commit_responses=[])
    store = VideoMapStore(_video_map())

    def fake_frame_sampler(video_path: str, start_sec: float, end_sec: float, max_frames: int) -> list[str]:
        del video_path, start_sec, end_sec, max_frames
        return ["/frames/demo/00005.jpg"]

    registry = build_workspace_v2_registry(
        video_map=store,
        backend=backend,
        workspace=workspace,
        index_refiner=IndexRefiner(backend=backend, frame_sampler=fake_frame_sampler),
    )
    registry.execute("read_segment", {"segment_id": "seg_0001", "mode": "index"})
    registry.execute(
        "read_segment",
        {
            "segment_id": "seg_0001",
            "mode": "refine",
            "sub_window": {"start_sec": 5.0, "end_sec": 20.0},
            "resolution": "medium",
            "focus": ["buffer"],
        },
    )

    prompt = compose_plan_prompt(question="Why?", workspace=workspace, video_map=store)

    assert "## Latest Index Patch" in prompt
    assert "Fresh refined map view with Austria-Hungary between Russia and Western Europe." in prompt
    assert "refined: 5.0-20.0s" in prompt


def test_forced_final_prompt_renders_video_map_store(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_final_prompt_store")
    store = VideoMapStore(_video_map())

    prompt = compose_final_prompt(question="Why?", workspace=workspace, video_map=store)

    assert "## Segment Cards" in prompt
    assert "seg_0001 [0.0-60.0s]" in prompt


def test_workspace_agent_rejects_exploration_tool_during_commit_phase_then_auto_pins(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_bad_commit")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"},"focus":["buffer"]}}',
        ],
        commit_responses=[
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"}}}',
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"}}}',
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"}}}',
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    _assert_forced_final_metadata(result)
    assert workspace.observation_status("obs_0001") == "committed"
    assert workspace.memory_entries()[0].kind == "unverified_capture"
    assert [request.task for request in backend.requests] == [
        "workspace_plan",
        "vision_read",
        "workspace_commit",
        "workspace_commit",
        "workspace_commit",
        "workspace_final",
    ]


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
                    "anchor_id":"clip_anch_seg_0001_00000000_00060000",
                    "kind":"visual",
                    "source_kind":"visual_fact",
                    "excerpt":"buffer between Russia and Western Europe"
                  }],
                  "memory": [{
                    "kind":"answer_support",
                    "claim":"Narration says Austria-Hungary was a buffer.",
                    "supports_option":"D",
                    "anchor_ids":["clip_anch_seg_0001_00000000_00060000"],
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
    assert "not in observation produced_anchors" in commit_prompts[1]


def test_workspace_agent_accepts_legacy_answer_field_after_commit(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_legacy_answer_arg")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"},"focus":["buffer"]}}',
            '{"tool":"answer","args":{"answer":"D","citations":["mem_0001"],"confidence":"high"}}',
        ],
        commit_responses=[
            """
            {
              "tool": "commit_observation",
              "args": {
                "observation_id": "obs_0001",
                "writes": {
                  "pinned_anchors": [{
                    "anchor_id":"clip_anch_seg_0001_00000000_00060000",
                    "kind":"visual",
                    "source_kind":"visual_fact",
                    "excerpt":"buffer between Russia and Western Europe"
                  }],
                  "memory": [{
                    "kind":"answer_support",
                    "claim":"Narration says Austria-Hungary was a buffer.",
                    "supports_option":"D",
                    "anchor_ids":["clip_anch_seg_0001_00000000_00060000"],
                    "evidence_obs_ids":["obs_0001"],
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


def test_workspace_agent_normalizes_legacy_commit_claim_to_answer_support(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_legacy_commit_claim")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"},"focus":["buffer"]}}',
            '{"tool":"answer","args":{"text":"D","citations":["mem_0001"],"confidence":"high"}}',
        ],
        commit_responses=[
            '{"tool":"commit_observation","args":{"claim":"Narration says Austria-Hungary was a buffer."}}',
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=2)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    assert result.answer == "D"
    assert result.citations == ("mem_0001",)
    assert workspace.observation_status("obs_0001") == "committed"
    assert workspace.memory_entries()[0].kind == "answer_support"


def test_workspace_agent_rejects_uncited_answer_and_continues(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_reject_uncited_answer")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"answer","args":{"answer":"B. Why the Austro-Hungarian Empire was divided."}}',
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"},"focus":["buffer"]}}',
            '{"tool":"answer","args":{"answer":"D","citations":["mem_0001"],"confidence":"high"}}',
        ],
        commit_responses=[
            """
            {
              "tool": "commit_observation",
              "args": {
                "observation_id": "obs_0001",
                "writes": {
                  "pinned_anchors": [{
                    "anchor_id":"clip_anch_seg_0001_00000000_00060000",
                    "kind":"visual",
                    "source_kind":"visual_fact",
                    "excerpt":"buffer between Russia and Western Europe"
                  }],
                  "memory": [{
                    "kind":"answer_support",
                    "claim":"Narration says Austria-Hungary was a buffer.",
                    "supports_option":"D",
                    "anchor_ids":["clip_anch_seg_0001_00000000_00060000"],
                    "evidence_obs_ids":["obs_0001"],
                    "confidence":"high"
                  }]
                }
              }
            }
            """,
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=3)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    assert result.answer == "D"
    assert result.citations == ("mem_0001",)
    assert [request.task for request in backend.requests] == [
        "workspace_plan",
        "workspace_plan",
        "vision_read",
        "workspace_commit",
        "workspace_plan",
    ]
    assert any(event["type"] == "workspace_answer_rejected" for event in workspace._read_jsonl_dicts("trace.jsonl"))


def test_compose_commit_prompt_includes_full_view_on_minimal_retry(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_commit_prompt_retry")
    observation = workspace.write_observation(
        tool_name="read_clip",
        claim="Austria-Hungary was seen as a buffer.",
        confidence=0.8,
        raw_output={
            "facts": [{"text": "Austria-Hungary was seen as a buffer.", "source_kind": "audio_fact"}],
            "produced_anchors": [
                {
                    "anchor_id": "anch_clip_seg_0001_001",
                    "observation_id": "__pending__",
                    "source_kind": "audio_fact",
                    "segment_id": "seg_0001",
                    "field_path": "facts[0].text",
                    "excerpt": "Austria-Hungary was seen as a buffer.",
                    "modality": "asr",
                }
            ],
        },
    )

    prompt = compose_commit_prompt(
        question="Why?",
        workspace=workspace,
        observation_id=observation.observation_id,
        validation_error="anchor_id=bad not in observation produced_anchors",
        attempt=3,
        prompt_mode="minimal",
    )

    assert "# Commit Phase (attempt 3)" in prompt
    assert "## Candidate Anchors (verbatim excerpts you may pin)" in prompt
    assert "anch_clip_seg_0001_001" in prompt
    assert "# Validation Error From Previous Attempt" in prompt
    assert "# Minimal Commit Mode" in prompt
    assert '"tool": "commit_observation"' in prompt
    assert '"pinned_anchors"' in prompt
    assert '"memory"' in prompt
    assert '"anchor_id": "<candidate_anchor_id>"' in prompt


def test_prompts_guide_whole_video_coverage_and_partial_evidence_commits(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_prompt_guidance")
    observation = workspace.write_observation(
        tool_name="read_segment",
        claim="The opening root shows Austria-Hungary rising through maps and narration.",
        confidence=0.8,
        raw_output={
            "facts": [{"text": "Austria-Hungary rises through maps and narration.", "source_kind": "visual_fact"}],
            "produced_anchors": [
                {
                    "anchor_id": "clip_anch_seg_0001_00000000_00060000",
                    "source_kind": "visual_fact",
                    "modality": "visual",
                    "excerpt": "Austria-Hungary rises through maps and narration.",
                }
            ],
        },
    )

    plan_prompt = compose_plan_prompt(
        question="What is the main idea of the video?",
        workspace=workspace,
        last_tool_result="observation rejected: obs_0001",
        video_map=_video_map(),
    )
    commit_prompt = compose_commit_prompt(
        question="What is the main idea of the video?",
        workspace=workspace,
        observation_id=observation.observation_id,
    )

    assert "whole-video or main-idea" in plan_prompt
    assert "early, middle, and late segment cards" in plan_prompt
    assert "partial evidence" in commit_prompt
    assert "answer_support" in commit_prompt
    assert "Reject only when" in commit_prompt


def test_compose_plan_prompt_blocks_uncited_answers_without_memory(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_plan_protocol")

    prompt = compose_plan_prompt(
        question="What's the main idea?",
        workspace=workspace,
        last_tool_result="answer rejected: final answer requires at least one planner-authored memory citation",
    )

    assert "Available plan tools" in PLAN_SYSTEM_PROMPT
    assert "scan_segment" in PLAN_SYSTEM_PROMPT
    assert "verify_window" in PLAN_SYSTEM_PROMPT
    assert "standalone verify" in PLAN_SYSTEM_PROMPT
    assert "Every answer call must include" in PLAN_SYSTEM_PROMPT
    assert '"text":"D"' not in PLAN_SYSTEM_PROMPT
    assert "Use Segment Cards as the starting navigation state" in prompt
    assert '{"tool":"scan_segment"' in prompt
    assert '{"tool":"verify_window"' in prompt
    assert "synthesize_memory is unavailable until committed memory exists" in prompt
    assert "duplicate_tool_call" in prompt
    assert "do not repeat the same semantic request" in prompt
    assert "refinement_output_invalid" in prompt
    assert "candidate_id" in prompt
    assert '"text":"D"' not in prompt
    assert '"text":"A"' in prompt


def test_workspace_agent_exports_downloadable_round_log(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_round_log")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"},"focus":["buffer"]}}',
            '{"tool":"answer","args":{"text":"D","citations":["mem_0001"],"confidence":"high"}}',
        ],
        commit_responses=[
            '{"tool":"commit_observation","args":{"claim":"Narration says Austria-Hungary was a buffer."}}',
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=2)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")
    artifact = export_workspace_round_log(
        workspace,
        question="Why was Austria-Hungary shown between Russia and Western Europe?",
        video_path="/videos/demo.mp4",
        final={"answer": result.answer, "status": "final", "citations": list(result.citations)},
    )

    log_path = Path(str(artifact["path"]))
    assert log_path.exists()
    markdown = log_path.read_text(encoding="utf-8")
    assert "## First Planner Workspace View" in markdown
    assert "### Round 1" in markdown
    assert "round_001_plan_prompt.txt" in markdown
    assert "round_001_commit_attempt_01_prompt.txt" in markdown
    assert "mem_0001 [answer_support]" in markdown
    plan_prompt = workspace.root / "workspace_logs/round_001_plan_prompt.txt"
    assert plan_prompt.exists()
    assert "# Workspace" in plan_prompt.read_text(encoding="utf-8")
    trace_events = workspace._read_jsonl_dicts("trace.jsonl")
    assert any(event["type"] == "workspace_round_log_export" for event in trace_events)


def test_workspace_agent_round_log_includes_forced_final_io(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_forced_final_round_log")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"search","args":{"query":"does-not-exist","modality":"asr"}}',
        ],
        commit_responses=[],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")
    artifact = export_workspace_round_log(
        workspace,
        question="Why was Austria-Hungary shown between Russia and Western Europe?",
        video_path="/videos/demo.mp4",
        final={"answer": result.answer, "status": "final", "citations": list(result.citations)},
    )

    markdown = Path(str(artifact["path"])).read_text(encoding="utf-8")
    assert "#### Forced Final IO" in markdown
    assert "round_001_final_prompt.txt" in markdown
    assert "round_001_final_response.txt" in markdown
    assert "Final prompt:" in markdown
    assert "# Forced Final Protocol" in markdown
    assert "Final response:" in markdown
    assert '"tool":"answer"' in markdown


def test_workspace_agent_commits_search_hit_with_evidence_excerpt(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_search_commit")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"search","args":{"query":"buffer Russia","modality":"asr"}}',
            '{"tool":"answer","args":{"text":"D","citations":["mem_0001"],"confidence":"high"}}',
        ],
        commit_responses=[
            """
            {
              "tool": "commit_observation",
              "args": {
                "observation_id": "obs_0001",
                "writes": {
                  "pinned_anchors": [{
                    "anchor_id":"anch_search_seg_0001_001",
                    "kind":"asr",
                    "source_kind":"audio_fact",
                    "excerpt":"Austria-Hungary was seen as a buffer between Russia and Western Europe."
                  }],
                  "memory": [{
                    "kind":"answer_support",
                    "claim":"ASR says Austria-Hungary was the buffer.",
                    "supports_option":"D",
                    "anchor_ids":["anch_search_seg_0001_001"],
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
    assert [request.task for request in backend.requests] == [
        "workspace_plan",
        "workspace_commit",
        "workspace_plan",
    ]


def test_workspace_agent_acknowledges_search_without_evidence_excerpt(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_search_no_evidence")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"search","args":{"query":"does-not-exist","modality":"asr"}}',
        ],
        commit_responses=[],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    _assert_forced_final_metadata(result)
    assert workspace.observation_status("obs_0001") == "acknowledged"
    assert [request.task for request in backend.requests] == ["workspace_plan", "workspace_final"]


def test_workspace_agent_auto_pins_retrieval_candidate_after_bad_commit_schema(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_search_candidate_auto_pin")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"search","args":{"query":"buffer Russia","modality":"asr"}}',
        ],
        commit_responses=[
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001","writes":"anch_search_seg_0001_001"}}',
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001","writes":"still invalid"}}',
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001"}}',
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    _assert_forced_final_metadata(result)
    assert workspace.observation_status("obs_0001") == "committed"
    memory = workspace.memory_entries()
    assert memory[0].kind == "retrieval_candidate"
    assert memory[0].metadata["requires_local_read"] is True
    assert "read_clip" in workspace.render_plan_view(question="Why?")
    validation_errors = [
        event["payload"]["error"]
        for event in workspace._read_jsonl_dicts("trace.jsonl")
        if event["type"] == "workspace_commit_validation_error"
    ]
    assert any("writes must be an object" in error for error in validation_errors)


def test_workspace_agent_commit_shorthand_preserves_retrieval_candidate_kind(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_search_candidate_shorthand")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"search","args":{"query":"buffer Russia","modality":"asr"}}',
        ],
        commit_responses=[
            (
                '{"tool":"commit_observation","args":{'
                '"anchor_id":"anch_search_seg_0001_001",'
                '"claim":"Candidate search hit requires local read before answer.",'
                '"kind":"retrieval_candidate",'
                '"confidence":"low"'
                "}}"
            ),
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    _assert_forced_final_metadata(result)
    memory = workspace.memory_entries()
    assert memory[0].kind == "retrieval_candidate"
    assert memory[0].metadata["requires_local_read"] is True
    assert memory[0].metadata["cannot_final_cite"] is True
    assert memory[0].metadata["recommended_next_tool"] == "read_clip"
    assert "read_clip candidate anch_search_seg_0001_001" in workspace.render_plan_view(question="Why?")


def test_workspace_agent_auto_pins_after_commit_retry_exhaustion(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_auto_pin")
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

    _assert_forced_final_metadata(result)
    assert workspace.observation_status("obs_0001") == "committed"
    memory = workspace.memory_entries()
    assert memory[0].kind == "unverified_capture"
    assert memory[0].metadata["auto_pinned"] is True
    assert any(event["type"] == "commit_auto_pinned" for event in workspace._read_jsonl_dicts("trace.jsonl"))


def test_workspace_agent_auto_pin_failure_defers_instead_of_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_auto_pin_failure_defers")
    observation = workspace.write_observation(
        tool_name="read_clip",
        claim="The model summarized an option-relevant visual cue.",
        confidence=0.7,
        raw_output={
            "facts": [{"text": "The visual summary points toward option C."}],
            "produced_anchors": [
                {
                    "anchor_id": "anch_clip_seg_0001_001",
                    "observation_id": "__pending__",
                    "source_kind": "visual_fact",
                    "segment_id": "seg_0001",
                    "field_path": "facts[0].text",
                    "excerpt": "visible source snippet",
                    "modality": "visual",
                }
            ],
        },
    )
    backend = ScriptedWorkspaceV2Backend(plan_responses=[], commit_responses=[])
    agent = WorkspaceVisualAgent(
        backend=backend,
        registry=build_workspace_primitives_registry(workspace=workspace),
        workspace=workspace,
    )

    def reject_commit(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("anchor_validation_failed: excerpt must appear in observation obs_0001")

    monkeypatch.setattr(workspace, "commit_observation", reject_commit)

    agent._auto_pin_observation(observation, reason="commit_format_failure: bad model commit")

    assert workspace.observation_status(observation.observation_id) == "deferred"
    trace_events = workspace._read_jsonl_dicts("trace.jsonl")
    deferred = [event for event in trace_events if event["type"] == "commit_auto_deferred"]
    assert deferred[-1]["payload"]["auto_pin_failed"] is True
    assert "anchor_validation_failed" in deferred[-1]["payload"]["auto_pin_error"]


def test_workspace_agent_auto_pins_after_malformed_disposition_args(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_auto_pin_tool_error")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"},"focus":["buffer"]}}',
        ],
        commit_responses=[
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001"}}',
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001"}}',
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001"}}',
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    _assert_forced_final_metadata(result)
    assert workspace.observation_status("obs_0001") == "committed"
    assert workspace.memory_entries()[0].kind == "unverified_capture"


def test_workspace_agent_treats_disposition_field_as_commit_tool(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_disposition_field")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"},"focus":["buffer"]}}',
        ],
        commit_responses=[
            '{"disposition":"defer_observation"}',
            '{"disposition":"defer_observation"}',
            '{"disposition":"defer_observation"}',
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    _assert_forced_final_metadata(result)
    assert workspace.observation_status("obs_0001") == "deferred"
    assert workspace.memory_entries() == []


def test_workspace_agent_accepts_obs_id_alias_and_surfaces_rejection_to_next_plan(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_commit_obs_id_alias")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"},"focus":["buffer"]}}',
            '{"tool":"answer","args":{"text":"D","citations":[],"confidence":"low"}}',
        ],
        commit_responses=[
            '{"tool":"reject_observation","args":{"obs_id":"obs_0001","reason":"not answer support yet"}}',
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=2)

    agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    assert workspace.observation_status("obs_0001") == "rejected"
    validation_errors = [
        event
        for event in workspace._read_jsonl_dicts("trace.jsonl")
        if event["type"] == "workspace_commit_validation_error"
    ]
    assert validation_errors == []
    second_plan = [request for request in backend.requests if request.task == "workspace_plan"][1]
    assert "observation rejected: obs_0001" in second_plan.prompt
    assert "not answer support yet" in second_plan.prompt


def test_workspace_agent_auto_pins_after_unparseable_commit_response(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_unparseable_commit")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"read_clip","args":{"scope":{"segment_id":"seg_0001"},"focus":["buffer"]}}',
        ],
        commit_responses=[
            "I should commit this observation, but here is prose instead.",
            "still not json",
            "{",
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    _assert_forced_final_metadata(result)
    assert workspace.observation_status("obs_0001") == "committed"
    assert workspace.memory_entries()[0].kind == "unverified_capture"
    validation_errors = [
        event
        for event in workspace._read_jsonl_dicts("trace.jsonl")
        if event["type"] == "workspace_commit_validation_error"
    ]
    assert [event["payload"]["attempt"] for event in validation_errors] == [1, 2, 3]


def test_workspace_agent_defers_after_retry_exhaustion_without_pin_material(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_auto_defer")
    registry = _empty_commit_required_registry(workspace)
    backend = EmptyCommitRequiredBackend(
        [
            '{"tool":"read_empty","args":{}}',
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001","writes":{"pinned_anchors":[{"anchor_id":"missing","kind":"asr","source_kind":"audio_fact","excerpt":"missing"}]}}}',
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001","writes":{"pinned_anchors":[{"anchor_id":"missing","kind":"asr","source_kind":"audio_fact","excerpt":"missing"}]}}}',
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001","writes":{"pinned_anchors":[{"anchor_id":"missing","kind":"asr","source_kind":"audio_fact","excerpt":"missing"}]}}}',
        ]
    )
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Why?")

    _assert_forced_final_metadata(result)
    assert workspace.observation_status("obs_0001") == "deferred"
    disposition = workspace.observation_dispositions()[-1]
    assert disposition["until"] == "manual_review"
    assert workspace.memory_entries() == []
