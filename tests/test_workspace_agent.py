from pathlib import Path

import pytest

from visual_coding_agent_harness.agents.workspace_agent import (
    PLAN_SYSTEM_PROMPT,
    WorkspaceVisualAgent,
    compose_commit_prompt,
    compose_plan_prompt,
    _parse_action,
)
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.evals.videomme.workspace_round_log import export_workspace_round_log
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


class EmptyCommitRequiredBackend(VisionLanguageBackend):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
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

    assert result.metadata == {"reason": "max_rounds_reached"}
    assert workspace.observation_status("obs_0001") == "committed"
    assert workspace.memory_entries()[0].kind == "unverified_capture"
    assert [request.task for request in backend.requests] == [
        "workspace_plan",
        "vision_read",
        "workspace_commit",
        "workspace_commit",
        "workspace_commit",
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


def test_compose_plan_prompt_blocks_uncited_answers_without_memory(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_plan_protocol")

    prompt = compose_plan_prompt(
        question="What's the main idea?",
        workspace=workspace,
        last_tool_result="answer rejected: final answer requires at least one planner-authored memory citation",
    )

    assert "Available plan tools" in PLAN_SYSTEM_PROMPT
    assert "Every answer call must include" in PLAN_SYSTEM_PROMPT
    assert "If Committed Memory is empty" in prompt
    assert "the next tool must be read_clip, search, list, read_workspace, verify, or synthesize_memory" in prompt
    assert '{"tool":"read_clip"' in prompt
    assert '{"tool":"answer","args":{"text":"D","citations":["mem_0001"],"confidence":"high"}}' in prompt


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

    assert result.metadata == {"reason": "max_rounds_reached"}
    assert workspace.observation_status("obs_0001") == "acknowledged"
    assert [request.task for request in backend.requests] == ["workspace_plan"]


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

    assert result.metadata == {"reason": "max_rounds_reached"}
    assert workspace.observation_status("obs_0001") == "committed"
    memory = workspace.memory_entries()
    assert memory[0].kind == "unverified_capture"
    assert memory[0].metadata["auto_pinned"] is True
    assert any(event["type"] == "commit_auto_pinned" for event in workspace._read_jsonl_dicts("trace.jsonl"))


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

    assert result.metadata == {"reason": "max_rounds_reached"}
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

    assert result.metadata == {"reason": "max_rounds_reached"}
    assert workspace.observation_status("obs_0001") == "deferred"
    assert workspace.memory_entries() == []


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

    assert result.metadata == {"reason": "max_rounds_reached"}
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

    assert result.metadata == {"reason": "max_rounds_reached"}
    assert workspace.observation_status("obs_0001") == "deferred"
    disposition = workspace.observation_dispositions()[-1]
    assert disposition["until"] == "manual_review"
    assert workspace.memory_entries() == []
