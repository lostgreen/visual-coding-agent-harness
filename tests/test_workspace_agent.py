from pathlib import Path

import pytest

from visual_coding_agent_harness.agents.workspace_agent import (
    FINAL_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    WorkspaceVisualAgent,
    compose_commit_prompt,
    compose_final_prompt,
    compose_plan_prompt,
    _memory_kind_for_verification_verdict,
    _structured_verify_writes,
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
        if request.task == "explore_caption_reasoning":
            return BackendResponse(text='{"mode":"candidate_discovery","support_status":"candidate_only","facts":[]}')
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


class StructuredVerifyFallbackBackend(VisionLanguageBackend):
    def __init__(self, plan_responses: list[str], commit_responses: list[str]) -> None:
        self.plan_responses = plan_responses
        self.commit_responses = commit_responses
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "vision_read":
            return BackendResponse(
                text="Structured verification found shoebox present and ruler not found in the inspected tool-making window.",
                raw={
                    "facts": [
                        {"text": "A shoebox is present in the inspected tool-making window.", "source_kind": "visual_fact", "confidence": 0.92},
                        {"text": "A ruler is not found in the inspected tool-making window.", "source_kind": "visual_fact", "confidence": 0.88},
                    ],
                    "verification_results": [
                        {
                            "target_id": "shoebox",
                            "claim": "A shoebox is used to make the solar eclipse viewer.",
                            "verdict": "supported",
                            "source_kind": "visual_fact",
                            "confidence": 0.92,
                            "rationale": "The shoebox is visible in the local window.",
                        },
                        {
                            "target_id": "ruler",
                            "claim": "A ruler is used to make the solar eclipse viewer.",
                            "verdict": "not_found_in_window",
                            "source_kind": "visual_fact",
                            "confidence": 0.88,
                            "rationale": "No ruler is visible or mentioned in the inspected local window.",
                        },
                    ],
                },
            )
        if request.task == "workspace_commit":
            return BackendResponse(text=self.commit_responses.pop(0))
        if request.task == "workspace_final":
            return BackendResponse(text='{"tool":"answer","args":{"text":"B","citations":["mem_0001"],"confidence":"medium"}}')
        return BackendResponse(text=self.plan_responses.pop(0))


class ExploreCaptionFallbackBackend(VisionLanguageBackend):
    def __init__(self, plan_responses: list[str], commit_responses: list[str]) -> None:
        self.plan_responses = plan_responses
        self.commit_responses = commit_responses
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "explore_caption_reasoning":
            return BackendResponse(
                text=(
                    '{"mode":"caption_fact","support_status":"caption_supported",'
                    '"claim":"Austria-Hungary was seen as a buffer between Russia and Western Europe.",'
                    '"confidence":0.86,'
                    '"facts":[{"claim":"Austria-Hungary was seen as a buffer between Russia and Western Europe.",'
                    '"evidence_text":"Austria-Hungary was seen as a buffer between Russia and Western Europe.",'
                    '"source_kind":"asr","segment_id":"seg_0001","time_range":[0,60],'
                    '"excerpt":"Austria-Hungary was seen as a buffer between Russia and Western Europe.","supports_option":"C"}],'
                    '"anchors":[{"source_kind":"asr","segment_id":"seg_0001","time_range":[0,60],'
                    '"excerpt":"Austria-Hungary was seen as a buffer between Russia and Western Europe."}],'
                    '"condition_match":{"matches_original_question":true,"match_level":"direct",'
                    '"reason":"The ASR directly answers why Austria-Hungary is between Russia and Western Europe."},'
                    '"answer_mapping":{"supports_option":"C"},"needs_visual_verify":false}'
                )
            )
        if request.task == "workspace_commit":
            return BackendResponse(text=self.commit_responses.pop(0))
        if request.task == "workspace_final":
            return BackendResponse(text='{"tool":"answer","args":{"text":"C","citations":["mem_0001"],"confidence":"medium"}}')
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


def test_workspace_agent_parse_action_flags_truncated_tool_json() -> None:
    with pytest.raises(ValueError, match="truncated_or_invalid_action_json"):
        _parse_action('{"tool":"explore","args":{"query":"sequence","targets":[{"target_id":"target_1"},')


def test_workspace_plan_request_allows_long_json_actions(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_plan_generation_budget")
    registry = ToolRegistry()
    backend = ScriptedWorkspaceBackend(['{"tool":"answer","args":{"text":"done","citations":[],"confidence":"low"}}'])
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    action = agent._decide_plan(question="Question: demo", round_number=1, last_tool_result="")

    assert action["tool"] == "answer"
    assert backend.requests[0].task == "workspace_plan"
    assert backend.requests[0].max_new_tokens == 2048


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


def test_workspace_agent_repeated_explore_emits_recovery_observation(tmp_path: Path) -> None:
    @tool(name="explore", description="Explore candidate windows.")
    def explore(query: str, original_question: str = "", answer_options: dict[str, str] | None = None) -> dict[str, object]:
        del original_question, answer_options
        return {
            "claim": "Candidate windows found.",
            "confidence": 0.5,
            "mode": "candidate_discovery",
            "support_status": "candidate_only",
            "query": query,
            "candidate_windows": [
                {"candidate_key": "obs_0001:cand_0001", "segment_id": "seg_0001", "time_range": [0.0, 10.0]}
            ],
        }

    workspace = EvidenceWorkspace.create(tmp_path, "workspace_repeated_explore_hint")
    registry = ToolRegistry()
    registry.register(
        ToolRuntimeSpec(
            tool_spec=explore,
            semantic_key_builder=lambda _ctx, request: f"explore:{request.arguments['query']}",
        )
    )
    backend = ScriptedWorkspaceBackend(
        [
            '{"tool":"explore","args":{"query":"same query"}}',
            '{"tool":"explore","args":{"query":"same query"}}',
            '{"tool":"answer","args":{"text":"done","citations":[],"confidence":"low"}}',
        ]
    )
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=3)

    result = agent.run("Question: demo")

    assert result.answer == "done"
    observations = workspace.read_observations()
    assert len(observations) == 2
    assert observations[-1].raw_output["mode"] == "planner_recovery_hint"
    events = workspace._read_jsonl_dicts("trace.jsonl")
    assert any(event["type"] == "planner_recovery_hint_emitted" for event in events)


def test_workspace_agent_repeated_explore_auto_verifies_after_recovery_hint(tmp_path: Path) -> None:
    @tool(name="explore", description="Explore candidate windows.")
    def explore(query: str, original_question: str = "", answer_options: dict[str, str] | None = None) -> dict[str, object]:
        del original_question, answer_options
        return {
            "claim": "Candidate windows found.",
            "confidence": 0.5,
            "mode": "candidate_discovery",
            "support_status": "candidate_only",
            "query": query,
            "candidate_windows": [
                {"candidate_key": "obs_0001:cand_0001", "segment_id": "seg_0001", "time_range": [0.0, 10.0]}
            ],
        }

    @tool(name="verify_window", description="Verify a pending candidate.")
    def verify_window(candidate_key: str, focus: list[str] | None = None) -> dict[str, object]:
        return {
            "claim": f"Auto-verified {candidate_key}.",
            "confidence": 0.8,
            "mode": "verify_window",
            "worker": "EvidenceVerifier",
            "candidate_key": candidate_key,
            "focus": focus or [],
        }

    workspace = EvidenceWorkspace.create(tmp_path, "workspace_repeated_explore_auto_verify")
    registry = ToolRegistry()
    registry.register(
        ToolRuntimeSpec(
            tool_spec=explore,
            semantic_key_builder=lambda _ctx, request: f"explore:{request.arguments['query']}",
        )
    )
    registry.register(ToolRuntimeSpec(tool_spec=verify_window))
    backend = ScriptedWorkspaceBackend(
        [
            '{"tool":"explore","args":{"query":"same query"}}',
            '{"tool":"explore","args":{"query":"same query"}}',
            '{"tool":"explore","args":{"query":"same query"}}',
        ]
    )
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=3)

    agent.run("Question: demo")
    observations = workspace.read_observations()
    assert observations[-1].raw_output["mode"] == "verify_window"
    assert observations[-1].raw_output["worker"] == "EvidenceVerifier"
    assert observations[-1].raw_output["candidate_key"] == "obs_0001:cand_0001"
    events = workspace._read_jsonl_dicts("trace.jsonl")
    assert any(event["type"] == "planner_recovery_hint_emitted" for event in events)
    assert any(event["type"] == "candidate_auto_verify_triggered" for event in events)


def test_workspace_agent_auto_verifies_when_pending_candidates_saturate(tmp_path: Path) -> None:
    @tool(name="explore", description="Explore candidate windows.")
    def explore(query: str, original_question: str = "", answer_options: dict[str, str] | None = None) -> dict[str, object]:
        del original_question, answer_options
        return {
            "claim": f"Planner repeated explore for {query}.",
            "confidence": 0.1,
            "mode": "candidate_discovery",
            "support_status": "candidate_only",
            "candidate_windows": [],
        }

    @tool(name="verify_window", description="Verify a pending candidate.")
    def verify_window(candidate_key: str, focus: list[str] | None = None) -> dict[str, object]:
        return {
            "claim": f"Auto-verified {candidate_key}.",
            "confidence": 0.8,
            "mode": "verify_window",
            "worker": "EvidenceVerifier",
            "candidate_key": candidate_key,
            "focus": focus or [],
        }

    workspace = EvidenceWorkspace.create(tmp_path, "workspace_pending_saturation_auto_verify")
    workspace.write_observation(
        tool_name="explore",
        claim="Found several pending windows.",
        confidence=0.6,
        raw_output={
            "mode": "candidate_discovery",
            "support_status": "candidate_only",
            "query": "decorations in the scene",
            "candidate_windows": [
                {"candidate_key": "obs_0001:cand_0001", "segment_id": "seg_0001", "time_range": [0.0, 10.0]},
                {"candidate_key": "obs_0001:cand_0002", "segment_id": "seg_0001", "time_range": [10.0, 20.0]},
                {"candidate_key": "obs_0001:cand_0003", "segment_id": "seg_0001", "time_range": [20.0, 30.0]},
            ],
        },
    )
    registry = ToolRegistry()
    registry.register(ToolRuntimeSpec(tool_spec=explore))
    registry.register(ToolRuntimeSpec(tool_spec=verify_window))
    agent = WorkspaceVisualAgent(
        backend=ScriptedWorkspaceBackend([]),
        registry=registry,
        workspace=workspace,
        max_rounds=1,
    )

    observation_ids = agent._execute_plan_action(
        {"tool": "explore", "args": {"query": "try another paraphrase"}},
        question="How many decorations are in the scene?",
        round_number=2,
        seen_tool_semantic_keys=set(),
    )

    observations = workspace.read_observations()
    assert observation_ids == (observations[-1].observation_id,)
    assert observations[-1].tool == "verify_window"
    assert observations[-1].raw_output["candidate_key"] == "obs_0001:cand_0001"
    assert observations[-1].raw_output["focus"] == ["How many decorations are in the scene?"]
    events = workspace._read_jsonl_dicts("trace.jsonl")
    assert any(
        event["type"] == "candidate_auto_verify_triggered"
        and event["payload"]["reason"] == "pending_candidate_saturation"
        for event in events
    )
    assert any(event["type"] == "explore_replaced_by_auto_verify" for event in events)


def test_memory_kind_for_verdict_splits_supported_and_not_found() -> None:
    assert _memory_kind_for_verification_verdict("supported") == "visual_support"
    assert _memory_kind_for_verification_verdict("not_found_in_window") == "local_negative"
    assert _memory_kind_for_verification_verdict("contradicted") == "answer_conflict"
    assert _memory_kind_for_verification_verdict("uncertain") == "verification_uncertain"
    assert _memory_kind_for_verification_verdict("") == "verification_uncertain"


def test_saturation_gate_treats_local_negative_as_no_grounding(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_saturation_local_negative")
    workspace.write_observation(
        tool_name="explore",
        claim="Found several pending windows.",
        confidence=0.6,
        raw_output={
            "mode": "candidate_discovery",
            "support_status": "candidate_only",
            "candidate_windows": [
                {"candidate_key": "obs_0001:cand_0001", "segment_id": "seg_0001", "time_range": [0.0, 10.0]},
                {"candidate_key": "obs_0001:cand_0002", "segment_id": "seg_0001", "time_range": [10.0, 20.0]},
                {"candidate_key": "obs_0001:cand_0003", "segment_id": "seg_0001", "time_range": [20.0, 30.0]},
            ],
        },
    )
    observation = workspace.write_observation(tool_name="verify_window", claim="Not found locally.", confidence=0.8)
    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [{"anchor_id": "anch_neg", "kind": "visual_fact", "source_kind": "visual_fact", "excerpt": "Not found locally."}],
            "memory": [
                {
                    "kind": "local_negative",
                    "claim": "The object was not found in the inspected window.",
                    "anchor_ids": ["anch_neg"],
                    "metadata": {"scope": {"segment_id": "seg_0001", "time_range": [0.0, 10.0]}},
                }
            ],
        },
    )

    assert workspace._pending_saturation_candidate_key() == "obs_0001:cand_0001"


def test_saturation_gate_disarmed_by_positive_support(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_saturation_positive_support")
    workspace.write_observation(
        tool_name="explore",
        claim="Found several pending windows.",
        confidence=0.6,
        raw_output={
            "mode": "candidate_discovery",
            "support_status": "candidate_only",
            "candidate_windows": [
                {"candidate_key": "obs_0001:cand_0001", "segment_id": "seg_0001", "time_range": [0.0, 10.0]},
                {"candidate_key": "obs_0001:cand_0002", "segment_id": "seg_0001", "time_range": [10.0, 20.0]},
                {"candidate_key": "obs_0001:cand_0003", "segment_id": "seg_0001", "time_range": [20.0, 30.0]},
            ],
        },
    )
    observation = workspace.write_observation(tool_name="verify_window", claim="The object is visible in the inspected window.", confidence=0.8)
    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [{"anchor_id": "anch_pos", "kind": "visual_fact", "source_kind": "visual_fact", "excerpt": "visible"}],
            "memory": [{"kind": "visual_support", "claim": "The object is visible in the inspected window.", "anchor_ids": ["anch_pos"]}],
        },
    )

    assert workspace._pending_saturation_candidate_key() is None


def test_saturation_picks_highest_score_pending(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_saturation_score")
    workspace.write_observation(
        tool_name="explore",
        claim="Found scored windows.",
        confidence=0.6,
        raw_output={
            "mode": "candidate_discovery",
            "support_status": "candidate_only",
            "candidate_windows": [
                {"candidate_key": "obs_0001:cand_low", "segment_id": "seg_0001", "time_range": [0.0, 10.0], "score": 0.1},
                {"candidate_key": "obs_0001:cand_high", "segment_id": "seg_0001", "time_range": [10.0, 20.0], "score": 0.8},
                {"candidate_key": "obs_0001:cand_mid", "segment_id": "seg_0001", "time_range": [20.0, 30.0], "score": 0.4},
            ],
        },
    )

    assert workspace._pending_saturation_candidate_key() == "obs_0001:cand_high"


def test_final_prompt_groups_positive_memory_by_option(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_final_option_evidence_map")
    obs = workspace.write_observation(
        tool_name="verify_window",
        claim="steak burgers spread throughout the United States; target_2 was not found in the inspected window.",
        confidence=0.9,
    )
    workspace.commit_observation(
        obs.observation_id,
        writes={
            "pinned_anchors": [
                {"anchor_id": "anch_c", "kind": "visual_fact", "source_kind": "visual_fact", "excerpt": "steak burgers spread"},
                {"anchor_id": "anch_d_neg", "kind": "visual_fact", "source_kind": "visual_fact", "excerpt": "not found"},
            ],
            "memory": [
                {
                    "kind": "visual_support",
                    "claim": "target_4 supported: narration says steak burgers spread throughout the United States.",
                    "supports_option": "C",
                    "anchor_ids": ["anch_c"],
                },
                {
                    "kind": "local_negative",
                    "claim": "target_2 was not found in the inspected window.",
                    "anchor_ids": ["anch_d_neg"],
                    "metadata": {"scope": {"segment_id": "seg_0001", "time_range": [0.0, 10.0]}},
                },
            ]
        },
    )

    prompt = compose_final_prompt(
        question=(
            "Which happened next?\n"
            "A. Beef with spices came from Russia to Germany.\n"
            "B. The steak began to be sandwiched between bread.\n"
            "C. Steak burgers spread throughout the United States.\n"
            "D. The standardization of hamburgers."
        ),
        workspace=workspace,
    )

    assert "# Option Evidence Map" in prompt
    assert "C. Steak burgers spread throughout the United States." in prompt
    assert "mem_0001 [visual_support]" in prompt
    assert "D. The standardization of hamburgers." in prompt
    assert "mem_0002" not in prompt.split("D. The standardization of hamburgers.", 1)[1].split("\n\n", 1)[0]
    assert "local_negative does NOT support absence at global scope" in prompt


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
    assert result.metadata["status"] == "unvalidated_guess"
    assert result.metadata["forced_final"] is True
    assert result.metadata["validated"] is False
    assert result.metadata["attempted_answer"] == "C"
    assert result.metadata["attempted_citations"] == []
    assert result.metadata["validation_error"] == ""
    trace_events = workspace._read_jsonl_dicts("trace.jsonl")
    assert any(event["type"] == "workspace_forced_unvalidated_guess" for event in trace_events)


def test_workspace_agent_forced_final_autocites_valid_support_memory(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_forced_final_autocite")
    observation = workspace.write_observation(
        tool_name="explore",
        claim="Caption directly supports option C.",
        confidence=0.8,
        raw_output={"mode": "caption_fact", "support_status": "caption_supported"},
    )
    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": "anch_caption_c",
                    "kind": "dense_caption",
                    "source_kind": "dense_caption",
                    "excerpt": "Caption directly supports option C.",
                }
            ],
            "memory": [
                {
                    "kind": "caption_support",
                    "claim": "Caption directly supports option C.",
                    "anchor_ids": ["anch_caption_c"],
                    "supports_option": "C",
                    "confidence": "medium",
                    "metadata": {
                        "source_tool": "explore",
                        "mode": "caption_fact",
                        "support_status": "caption_supported",
                        "condition_match": {"matches_original_question": True, "match_level": "direct"},
                    },
                }
            ],
        },
    )
    backend = ScriptedWorkspaceBackend(
        [
            '{"tool":"read_workspace","args":{"section":"memory"}}',
            '{"tool":"answer","args":{"text":"C","citations":[],"confidence":"low"}}',
        ]
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Question: demo\nA. one\nB. two\nC. three\nD. four")

    assert result.answer == "C"
    assert result.citations == ("mem_0001",)
    assert result.metadata is not None
    assert result.metadata["validated"] is True
    assert result.metadata["auto_cited"] is True


def test_structured_verify_writes_false_option_negative_as_answer_support(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_false_option_truth_table")
    raw_output = {
        "claim": "The inspected frame contradicts the white-hat option.",
        "mode": "verify_window",
        "produced_anchors": [
            {
                "anchor_id": "anch_option_c",
                "source_kind": "visual_fact",
                "modality": "visual",
                "excerpt": "The inspected frame contradicts the white-hat option.",
            }
        ],
        "verification_results": [
            {
                "target_id": "option_C",
                "claim": "Ferdinand was wearing a white hat.",
                "verdict": "contradicted",
                "supports_option": "C",
                "answer_polarity": "select_false_option",
                "anchor_ids": ["anch_option_c"],
                "scope": {"segment_id": "seg_0001", "time_range": [0.0, 20.0]},
                "confidence": 0.92,
            }
        ],
    }
    observation = workspace.write_observation(
        tool_name="verify_window",
        claim="The inspected frame contradicts the white-hat option.",
        confidence=0.9,
        raw_output=raw_output,
    )
    writes = _structured_verify_writes(
        raw_output,
        anchors=[
            {
                "anchor_id": "anch_option_c",
                "source_kind": "visual_fact",
                "modality": "visual",
                "excerpt": "The inspected frame contradicts the white-hat option.",
            }
        ],
        reason="test",
    )

    workspace.commit_observation(observation.observation_id, writes=writes)
    memory = workspace.memory_entries()[0]
    assert memory.kind == "answer_conflict_resolved"
    assert memory.supports_option == "C"
    assert memory.metadata["option_truth_status"] == "contradicted"
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=ScriptedWorkspaceBackend([]), workspace=workspace)
    accepted = registry.execute("answer", {"text": "C", "citations": ["mem_0001"], "confidence": "medium"})
    assert accepted["accepted"] is True


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


def test_workspace_agent_runs_verify_window_commit_answer(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_verify_window_commit")
    sampled = []

    def fake_frame_sampler(video_path: str, start_sec: float, end_sec: float, max_frames: int) -> list[str]:
        sampled.append((video_path, start_sec, end_sec, max_frames))
        return ["/frames/demo/00005.jpg"]

    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[5,20],"evidence_mode":"visual","focus":["buffer"]}}',
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
                    "evidence_obs_ids": ["obs_0001"],
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
        frame_sampler=fake_frame_sampler,
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
    assert workspace.observation_status("obs_0001") == "committed"
    assert workspace.memory_entries()[0].kind == "answer_support"
    assert sampled == [("/videos/demo.mp4", 5.0, 20.0, 30)]
    assert [request.task for request in backend.requests] == [
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

    IndexRefiner(backend=backend, frame_sampler=fake_frame_sampler).refine(
        store,
        parent_segment_id="seg_0001",
        requested_start_sec=5.0,
        requested_end_sec=20.0,
        resolution="medium",
        focus=["buffer"],
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
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],"focus":["buffer"]}}',
        ],
        commit_responses=[
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60]}}',
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60]}}',
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60]}}',
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
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],"focus":["buffer"]}}',
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
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],"focus":["buffer"]}}',
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
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],"focus":["buffer"]}}',
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
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],"focus":["buffer"]}}',
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


def test_prompts_keep_generic_planning_and_scope_local_negatives(tmp_path: Path) -> None:
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

    assert "whole-video or main-idea" not in PLAN_SYSTEM_PROMPT
    assert "early, middle, and late" not in PLAN_SYSTEM_PROMPT
    assert "whole-video or main-idea" not in plan_prompt
    assert "early, middle, and late segment cards" not in plan_prompt
    assert "local_negative" in commit_prompt
    assert "global_negation_allowed" in commit_prompt
    assert "scope" in commit_prompt
    assert "cannot support a final answer" in commit_prompt
    assert "answer_support" in commit_prompt
    assert 'mode is "caption_fact" or "mixed"' in commit_prompt
    assert "condition_match.matches_original_question is true" in commit_prompt
    assert "Reject only when" in commit_prompt


def test_compose_plan_prompt_includes_pending_saturation_hard_rule(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_pending_saturation_prompt")
    workspace.write_observation(
        tool_name="explore",
        claim="Found several pending windows.",
        confidence=0.6,
        raw_output={
            "mode": "candidate_discovery",
            "support_status": "candidate_only",
            "query": "objects in the scene",
            "candidate_windows": [
                {"candidate_key": "obs_0001:cand_0001", "segment_id": "seg_0001", "time_range": [0.0, 10.0]},
                {"candidate_key": "obs_0001:cand_0002", "segment_id": "seg_0001", "time_range": [10.0, 20.0]},
                {"candidate_key": "obs_0001:cand_0003", "segment_id": "seg_0001", "time_range": [20.0, 30.0]},
            ],
        },
    )

    prompt = compose_plan_prompt(question="How many objects are in the scene?", workspace=workspace, video_map=_video_map())

    assert "HARD RULE" in prompt
    assert "Pending Candidate Windows count >= 3" in prompt
    assert "MUST be verify_window" in prompt


def test_render_plan_view_marks_pending_saturation_recommendation_as_must(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_pending_saturation_view")
    workspace.write_observation(
        tool_name="explore",
        claim="Found several pending windows.",
        confidence=0.6,
        raw_output={
            "mode": "candidate_discovery",
            "support_status": "candidate_only",
            "query": "objects in the scene",
            "candidate_windows": [
                {"candidate_key": "obs_0001:cand_0001", "segment_id": "seg_0001", "time_range": [0.0, 10.0]},
                {"candidate_key": "obs_0001:cand_0002", "segment_id": "seg_0001", "time_range": [10.0, 20.0]},
                {"candidate_key": "obs_0001:cand_0003", "segment_id": "seg_0001", "time_range": [20.0, 30.0]},
            ],
        },
    )

    plan_view = workspace.render_plan_view(question="How many objects are in the scene?", video_map=_video_map())

    assert "MUST verify_window before exploring again" in plan_view
    assert "obs_0001:cand_0001" in plan_view


def test_compose_plan_prompt_blocks_uncited_answers_without_memory(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_plan_protocol")

    prompt = compose_plan_prompt(
        question="What's the main idea?",
        workspace=workspace,
        last_tool_result="answer rejected: final answer requires at least one planner-authored memory citation",
    )

    assert "Available plan tools" in PLAN_SYSTEM_PROMPT
    assert "explore" in PLAN_SYSTEM_PROMPT
    assert "verify_window" in PLAN_SYSTEM_PROMPT
    assert "Every answer call must include" in PLAN_SYSTEM_PROMPT
    assert '"text":"D"' not in PLAN_SYSTEM_PROMPT
    for removed_tool in ("scan_segment", "read_clip", "read_segment", "standalone verify", "search", "list"):
        assert removed_tool not in PLAN_SYSTEM_PROMPT
        assert removed_tool not in prompt
    assert "Use Segment Cards as the starting navigation state" in prompt
    assert '{"tool":"explore"' in prompt
    assert '{"tool":"verify_window"' in prompt
    assert '"segment_id":"seg_0001","time_range":[20.0,40.0]' in prompt
    assert "sweep unexplored regions" in PLAN_SYSTEM_PROMPT
    assert "different part of the segment" in prompt
    assert "synthesize_memory is unavailable until committed support memory exists" in prompt
    assert "duplicate_tool_call" in prompt
    assert "do not repeat the same semantic request" in prompt
    assert "candidate_key" in prompt
    assert "Query Framing Policy" in PLAN_SYSTEM_PROMPT
    assert "write the query as a verification question" in PLAN_SYSTEM_PROMPT
    assert "The original question defines what counts as evidence" in PLAN_SYSTEM_PROMPT
    assert "Check whether option X answers the original question condition" in PLAN_SYSTEM_PROMPT
    assert "Do not begin by copying terms from only one answer option" in PLAN_SYSTEM_PROMPT
    assert "question-centered condition" in prompt
    assert '"text":"D"' not in prompt
    assert '"text":"A"' not in prompt


def test_forced_final_prompt_does_not_default_to_option_a(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_forced_final_prompt")

    prompt = compose_final_prompt(question="Which option is correct?", workspace=workspace, video_map=_video_map())

    assert '"text":"A"' not in FINAL_SYSTEM_PROMPT
    assert '"text":"A"' not in prompt
    assert "best supported option" in prompt
    assert "do not default" in prompt
    assert "citations: []" in prompt


def test_workspace_agent_exports_downloadable_round_log(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_round_log")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],"focus":["buffer"]}}',
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
            '{"tool":"read_workspace","args":{"section":"memory"}}',
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


def test_workspace_agent_commits_verify_window_with_evidence_excerpt(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_verify_commit")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],"focus":["buffer"]}}',
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
                    "anchor_id":"clip_anch_seg_0001_00000000_00060000",
                    "kind":"asr",
                    "source_kind":"audio_fact",
                    "excerpt":"Austria-Hungary was seen as a buffer between Russia and Western Europe."
                  }],
                  "memory": [{
                    "kind":"answer_support",
                    "claim":"ASR says Austria-Hungary was the buffer.",
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
    assert [request.task for request in backend.requests] == [
        "workspace_plan",
        "vision_read",
        "workspace_commit",
        "workspace_plan",
    ]


def test_workspace_agent_acknowledges_workspace_read_without_commit(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_read_workspace_no_commit")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"read_workspace","args":{"section":"memory"}}',
        ],
        commit_responses=[],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    _assert_forced_final_metadata(result)
    assert workspace.observation_status("obs_0001") == "auto_acknowledged"
    assert [request.task for request in backend.requests] == ["workspace_plan", "workspace_final"]


def test_workspace_agent_auto_pins_retrieval_candidate_after_bad_commit_schema(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_search_candidate_auto_pin")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"explore","args":{"query":"buffer Russia","modalities":["asr"],"top_k":1}}',
        ],
        commit_responses=[
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001","writes":"anch_explore_obs_0001_cand_0001"}}',
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
    assert "verify_window" in workspace.render_plan_view(question="Why?")
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
            '{"tool":"explore","args":{"query":"buffer Russia","modalities":["asr"],"top_k":1}}',
        ],
        commit_responses=[
            (
                '{"tool":"commit_observation","args":{'
                '"anchor_id":"anch_explore_obs_0001_cand_0001",'
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
    assert memory[0].metadata["recommended_next_tool"] == "verify_window"
    assert "verify_window candidate anch_explore_obs_0001_cand_0001" in workspace.render_plan_view(question="Why?")


def test_workspace_agent_auto_pins_after_commit_retry_exhaustion(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_auto_pin")
    backend = ScriptedWorkspaceV2Backend(
        plan_responses=[
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],"focus":["buffer"]}}',
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


def test_workspace_agent_auto_pins_structured_verify_results_after_commit_parse_failure(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_structured_verify_auto_pin")
    backend = StructuredVerifyFallbackBackend(
        plan_responses=[
            (
                '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],'
                '"checks":[{"target_id":"shoebox","claim":"A shoebox is used to make the solar eclipse viewer."},'
                '{"target_id":"ruler","claim":"A ruler is used to make the solar eclipse viewer."}]}}'
            ),
        ],
        commit_responses=[
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001"}}',
            'not json',
            '{"tool":"no_commit_needed","args":{"observation_id":"obs_0001","comment":"bad alias"}}',
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Which item is not used to make the solar eclipse observing tool?")

    assert result.answer == "B"
    assert workspace.observation_status("obs_0001") == "committed"
    memories = workspace.memory_entries()
    assert [memory.kind for memory in memories] == ["visual_support", "local_negative"]
    assert {memory.metadata["target_id"] for memory in memories} == {"shoebox", "ruler"}
    assert {memory.metadata["verdict"] for memory in memories} == {"supported", "not_found_in_window"}
    assert all(memory.metadata["source_tool"] == "verify_window" for memory in memories)
    assert not any(memory.kind == "unverified_capture" for memory in memories)
    assert any(event["type"] == "deterministic_verify_commit" for event in workspace._read_jsonl_dicts("trace.jsonl"))


def test_workspace_agent_deterministically_commits_structured_verify_before_llm_commit(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_structured_verify_deterministic")
    backend = StructuredVerifyFallbackBackend(
        plan_responses=[
            (
                '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],'
                '"checks":[{"target_id":"shoebox","claim":"A shoebox is used to make the solar eclipse viewer."},'
                '{"target_id":"ruler","claim":"A ruler is used to make the solar eclipse viewer."}]}}'
            ),
        ],
        commit_responses=['{"tool":"commit_observation","args":{"observation_id":"obs_0001"}}'],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Which item is not used to make the solar eclipse observing tool?")

    assert result.answer == "B"
    assert "workspace_commit" not in [request.task for request in backend.requests]
    assert workspace.observation_status("obs_0001") == "committed"
    memories = workspace.memory_entries()
    assert [memory.metadata["verdict"] for memory in memories] == ["supported", "not_found_in_window"]
    ruler_memory = next(memory for memory in memories if memory.metadata["target_id"] == "ruler")
    assert ruler_memory.metadata["claim_scope"] == "window_negative"
    assert ruler_memory.metadata["global_answer_support"] is False
    assert any(event["type"] == "deterministic_verify_commit" for event in workspace._read_jsonl_dicts("trace.jsonl"))


def test_workspace_agent_auto_pins_caption_fact_after_commit_parse_failure(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_agent_caption_fact_auto_pin")
    backend = ExploreCaptionFallbackBackend(
        plan_responses=['{"tool":"explore","args":{"query":"buffer between Russia and Western Europe","modalities":["asr"],"top_k":1}}'],
        commit_responses=[
            "not json",
            '{"tool":"commit_observation","args":{"observation_id":"obs_0001"}}',
            '{"tool":"no_commit_needed","args":{"observation_id":"obs_0001","comment":"bad alias"}}',
        ],
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    agent = WorkspaceVisualAgent(backend=backend, registry=registry, workspace=workspace, max_rounds=1)

    result = agent.run("Why was Austria-Hungary shown between Russia and Western Europe?")

    assert result.answer == "C"
    assert workspace.observation_status("obs_0001") == "committed"
    memories = workspace.memory_entries()
    assert [memory.kind for memory in memories] == ["caption_support"]
    assert memories[0].supports_option == "C"
    assert memories[0].metadata["source_tool"] == "explore"
    assert memories[0].metadata["support_status"] == "caption_supported"
    assert memories[0].metadata["auto_pinned"] is True
    assert memories[0].metadata["requires_visual_verify"] is False
    assert not any(memory.kind == "unverified_capture" for memory in memories)
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
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],"focus":["buffer"]}}',
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
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],"focus":["buffer"]}}',
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
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],"focus":["buffer"]}}',
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
            '{"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[0,60],"focus":["buffer"]}}',
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
