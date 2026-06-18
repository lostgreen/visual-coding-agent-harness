from __future__ import annotations

from types import SimpleNamespace

from visual_coding_agent_harness.agents.budget import AgentBudget
from visual_coding_agent_harness.agents.runtime.hooks.budget import BudgetHook
from visual_coding_agent_harness.agents.runtime.hooks.duplicate_guard import DuplicateGuardHook
from visual_coding_agent_harness.agents.runtime.hooks.evidence_promotion import EvidencePromotionHook
from visual_coding_agent_harness.agents.runtime.hooks.observation_adapter import ObservationAdapterHook
from visual_coding_agent_harness.agents.runtime.hooks.permission import PermissionHook
from visual_coding_agent_harness.agents.runtime.hooks.telemetry import TelemetryHook
from visual_coding_agent_harness.agents.runtime.hooks.zero_yield import ZeroYieldHook
from visual_coding_agent_harness.agents.runtime.lifecycle import RunContext, mark_successful_tool_call
from visual_coding_agent_harness.agents.runtime.program_normalizer import ProgramNormalizer
from visual_coding_agent_harness.agents.runtime.state import RoundState, RunState
from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.contracts import TargetRegistry, TargetSpec
from visual_coding_agent_harness.protocol import ToolRequest, ToolResult
from visual_coding_agent_harness.registry import DuplicateGuardPolicy, ToolError, ToolRegistry, ToolRuntimeSpec, tool
from visual_coding_agent_harness.tools.exploration import build_video_exploration_registry
from visual_coding_agent_harness.tools.runtime_specs import install_video_runtime_specs
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_runtime_spec_default_equiv_to_tool_spec() -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"value": value}

    legacy_registry = ToolRegistry()
    legacy_registry.register(echo)
    runtime_registry = ToolRegistry()
    runtime_registry.register(ToolRuntimeSpec(tool_spec=echo))

    assert legacy_registry.execute("echo", {"value": "ok"}) == {"value": "ok"}
    assert runtime_registry.execute("echo", {"value": "ok"}) == {"value": "ok"}
    assert runtime_registry.get("echo") is echo
    assert runtime_registry.get_runtime_spec("echo").tool_spec is echo


def test_runtime_spec_can_carry_lifecycle_hooks() -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"value": value}

    marker = object()
    registry = ToolRegistry()
    registry.register(ToolRuntimeSpec(tool_spec=echo, argument_normalizer=marker))

    assert registry.get_runtime_spec("echo").argument_normalizer is marker


def test_runtime_spec_commit_required_metadata_defaults_and_survives_updates() -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"value": value}

    registry = ToolRegistry()
    registry.register(ToolRuntimeSpec(tool_spec=echo))

    assert registry.get_runtime_spec("echo").commit_required is False

    registry.replace_runtime_spec("echo", commit_required=True)

    assert registry.get_runtime_spec("echo").commit_required is True


def test_registry_extend_preserves_runtime_spec_metadata() -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"value": value}

    child = ToolRegistry()
    normalizer = object()
    key_builder = object()
    child.register(
        ToolRuntimeSpec(
            tool_spec=echo,
            argument_normalizer=normalizer,
            semantic_key_builder=key_builder,
            duplicate_guard_policy=DuplicateGuardPolicy.ADVISORY,
            commit_required=True,
        )
    )
    parent = ToolRegistry()

    parent.extend(child)

    runtime_spec = parent.get_runtime_spec("echo")
    assert runtime_spec.argument_normalizer is normalizer
    assert runtime_spec.semantic_key_builder is key_builder
    assert runtime_spec.duplicate_guard_policy is DuplicateGuardPolicy.ADVISORY
    assert runtime_spec.commit_required is True


def test_video_exploration_registry_installs_runtime_specs(tmp_path) -> None:
    class StaticBackend(VisionLanguageBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            return BackendResponse(text='{"T1":{"verdict":"supports","cue_ids":["cue_0001"],"quote":"alpha"}}')

    workspace = EvidenceWorkspace.create(tmp_path, "video_runtime_specs")
    video_map = VideoMap(
        video_path="/videos/demo.mp4",
        duration_sec=10.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=10.0,
                low_fps_caption="alpha appears",
                asr_sentences=[{"cue_id": "cue_0001", "start_sec": 0.0, "end_sec": 1.0, "text": "alpha appears"}],
            )
        ],
    )

    registry = build_video_exploration_registry(
        video_map=video_map,
        backend=StaticBackend(),
        workspace=workspace,
    )

    runtime_spec_tools = [
        "bind_asr_claim",
        "target_coverage",
        "ground_question",
        "read_segment_detail",
        "vision_read",
        "verify_segment_anchors",
        "verify_ledger_answer",
        "global_gist",
        "query_context",
        "search_segments",
        "caption_segment",
        "read_timeline_sorted",
        "write_memory",
        "read_workspace",
        "commit_observation",
        "reject_observation",
        "defer_observation",
        "no_commit_needed",
        "read_clip",
        "search",
        "list",
        "verify",
        "synthesize_memory",
        "answer",
    ]
    for tool_name in runtime_spec_tools:
        runtime_spec = registry.get_runtime_spec(tool_name)
        assert runtime_spec.argument_normalizer is not None, tool_name
        assert runtime_spec.semantic_key_builder is not None, tool_name

    assert registry.get_runtime_spec("vision_read").commit_required is True
    assert registry.get_runtime_spec("verify_segment_anchors").commit_required is True
    assert registry.get_runtime_spec("read_clip").commit_required is True
    assert registry.get_runtime_spec("verify").commit_required is True
    assert registry.get_runtime_spec("synthesize_memory").commit_required is False
    assert registry.get_runtime_spec("search_segments").commit_required is False
    assert registry.get_runtime_spec("search").commit_required is False
    assert registry.get_runtime_spec("list").commit_required is False
    assert registry.get_runtime_spec("commit_observation").commit_required is False


def test_video_runtime_spec_required_mode_rejects_missing_core_tool() -> None:
    registry = ToolRegistry()

    try:
        install_video_runtime_specs(registry, required=True)
    except ToolError as exc:
        assert "Missing required runtime spec tools" in str(exc)
        assert "bind_asr_claim" in str(exc)
    else:
        raise AssertionError("required runtime spec installation should fail for missing core tools")


def test_permission_hook_blocks_forbidden_action() -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"value": value}

    registry = ToolRegistry()
    registry.register(ToolRuntimeSpec(tool_spec=echo, permission_predicate=lambda _ctx, _request: False))
    ctx = RunContext(
        workspace=None,
        scene_index=None,
        budget=AgentBudget(max_tool_calls_per_round=2),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
    )

    decision = PermissionHook()(ctx, ToolRequest(tool="echo", arguments={"value": "ok"}))

    assert decision.rejected is True
    assert decision.reason == "permission_denied"


def test_permission_hook_blocks_tool_outside_active_skill_allowed_actions() -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"value": value}

    registry = ToolRegistry()
    registry.register(echo)
    ctx = RunContext(
        workspace=None,
        scene_index=None,
        budget=AgentBudget(max_tool_calls_per_round=2),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
        skill_runtime=SimpleNamespace(
            effective_skill=SimpleNamespace(name="focused_skill", allowed_actions=("other_tool",))
        ),
    )

    decision = PermissionHook()(ctx, ToolRequest(tool="echo", arguments={"value": "ok"}))

    assert decision.rejected is True
    assert decision.reason == "tool_not_allowed_by_active_skill"


def test_duplicate_guard_hook_blocks_repeated_semantic_key() -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"value": value}

    registry = ToolRegistry()
    registry.register(
        ToolRuntimeSpec(
            tool_spec=echo,
            semantic_key_builder=lambda _ctx, request: f"echo:{request.arguments['value']}",
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
        )
    )
    ctx = RunContext(
        workspace=None,
        scene_index=None,
        budget=AgentBudget(max_tool_calls_per_round=2),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
    )
    request = ToolRequest(tool="echo", arguments={"value": "ok"})

    assert DuplicateGuardHook()(ctx, request).rejected is False
    mark_successful_tool_call(ctx, request)
    assert DuplicateGuardHook()(ctx, request).reason == "duplicate_tool_call"


def test_budget_hook_blocks_when_round_tool_budget_exhausted() -> None:
    registry = ToolRegistry()
    round_state = RoundState(round_number=1, issued_tool_calls=1)
    ctx = RunContext(
        workspace=None,
        scene_index=None,
        budget=AgentBudget(max_tool_calls_per_round=1),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=round_state,
        registry=registry,
    )

    decision = BudgetHook()(ctx, ToolRequest(tool="echo", arguments={}))

    assert decision.rejected is True
    assert decision.reason == "round_tool_budget_exhausted"
    assert ctx.issued_tool_calls == 1
    assert ctx.round_state is round_state


def test_observation_adapter_hook_uses_runtime_spec_adapter() -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"value": value}

    registry = ToolRegistry()
    registry.register(ToolRuntimeSpec(tool_spec=echo, observation_adapter=lambda _ctx, _request, _result: ("obs_1",)))
    ctx = RunContext(
        workspace=None,
        scene_index=None,
        budget=AgentBudget(),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
    )

    effects = ObservationAdapterHook()(ctx, ToolRequest(tool="echo"), ToolResult(tool="echo", request_id="", claim="", confidence=0.0))

    assert effects.observation_ids == ("obs_1",)


def test_evidence_promotion_hook_uses_runtime_spec_promoter() -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"value": value}

    events: list[tuple[str, dict[str, object]]] = []
    registry = ToolRegistry()
    registry.register(
        ToolRuntimeSpec(
            tool_spec=echo,
            evidence_promoter=lambda _ctx, _request, _result: (("evidence_promoted", {"tool": "echo"}),),
        )
    )
    ctx = RunContext(
        workspace=None,
        scene_index=None,
        budget=AgentBudget(),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
        record_trace=lambda event_type, payload: events.append((event_type, dict(payload))),
    )

    effects = EvidencePromotionHook()(ctx, ToolRequest(tool="echo"), ToolResult(tool="echo", request_id="", claim="", confidence=0.0))

    assert effects.trace_events == (("evidence_promoted", {"tool": "echo"}),)
    assert events == [("evidence_promoted", {"tool": "echo"})]


def test_zero_yield_and_telemetry_hooks_emit_trace_events() -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"value": value}

    events: list[tuple[str, dict[str, object]]] = []
    registry = ToolRegistry()
    registry.register(echo)
    ctx = RunContext(
        workspace=None,
        scene_index=None,
        budget=AgentBudget(),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
        record_trace=lambda event_type, payload: events.append((event_type, dict(payload))),
    )
    request = ToolRequest(tool="echo", arguments={"value": "ok"})
    result = ToolResult(tool="echo", request_id="", claim="", confidence=0.0, limitations="no evidence found")

    ZeroYieldHook()(ctx, request, result)
    TelemetryHook()(ctx, request, result)

    assert events[0][0] == "zero_yield_tool_result"
    assert events[1][0] == "tool_lifecycle_result"


def test_program_interpreter_can_apply_lifecycle_post_hooks(tmp_path) -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"claim": value, "confidence": 1.0}

    events: list[tuple[str, dict[str, object]]] = []
    registry = ToolRegistry()
    registry.register(echo)
    workspace = EvidenceWorkspace.create(tmp_path, "lifecycle")
    ctx = RunContext(
        workspace=workspace,
        scene_index=None,
        budget=AgentBudget(),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
        record_trace=lambda event_type, payload: events.append((event_type, dict(payload))),
    )

    result = ProgramInterpreter(
        registry,
        workspace,
        lifecycle_context=ctx,
        post_tool_hooks=(TelemetryHook(),),
    ).run([{"tool": "echo", "args": {"value": "ok"}}])

    assert len(result.observation_ids) == 1
    assert events == [("tool_lifecycle_result", {"tool": "echo", "request_id": "1", "claim_chars": 2, "confidence": 1.0})]


def test_program_interpreter_consumes_post_hook_observation_ids_as_additional_observations(tmp_path) -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"claim": value, "confidence": 1.0}

    registry = ToolRegistry()
    workspace = EvidenceWorkspace.create(tmp_path, "post_hook_observation_ids")

    def adapter(_ctx: RunContext, _request: ToolRequest, _result: ToolResult):
        extra = workspace.write_observation(
            tool_name="echo_adapter",
            input_artifacts=[],
            claim="adapter observation",
            confidence=0.9,
            regions=[],
            raw_output={"claim": "adapter observation", "confidence": 0.9},
        )
        return (extra.observation_id,)

    registry.register(ToolRuntimeSpec(tool_spec=echo, observation_adapter=adapter))
    ctx = RunContext(
        workspace=workspace,
        scene_index=None,
        budget=AgentBudget(),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
    )

    result = ProgramInterpreter(
        registry,
        workspace,
        lifecycle_context=ctx,
        post_tool_hooks=(ObservationAdapterHook(),),
    ).run([{"tool": "echo", "args": {"value": "ok"}, "assign": "answer"}])

    assert result.observation_ids == ["obs_0001", "obs_0002"]
    assert result.assignments == {"answer": "obs_0001"}
    assert workspace.get_observation("obs_0001").tool == "echo"
    assert workspace.get_observation("obs_0002").tool == "echo_adapter"


def test_pre_tool_rejection_does_not_write_observation_or_post_hooks(tmp_path) -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"claim": value, "confidence": 1.0}

    events: list[tuple[str, dict[str, object]]] = []
    registry = ToolRegistry()
    registry.register(ToolRuntimeSpec(tool_spec=echo, permission_predicate=lambda _ctx, _request: False))
    workspace = EvidenceWorkspace.create(tmp_path, "pre_reject")
    ctx = RunContext(
        workspace=workspace,
        scene_index=None,
        budget=AgentBudget(max_tool_calls_per_round=2),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
        record_trace=lambda event_type, payload: events.append((event_type, dict(payload))),
    )

    result = ProgramInterpreter(
        registry,
        workspace,
        lifecycle_context=ctx,
        pre_tool_hooks=(PermissionHook(),),
        post_tool_hooks=(TelemetryHook(),),
    ).run([{"tool": "echo", "args": {"value": "ok"}}])

    assert result.observation_ids == []
    assert workspace.observation_count(tool_name="echo") == 0
    assert not events
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert "tool_call_rejected" in trace
    assert "tool_lifecycle_result" not in trace


def test_budget_hook_counts_expanded_dispatches_and_rejections_are_not_observations(tmp_path) -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"claim": value, "confidence": 1.0}

    registry = ToolRegistry()
    registry.register(echo)
    workspace = EvidenceWorkspace.create(tmp_path, "budget_foreach")
    ctx = RunContext(
        workspace=workspace,
        scene_index=None,
        budget=AgentBudget(max_tool_calls_per_round=1),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
    )

    result = ProgramInterpreter(
        registry,
        workspace,
        lifecycle_context=ctx,
        pre_tool_hooks=(BudgetHook(),),
        post_tool_hooks=(TelemetryHook(),),
    ).run(
        [{"tool": "echo", "foreach": "items", "args": {"value": "{item}"}}],
        slots={"items": ["first", "second"]},
    )

    assert result.observation_ids == ["obs_0001"]
    assert ctx.issued_tool_calls == 1
    assert workspace.observation_count(tool_name="echo") == 1
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert "round_tool_budget_exhausted" in trace


def test_duplicate_guard_commits_only_after_successful_execution(tmp_path) -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"claim": value, "confidence": 1.0}

    registry = ToolRegistry()
    registry.register(
        ToolRuntimeSpec(
            tool_spec=echo,
            semantic_key_builder=lambda _ctx, request: f"echo:{request.arguments['value']}",
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
        )
    )
    workspace = EvidenceWorkspace.create(tmp_path, "duplicate_success")
    run_state = RunState(question="q", video_path="/v.mp4", question_route="needle_local")
    ctx = RunContext(
        workspace=workspace,
        scene_index=None,
        budget=AgentBudget(max_tool_calls_per_round=1),
        run_state=run_state,
        round_state=RoundState(round_number=1, issued_tool_calls=1),
        registry=registry,
    )
    request = ToolRequest(tool="echo", arguments={"value": "alpha"}, request_id="1")

    assert DuplicateGuardHook()(ctx, request).rejected is False
    assert BudgetHook()(ctx, request).rejected is True
    assert "echo:alpha" not in ctx.seen_tool_semantic_keys
    assert ctx.seen_tool_semantic_keys is run_state.seen_tool_semantic_keys


def test_run_context_proxies_state_scoped_fields() -> None:
    registry = ToolRegistry()
    run_state = RunState(question="q", video_path="/v.mp4", question_route="needle_local")
    round_state = RoundState(round_number=1)
    ctx = RunContext(
        workspace=None,
        scene_index=None,
        budget=AgentBudget(max_tool_calls_per_round=2),
        run_state=run_state,
        round_state=round_state,
        registry=registry,
    )

    ctx.increment_tool_calls()
    ctx.seen_tool_semantic_keys.add("echo:alpha")

    assert ctx.issued_tool_calls == 1
    assert round_state.issued_tool_calls == 1
    assert ctx.seen_tool_semantic_keys is run_state.seen_tool_semantic_keys
    assert run_state.seen_tool_semantic_keys == {"echo:alpha"}


class RuntimeBackendForUnexpectedCalls(VisionLanguageBackend):
    def generate(self, request: BackendRequest) -> BackendResponse:
        raise AssertionError(f"unexpected backend request: {request.task}")


def test_add_new_tool_no_loop_change(tmp_path) -> None:
    @tool(name="brand_new_probe", description="Probe a synthetic target.")
    def brand_new_probe(target: str):
        return {"claim": f"found {target}", "confidence": 0.9}

    events: list[tuple[str, dict[str, object]]] = []
    registry = ToolRegistry()
    registry.register(
        ToolRuntimeSpec(
            tool_spec=brand_new_probe,
            argument_normalizer=lambda _ctx, request: {"target": str(request.arguments["target"]).strip().lower()},
            semantic_key_builder=lambda _ctx, request: f"brand_new_probe:{request.arguments['target']}",
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
        )
    )
    workspace = EvidenceWorkspace.create(tmp_path, "new_tool_runtime")
    ctx = RunContext(
        workspace=workspace,
        scene_index=None,
        budget=AgentBudget(max_tool_calls_per_round=3),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
        record_trace=lambda event_type, payload: events.append((event_type, dict(payload))),
    )
    requests = ProgramNormalizer(registry).normalize(
        [{"tool": "brand_new_probe", "args": {"target": "  Alpha  "}}],
        ctx=ctx,
    )

    result = ProgramInterpreter(
        registry,
        workspace,
        lifecycle_context=ctx,
        pre_tool_hooks=(PermissionHook(), DuplicateGuardHook(), BudgetHook()),
        post_tool_hooks=(TelemetryHook(),),
    ).run([{"tool": request.tool, "args": request.arguments} for request in requests])

    assert len(result.observation_ids) == 1
    assert workspace.read_observations()[0].raw_output["claim"] == "found alpha"
    assert events == [
        (
            "tool_lifecycle_result",
            {
                "tool": "brand_new_probe",
                "request_id": "1",
                "claim_chars": 11,
                "confidence": 0.9,
                "semantic_key": "brand_new_probe:alpha",
            },
        )
    ]
