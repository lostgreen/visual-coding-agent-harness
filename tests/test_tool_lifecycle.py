from __future__ import annotations

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget
from visual_coding_agent_harness.agents.runtime.hooks.budget import BudgetHook
from visual_coding_agent_harness.agents.runtime.hooks.duplicate_guard import DuplicateGuardHook
from visual_coding_agent_harness.agents.runtime.hooks.evidence_promotion import EvidencePromotionHook
from visual_coding_agent_harness.agents.runtime.hooks.observation_adapter import ObservationAdapterHook
from visual_coding_agent_harness.agents.runtime.hooks.permission import PermissionHook
from visual_coding_agent_harness.agents.runtime.hooks.telemetry import TelemetryHook
from visual_coding_agent_harness.agents.runtime.hooks.zero_yield import ZeroYieldHook
from visual_coding_agent_harness.agents.runtime.lifecycle import RunContext
from visual_coding_agent_harness.agents.runtime.program_normalizer import ProgramNormalizer
from visual_coding_agent_harness.agents.runtime.state import RoundState, RunState
from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.protocol import ToolRequest, ToolResult
from visual_coding_agent_harness.registry import DuplicateGuardPolicy, ToolRegistry, ToolRuntimeSpec, tool
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
    assert DuplicateGuardHook()(ctx, request).reason == "duplicate_tool_call"


def test_budget_hook_blocks_when_round_tool_budget_exhausted() -> None:
    registry = ToolRegistry()
    ctx = RunContext(
        workspace=None,
        scene_index=None,
        budget=AgentBudget(max_tool_calls_per_round=1),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
        issued_tool_calls=1,
    )

    decision = BudgetHook()(ctx, ToolRequest(tool="echo", arguments={}))

    assert decision.rejected is True
    assert decision.reason == "round_tool_budget_exhausted"


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
    assert events == [("tool_lifecycle_result", {"tool": "echo", "args": {"value": "ok"}, "claim": "ok", "confidence": 1.0})]


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
            {"tool": "brand_new_probe", "args": {"target": "alpha"}, "claim": "found alpha", "confidence": 0.9},
        )
    ]
