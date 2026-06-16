from __future__ import annotations

import pytest

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget
from visual_coding_agent_harness.agents.runtime.lifecycle import RunContext
from visual_coding_agent_harness.agents.runtime.program_normalizer import ProgramNormalizer
from visual_coding_agent_harness.agents.runtime.state import RoundState, RunState
from visual_coding_agent_harness.registry import ToolRegistry, ToolRuntimeSpec, tool


def _ctx(registry: ToolRegistry) -> RunContext:
    return RunContext(
        workspace=None,
        scene_index=None,
        budget=AgentBudget(max_tool_calls_per_round=2),
        run_state=RunState(question="q", video_path="/v.mp4", question_route="needle_local"),
        round_state=RoundState(round_number=1),
        registry=registry,
    )


def test_program_normalizer_uses_tool_runtime_argument_normalizer() -> None:
    @tool(name="echo", description="Echo a value.")
    def echo(value: str):
        return {"value": value}

    registry = ToolRegistry()
    registry.register(
        ToolRuntimeSpec(
            tool_spec=echo,
            argument_normalizer=lambda _ctx, request: {"value": str(request.arguments["value"]).strip()},
        )
    )

    requests = ProgramNormalizer(registry).normalize([{"tool": "echo", "args": {"value": " ok "}}], ctx=_ctx(registry))

    assert len(requests) == 1
    assert requests[0].tool == "echo"
    assert requests[0].arguments == {"value": "ok"}


def test_pipeline_order_alias_before_normalizer() -> None:
    @tool(name="verify_ledger_answer", description="Verify answer.")
    def verify_ledger_answer(answer: str):
        return {"claim": answer, "confidence": 1.0}

    seen_tools: list[str] = []
    registry = ToolRegistry()
    registry.register(
        ToolRuntimeSpec(
            tool_spec=verify_ledger_answer,
            aliases=("verify",),
            argument_normalizer=lambda _ctx, request: (
                seen_tools.append(request.tool) or {"answer": str(request.arguments["answer"]).strip()}
            ),
        )
    )

    requests = ProgramNormalizer(registry).normalize(
        [{"tool": "verify", "args": {"answer": "  B  "}}],
        ctx=_ctx(registry),
    )

    assert seen_tools == ["verify_ledger_answer"]
    assert requests[0].tool == "verify_ledger_answer"
    assert requests[0].arguments == {"answer": "B"}


def test_program_normalizer_rejects_unknown_tools() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Unknown tool: missing"):
        ProgramNormalizer(registry).normalize([{"tool": "missing", "args": {}}], ctx=_ctx(registry))
