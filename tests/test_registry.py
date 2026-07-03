from __future__ import annotations

import pytest

from visual_coding_agent_harness.core.registry import ToolError, ToolRegistry, ToolRuntimeSpec, tool


def test_resolve_alias() -> None:
    @tool(name="verify_answer", description="Verify answer evidence.")
    def verify_answer(answer: str):
        return {"claim": answer, "confidence": 1.0}

    registry = ToolRegistry()
    registry.register(ToolRuntimeSpec(tool_spec=verify_answer, aliases=("verify",)))

    assert registry.resolve_alias("verify") == "verify_answer"
    assert registry.resolve_alias("verify_answer") == "verify_answer"
    assert registry.resolve_alias("unknown_tool") == "unknown_tool"


def test_duplicate_alias_is_rejected() -> None:
    @tool(name="first", description="First tool.")
    def first():
        return {"claim": "first"}

    @tool(name="second", description="Second tool.")
    def second():
        return {"claim": "second"}

    registry = ToolRegistry()
    registry.register(ToolRuntimeSpec(tool_spec=first, aliases=("shared",)))

    with pytest.raises(ToolError, match="Tool alias already registered"):
        registry.register(ToolRuntimeSpec(tool_spec=second, aliases=("shared",)))
