from visual_coding_agent_harness.agents.iterative_agent import AgentBudget


def test_budget_only_exposes_round_limits() -> None:
    budget = AgentBudget(max_rounds=20, max_tool_calls_per_round=2)

    assert budget.max_rounds == 20
    assert budget.max_tool_calls_per_round == 2
    assert budget.reserve_final_round is True

    field_names = {field.name for field in budget.__dataclass_fields__.values()}
    assert "cheap_tool_budget" not in field_names
    assert "expensive_tool_budget" not in field_names
    assert "verifier_tool_budget" not in field_names
    assert "free_exploration" not in field_names


def test_budget_snapshot_block_omits_per_class_budgets() -> None:
    from visual_coding_agent_harness.agents.prompt_stack import _budget_snapshot_block

    body = _budget_snapshot_block(
        round_number=3,
        budget=AgentBudget(max_rounds=20, max_tool_calls_per_round=2),
        final_round_reserved=False,
    )

    assert "Round: 3/20" in body
    assert "Request at most 2 new tool call(s)" in body
    assert "Remaining tool budgets" not in body
    assert "cheap=" not in body
    assert "expensive=" not in body
    assert "free exploration mode" not in body


def test_tool_class_module_constants_removed() -> None:
    import visual_coding_agent_harness.agents.iterative_agent as agent_mod

    assert not hasattr(agent_mod, "_CHEAP_TOOLS")
    assert not hasattr(agent_mod, "_EXPENSIVE_TOOLS")
    assert not hasattr(agent_mod, "_VERIFIER_TOOLS")
    assert not hasattr(agent_mod, "_TOOL_CLASSES")
    assert not hasattr(agent_mod, "_tool_budget_available")
