from visual_coding_agent_harness.agents.contracts import resolve_nframes


def test_default_returns_128():
    assert resolve_nframes(None) == (128, "default_contract")


def test_user_override():
    assert resolve_nframes(64) == (64, "user_override")


def test_clamp_to_max():
    assert resolve_nframes(999) == (256, "user_override")


def test_cap_dominates():
    assert resolve_nframes(128, tool_cap=32) == (32, "tool_capability_cap")
