from __future__ import annotations

from visual_coding_agent_harness.backends.capabilities import (
    BACKEND_CAPABILITIES,
    BackendCapabilities,
    SystemMessageSupport,
)


def test_backend_capability_matrix_covers_production_backends() -> None:
    assert set(BACKEND_CAPABILITIES) == {"openai_chat", "qwen_text", "qwen_vl"}
    assert all(isinstance(value, BackendCapabilities) for value in BACKEND_CAPABILITIES.values())


def test_backend_capabilities_record_system_message_strategy() -> None:
    assert BACKEND_CAPABILITIES["openai_chat"].system_message is SystemMessageSupport.NATIVE
    assert BACKEND_CAPABILITIES["openai_chat"].prefix_cache is True
    assert BACKEND_CAPABILITIES["openai_chat"].persistent_conversation is False
    assert BACKEND_CAPABILITIES["openai_chat"].max_context_tokens == 128_000
    assert BACKEND_CAPABILITIES["qwen_text"].system_message is SystemMessageSupport.NATIVE
    assert BACKEND_CAPABILITIES["qwen_vl"].system_message is SystemMessageSupport.NATIVE
