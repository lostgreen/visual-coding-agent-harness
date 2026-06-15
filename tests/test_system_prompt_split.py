from __future__ import annotations

from visual_coding_agent_harness.agents.context_budget import default_context_budget_allocator
from visual_coding_agent_harness.agents.iterative_agent import AgentBudget
from visual_coding_agent_harness.agents.prompt_stack import build_replanning_prompt, compose_planner_prompts
from visual_coding_agent_harness.backends.base import BackendRequest
from visual_coding_agent_harness.backends.openai_chat import _messages_for_request
from visual_coding_agent_harness.backends.qwen_text import _text_messages_for_request
from visual_coding_agent_harness.backends.qwen_vl import _messages_for_request as _qwen_vl_messages_for_request
from visual_coding_agent_harness.video_index import fixed_window_scene_index


def _prompt_kwargs() -> dict[str, object]:
    return {
        "question": "What is visible?",
        "scene_index": fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=10.0, window_sec=5.0),
        "ledger_text": "# Compact Evidence Context\n(none)",
        "round_number": 1,
        "budget": AgentBudget(max_rounds=2),
        "allocator": default_context_budget_allocator(total_budget_tokens=12000),
        "active_skill": "general_exploration@v1",
    }


def test_backend_request_default_unchanged() -> None:
    request = BackendRequest(task="replan", prompt="hello")

    assert request.system_prompt == ""


def test_split_disabled_byte_equal_to_legacy() -> None:
    legacy_prompt, legacy_report = build_replanning_prompt(**_prompt_kwargs())
    pair = compose_planner_prompts(prompt_role_split_enabled=False, **_prompt_kwargs())

    assert pair.system_prompt == ""
    assert pair.user_prompt == legacy_prompt
    assert pair.context_report == legacy_report


def test_split_enabled_concat_equals_legacy() -> None:
    legacy_prompt, _legacy_report = build_replanning_prompt(**_prompt_kwargs())
    pair = compose_planner_prompts(prompt_role_split_enabled=True, **_prompt_kwargs())

    assert pair.system_prompt
    assert pair.user_prompt
    assert f"{pair.system_prompt}\n\n{pair.user_prompt}" == legacy_prompt


def test_openai_chat_native_system_prompt_routing() -> None:
    messages = _messages_for_request(BackendRequest(task="answer_from_evidence", prompt="User body.", system_prompt="System body."))

    assert messages[0] == {"role": "system", "content": "System body."}
    assert messages[1]["role"] == "system"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"].startswith("User body.")
    assert "Return only one parseable JSON object" in messages[2]["content"]


def test_qwen_vl_native_system_prompt_routing() -> None:
    messages = _qwen_vl_messages_for_request(
        BackendRequest(task="vision_read", prompt="User body.", system_prompt="System body.", media_path="/tmp/frame.png", media_type="image")
    )

    assert messages[0] == {"role": "system", "content": [{"type": "text", "text": "System body."}]}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0] == {"type": "image", "image": "/tmp/frame.png"}
    assert messages[1]["content"][-1] == {"type": "text", "text": "User body."}


def test_qwen_text_native_system_prompt_routing() -> None:
    messages = _text_messages_for_request(BackendRequest(task="answer_from_evidence", prompt="User body.", system_prompt="System body."))

    assert messages == [
        {"role": "system", "content": "System body."},
        {"role": "user", "content": "User body."},
    ]
