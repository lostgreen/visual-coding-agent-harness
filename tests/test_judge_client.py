from __future__ import annotations

from typing import Any

from evaluate.common.judge_client import OpenAICompatibleJudgeClient


class _Response:
    status_code = 200
    headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {"content": "Final Score: 5"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 8},
        }


def _config(*, model: str, temperature: float | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "base": "https://judge.invalid/v1",
        "model": model,
        "api_key": "test-key",
        "max_retries": 0,
    }
    if temperature is not None:
        config["temperature"] = temperature
    return config


def test_gpt5_judge_omits_unconfigured_temperature(monkeypatch: Any) -> None:
    requests: list[dict[str, Any]] = []

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        requests.append(kwargs["json"])
        return _Response()

    monkeypatch.setattr("evaluate.common.judge_client.requests.post", fake_post)
    client = OpenAICompatibleJudgeClient(_config(model="gateway-gpt-5.5"))

    response = client.chat(system_prompt="Judge", user_prompt="Candidate")

    assert response == "Final Score: 5"
    assert requests[0]["max_completion_tokens"] == 4096
    assert "max_tokens" not in requests[0]
    assert "temperature" not in requests[0]
    assert client.temperature is None


def test_judge_preserves_explicit_temperature(monkeypatch: Any) -> None:
    requests: list[dict[str, Any]] = []

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        requests.append(kwargs["json"])
        return _Response()

    monkeypatch.setattr("evaluate.common.judge_client.requests.post", fake_post)
    client = OpenAICompatibleJudgeClient(
        _config(model="gateway-gpt-5.5", temperature=0.0)
    )

    client.chat(system_prompt="Judge", user_prompt="Candidate")

    assert requests[0]["temperature"] == 0.0


def test_non_gpt_judge_defaults_to_deterministic_temperature(monkeypatch: Any) -> None:
    requests: list[dict[str, Any]] = []

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        requests.append(kwargs["json"])
        return _Response()

    monkeypatch.setattr("evaluate.common.judge_client.requests.post", fake_post)
    client = OpenAICompatibleJudgeClient(_config(model="judge-v1"))

    client.chat(system_prompt="Judge", user_prompt="Candidate")

    assert requests[0]["max_tokens"] == 4096
    assert requests[0]["temperature"] == 0.0
