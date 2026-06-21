from __future__ import annotations

import io
import json
import urllib.error

import pytest

from visual_coding_agent_harness.backends.base import BackendRequest


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_openai_chat_backend_sends_thinking_budget_and_returns_structured_json(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "private chain of thought",
                            "content": (
                                "The user wants a planner step.\n"
                                '{"status":"continue","program":[],"rationale":"budgeted"}'
                            ),
                        }
                    }
                ],
                "usage": {"completion_tokens": 19},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    from visual_coding_agent_harness.backends.openai_chat import OpenAIChatTextBackend

    backend = OpenAIChatTextBackend(
        api_base="http://planner-host:8000/v1",
        model="Qwen3.5-9B",
        api_key="EMPTY",
        timeout=12.5,
        thinking_token_budget=512,
        enable_thinking=True,
    )

    response = backend.generate(
        BackendRequest(
            task="replan",
            prompt="Return planner JSON only.",
            max_new_tokens=4096,
            temperature=0.0,
        )
    )

    assert captured["url"] == "http://planner-host:8000/v1/chat/completions"
    assert captured["timeout"] == 12.5
    assert captured["headers"]["Authorization"] == "Bearer EMPTY"
    body = captured["body"]
    assert body["model"] == "Qwen3.5-9B"
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0.0
    assert body["thinking_token_budget"] == 512
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["messages"][0]["role"] == "system"
    assert "Do not explain your reasoning" in body["messages"][0]["content"]
    assert "Return only one parseable JSON object" in body["messages"][1]["content"]
    assert response.text == '{"status":"continue","program":[],"rationale":"budgeted"}'
    assert response.raw["backend"] == "openai_chat"
    assert response.raw["task"] == "replan"
    assert response.raw["model"] == "Qwen3.5-9B"
    assert response.raw["api_base"] == "http://planner-host:8000/v1"
    assert response.raw["reasoning_content_present"] is True


def test_openai_chat_backend_supports_request_metadata_override(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({"choices": [{"message": {"content": '{"answer":"A"}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    from visual_coding_agent_harness.backends.openai_chat import OpenAIChatTextBackend

    backend = OpenAIChatTextBackend(
        api_base="http://planner-host:8000",
        model="Qwen3.5-9B",
        thinking_token_budget=1024,
        enable_thinking=True,
    )

    backend.generate(
        BackendRequest(
            task="answer_from_evidence",
            prompt="Answer.",
            metadata={
                "thinking_token_budget": 128,
                "chat_template_kwargs": {"enable_thinking": False},
                "extra_body": {"top_k": 20},
            },
        )
    )

    assert captured["body"]["thinking_token_budget"] == 128
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["body"]["top_k"] == 20


def test_openai_chat_backend_azure_reads_endpoint_key_and_deployment_from_env(monkeypatch):
    captured = {}
    monkeypatch.setenv("ENDPOINT_URL", "https://example-resource.openai.azure.com")
    monkeypatch.setenv("DEPLOYMENT_NAME", "gpt-prod-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-from-env")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({"choices": [{"message": {"content": '{"answer":"A"}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    from visual_coding_agent_harness.backends.openai_chat import OpenAIChatTextBackend

    backend = OpenAIChatTextBackend(api_type="azure", timeout=30.0)

    response = backend.generate(
        BackendRequest(
            task="answer_from_evidence",
            prompt="Answer from evidence.",
            max_new_tokens=2048,
            temperature=0.0,
        )
    )

    assert captured["url"] == (
        "https://example-resource.openai.azure.com/openai/deployments/"
        "gpt-prod-deployment/chat/completions?api-version=2025-01-01-preview"
    )
    assert captured["timeout"] == 30.0
    assert captured["headers"]["api-key"] == "secret-from-env"
    assert "authorization" not in captured["headers"]
    body = captured["body"]
    assert "model" not in body
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 2048
    assert body["messages"][0]["role"] == "developer"
    assert body["messages"][0]["content"][0]["type"] == "text"
    assert "Return only one parseable JSON object" in body["messages"][1]["content"][0]["text"]
    assert response.text == '{"answer":"A"}'
    assert response.raw["api_type"] == "azure_openai"
    assert response.raw["api_base_set"] is True
    assert response.raw["api_base_env"] == "ENDPOINT_URL"
    assert response.raw["model"] == "gpt-prod-deployment"
    assert "api_base" not in response.raw


def test_openai_chat_backend_azure_can_send_sampled_frames_when_media_enabled(monkeypatch, tmp_path):
    captured = {}
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"fake-image-bytes")
    monkeypatch.setenv("ENDPOINT_URL", "https://example-resource.openai.azure.com")
    monkeypatch.setenv("DEPLOYMENT_NAME", "gpt-prod-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-from-env")

    def fake_urlopen(request, timeout):
        del timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({"choices": [{"message": {"content": "visible fact"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    from visual_coding_agent_harness.backends.openai_chat import OpenAIChatTextBackend

    backend = OpenAIChatTextBackend(api_type="azure", allow_media=True)
    response = backend.generate(
        BackendRequest(
            task="vision_read",
            prompt="Read this frame.",
            media_type="video",
            frames=[str(frame_path)],
            max_new_tokens=128,
        )
    )

    content = captured["body"]["messages"][1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "Read this frame."}
    assert response.text == "visible fact"


def test_openai_chat_backend_media_video_requires_sampled_frames_when_enabled(monkeypatch):
    monkeypatch.setenv("ENDPOINT_URL", "https://example-resource.openai.azure.com")
    monkeypatch.setenv("DEPLOYMENT_NAME", "gpt-prod-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-from-env")

    from visual_coding_agent_harness.backends.openai_chat import OpenAIChatTextBackend

    backend = OpenAIChatTextBackend(api_type="azure", allow_media=True)

    with pytest.raises(ValueError, match="sampled frame paths"):
        backend.generate(
            BackendRequest(
                task="vision_read",
                prompt="Read this clip.",
                media_path="/tmp/video.mp4",
                media_type="video",
            )
        )


def test_openai_chat_backend_rejects_media_requests(monkeypatch):
    from visual_coding_agent_harness.backends.openai_chat import OpenAIChatTextBackend

    backend = OpenAIChatTextBackend(api_base="http://planner-host:8000/v1", model="Qwen3.5-9B")

    with pytest.raises(ValueError, match="media support is disabled"):
        backend.generate(BackendRequest(task="caption_segment", prompt="Describe.", media_path="/tmp/video.mp4"))

    with pytest.raises(ValueError, match="media support is disabled"):
        backend.generate(BackendRequest(task="caption_frames", prompt="Describe.", frames=["frame-1.png"]))


def test_openai_chat_backend_reports_http_error(monkeypatch):
    def fake_urlopen(request, timeout):
        del request, timeout
        raise urllib.error.HTTPError(
            "http://planner-host:8000/v1/chat/completions",
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"bad thinking budget"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    from visual_coding_agent_harness.backends.openai_chat import OpenAIChatTextBackend

    backend = OpenAIChatTextBackend(api_base="http://planner-host:8000/v1", model="Qwen3.5-9B")

    with pytest.raises(RuntimeError, match="bad thinking budget"):
        backend.generate(BackendRequest(task="replan", prompt="Return JSON."))


def test_openai_chat_backend_falls_back_when_vllm_rejects_thinking_budget(monkeypatch):
    bodies = []

    def fake_urlopen(request, timeout):
        del timeout
        body = json.loads(request.data.decode("utf-8"))
        bodies.append(body)
        if len(bodies) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=io.BytesIO(b'{"error":{"message":"Extra inputs are not permitted: thinking_token_budget"}}'),
            )
        return FakeHTTPResponse(
            {"choices": [{"message": {"content": '{"status":"continue","program":[]}'}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    from visual_coding_agent_harness.backends.openai_chat import OpenAIChatTextBackend

    backend = OpenAIChatTextBackend(
        api_base="http://planner-host:8000/v1",
        model="Qwen3.5-9B",
        thinking_token_budget=512,
        enable_thinking=True,
    )

    response = backend.generate(BackendRequest(task="replan", prompt="Return JSON."))

    assert bodies[0]["thinking_token_budget"] == 512
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "thinking_token_budget" not in bodies[1]
    assert bodies[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert response.text == '{"status":"continue","program":[]}'
    assert response.raw["thinking_budget_fallback"] is True
