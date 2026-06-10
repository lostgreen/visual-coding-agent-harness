from __future__ import annotations

from visual_coding_agent_harness.backends.base import BackendRequest
from visual_coding_agent_harness.backends.qwen_vl import _message_content


def test_qwen_vl_backend_serializes_frame_requests_as_video_frame_list() -> None:
    content = _message_content(
        BackendRequest(
            task="caption_segment",
            prompt="Describe the selected frames.",
            media_type="video",
            frames=("/frames/demo/frame_000000003.jpg", "/frames/demo/frame_000000004.jpg"),
            metadata={"nframes": 64, "max_pixels": 151200},
        )
    )

    assert content == [
        {
            "type": "video",
            "video": ["/frames/demo/frame_000000003.jpg", "/frames/demo/frame_000000004.jpg"],
            "nframes": 64,
            "max_pixels": 151200,
        },
        {"type": "text", "text": "Describe the selected frames."},
    ]
