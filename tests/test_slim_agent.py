from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from vcah.agent import VideoAgent
from vcah.model import ScriptedModel
from vcah.types import Frame


class ColorModel(ScriptedModel):
    embedding_dim = 3

    def embed_image(self, paths: Sequence[str]) -> np.ndarray:
        rows = []
        for path in paths:
            with Image.open(path) as image:
                pixel = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
            channel = int(max(range(3), key=lambda idx: pixel[idx]))
            row = [0.0, 0.0, 0.0]
            row[channel] = 1.0
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)

    def embed_text(self, queries: Sequence[str]) -> np.ndarray:
        rows = []
        for query in queries:
            rows.append([0.0, 0.0, 1.0] if "blue" in query.lower() else [1.0, 0.0, 0.0])
        return np.asarray(rows, dtype=np.float32)


def _sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame_{int(start_sec):03d}.jpg"
    Image.new("RGB", (32, 18), color=(20, 40, 230) if start_sec >= 4.0 else (240, 20, 20)).save(path)
    return (Frame(frame_id="fr001", time_sec=start_sec, path=str(path)),)


def test_video_agent_answers_only_with_verified_evidence(tmp_path: Path) -> None:
    model = ColorModel(
        actions=[
            {"type": "search_text", "query": "blue sign"},
            {"type": "focus_clip", "beat_id": "bt00002"},
            {"type": "answer", "answer": "A blue sign appears near the end.", "citations": ["ev_0001"]},
        ]
    )
    agent = VideoAgent(model=model, max_steps=5)

    answer = agent.ask(
        "/videos/demo.mp4",
        "What appears near the end?",
        run_dir=tmp_path,
        duration_sec=8.0,
        asr_cues=({"start": 4.0, "end": 8.0, "text": "a blue sign appears"},),
        range_detector=lambda _video_path, _duration: ((0.0, 4.0), (4.0, 8.0)),
        keyframe_sampler=_sampler,
    )

    assert answer.answer == "A blue sign appears near the end."
    assert answer.citations == ("ev_0001",)
    assert (tmp_path / "run" / "answer.json").exists()
    assert (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").count("\n") == 3


def test_video_agent_rejects_unverified_final_citation(tmp_path: Path) -> None:
    model = ColorModel(actions=[{"type": "answer", "answer": "Unsupported", "citations": ["ev_missing"]}])
    agent = VideoAgent(model=model, max_steps=1)

    answer = agent.ask(
        "/videos/demo.mp4",
        "What happens?",
        run_dir=tmp_path,
        duration_sec=4.0,
        range_detector=lambda _video_path, _duration: ((0.0, 4.0),),
        keyframe_sampler=_sampler,
    )

    assert answer.answer == "Insufficient verified evidence."
    assert answer.citations == ()
