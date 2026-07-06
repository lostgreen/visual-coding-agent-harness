from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from PIL import Image

from vcah.agent import VideoAgent
from vcah.model import ScriptedModel
from vcah.types import Frame


class VisualAttestModel(ScriptedModel):
    def __init__(self) -> None:
        super().__init__(
            actions=[
                {"type": "inspect_window", "start_sec": 0, "end_sec": 4, "modalities": ["frames"]},
                {"type": "answer", "answer": "A blue sign is visible.", "citations": ["ev_0001"]},
            ]
        )
        self.prompts: list[str] = []
        self.paths: list[tuple[str, ...]] = []

    def attest(self, image_paths: Sequence[str], prompt: str) -> tuple[str, ...]:
        self.paths.append(tuple(image_paths))
        self.prompts.append(prompt)
        return ("A blue sign is visible.", "White text is visible on the sign.")


def _sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame_{int(start_sec):03d}.jpg"
    Image.new("RGB", (32, 18), color=(20, 40, 230)).save(path)
    return (Frame(frame_id="fr001", time_sec=start_sec, path=str(path)),)


def test_inspect_window_frames_creates_one_visual_evidence_per_atomic_observation(tmp_path: Path) -> None:
    model = VisualAttestModel()
    agent = VideoAgent(model=model, max_steps=3)

    answer = agent.ask(
        "/videos/demo.mp4",
        "What is visible?",
        run_dir=tmp_path,
        duration_sec=4.0,
        range_detector=lambda _video_path, _duration: ((0.0, 4.0),),
        keyframe_sampler=_sampler,
    )

    evidence_lines = [
        json.loads(line)
        for line in (tmp_path / "run" / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert answer.citations == ("ev_0001",)
    assert [record["evidence_id"] for record in evidence_lines] == ["ev_0001", "ev_0002"]
    assert all(record["modality"] == "visual" for record in evidence_lines)
    assert all(record["frame_refs"] for record in evidence_lines)
    assert "A." not in model.prompts[0]
