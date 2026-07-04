from __future__ import annotations

from pathlib import Path

from PIL import Image

from visual_coding_agent_harness.video.pipeline import shots_to_beats


def _image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 18), color=color).save(path)
    return str(path)


def test_shots_to_beats_merges_adjacent_similar_keyframes(tmp_path: Path) -> None:
    keyframes = (
        _image(tmp_path / "red_1.jpg", (240, 20, 20)),
        _image(tmp_path / "red_2.jpg", (241, 21, 20)),
        _image(tmp_path / "red_3.jpg", (239, 20, 21)),
        _image(tmp_path / "green.jpg", (20, 220, 30)),
        _image(tmp_path / "blue.jpg", (20, 40, 230)),
    )
    shots = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 25.0))

    groups = shots_to_beats(shots, keyframes, sim_threshold=0.95, max_beat_sec=60.0)

    assert groups == ((0, 1, 2), (3,), (4,))


def test_shots_to_beats_respects_max_beat_duration(tmp_path: Path) -> None:
    keyframes = tuple(_image(tmp_path / f"red_{idx}.jpg", (240, 20, 20)) for idx in range(3))
    shots = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0))

    groups = shots_to_beats(shots, keyframes, sim_threshold=0.95, max_beat_sec=20.0)

    assert groups == ((0, 1), (2,))
