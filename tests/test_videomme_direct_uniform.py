from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from PIL import Image
import pytest

from vcah.direct_baseline import (
    build_direct_prompt,
    format_timestamped_asr,
    materialize_uniform_frames,
    parse_direct_response,
    render_contact_sheets,
    request_direct_answer,
    summarize_results,
    uniform_midpoint_times,
)


def _load_direct_runner() -> Any:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        path = tools_dir / "run_videomme_direct_uniform.py"
        spec = importlib.util.spec_from_file_location("videomme_direct_uniform_runner", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(tools_dir))


_direct_runner = _load_direct_runner()


def test_uniform_midpoint_times_cover_all_bins() -> None:
    assert uniform_midpoint_times(100.0, 4) == (12.5, 37.5, 62.5, 87.5)


def test_materialize_uniform_frames_writes_exact_manifest(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_runner(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(list(command))
        pattern = str(command[-1])
        for index in range(1, 5):
            Image.new("RGB", (32, 18), color=(index * 20, 40, 80)).save(pattern.replace("%04d", f"{index:04d}"))
        return subprocess.CompletedProcess(command, 0, "", "")

    rows = materialize_uniform_frames(
        video_path=tmp_path / "source.mp4",
        duration_sec=100.0,
        out_dir=tmp_path / "frames",
        frame_count=4,
        max_image_edge=512,
        runner=fake_runner,
    )

    assert len(calls) == 1
    assert "-ss" in calls[0]
    assert calls[0][calls[0].index("-ss") + 1] == "12.500000"
    assert len(rows) == 4
    assert [row["frame_index"] for row in rows] == [1, 2, 3, 4]
    assert [row["time_sec"] for row in rows] == [12.5, 37.5, 62.5, 87.5]
    manifest_rows = [json.loads(line) for line in (tmp_path / "frames" / "frame_manifest.jsonl").read_text().splitlines()]
    assert manifest_rows == list(rows)


def test_materialize_uniform_frames_recovers_missing_tail_frame(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_runner(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(list(command))
        if "%04d" in str(command[-1]):
            pattern = str(command[-1])
            for index in range(1, 4):
                Image.new("RGB", (32, 18), color=(index * 30, 50, 90)).save(
                    pattern.replace("%04d", f"{index:04d}")
                )
        else:
            Image.new("RGB", (32, 18), color=(120, 50, 90)).save(command[-1])
        return subprocess.CompletedProcess(command, 0, "", "")

    rows = materialize_uniform_frames(
        video_path=tmp_path / "source.mp4",
        duration_sec=100.0,
        out_dir=tmp_path / "frames",
        frame_count=4,
        max_image_edge=512,
        runner=fake_runner,
    )

    assert len(calls) == 2
    assert calls[1][calls[1].index("-ss") + 1] == "87.500000"
    assert Path(rows[-1]["path"]).name == "frame_0004.jpg"
    assert rows[-1]["time_sec"] == 87.5


def test_format_timestamped_asr_keeps_complete_source_times() -> None:
    text = format_timestamped_asr(
        (
            {"start": 1.0, "end": 2.0, "text": "hello"},
            {"start": 3661.25, "end": 3662.5, "text": "later cue"},
        )
    )

    assert "[00:00:01.000-00:00:02.000] hello" in text
    assert "[01:01:01.250-01:01:02.500] later cue" in text


def test_build_direct_prompt_includes_frame_map_asr_and_cot_boundary() -> None:
    prompt = build_direct_prompt(
        question="What happens?",
        options={"A": "First", "B": "Second"},
        frame_rows=(
            {"frame_index": 1, "time_sec": 2.5, "path": "frame_0001.jpg"},
            {"frame_index": 2, "time_sec": 7.5, "path": "frame_0002.jpg"},
        ),
        asr_text="[00:00:02.000-00:00:03.000] hello",
    )

    assert "F0001=00:00:02.500" in prompt
    assert "F0002=00:00:07.500" in prompt
    assert "[00:00:02.000-00:00:03.000] hello" in prompt
    assert "Do not provide hidden chain-of-thought" in prompt


def test_parse_direct_response_accepts_fenced_json_and_structured_answer() -> None:
    parsed = parse_direct_response(
        '```json\n{"answer":{"option":"B","text":"Second"},"rationale":"visible evidence",'
        '"evidence":[{"frame_index":4,"time_sec":8.5}]}\n```'
    )

    assert parsed["answer"] == "B"
    assert parsed["rationale"] == "visible evidence"
    assert parsed["evidence"] == ({"frame_index": 4, "time_sec": 8.5},)


def test_render_contact_sheets_preserves_all_512_frames_without_padding(tmp_path: Path) -> None:
    frame_paths = []
    for index in range(1, 513):
        path = tmp_path / "frames" / f"frame_{index:04d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 9), color=(index % 255, 30, 60)).save(path)
        frame_paths.append(path)

    sheets = render_contact_sheets(tuple(frame_paths), tmp_path / "sheets")

    assert len(sheets) == 32
    assert all(path.exists() for path in sheets)
    with Image.open(sheets[0]) as image:
        assert image.size == (640, 360)


class ScriptedApi:
    def __init__(self, responses: Sequence[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, prompt: str, *, image_paths: Sequence[str] = (), max_tokens: int = 900) -> str:
        self.calls.append({"prompt": prompt, "image_paths": tuple(image_paths), "max_tokens": max_tokens})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_request_direct_answer_falls_back_for_request_shape_error(tmp_path: Path) -> None:
    frame_paths = []
    for index in range(1, 513):
        path = tmp_path / "frames" / f"frame_{index:04d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 9), color=(40, index % 255, 80)).save(path)
        frame_paths.append(path)
    api = ScriptedApi((RuntimeError("HTTP 413: request too large"), '{"answer":"A","rationale":"fallback"}'))

    parsed, raw, input_mode, submitted_paths = request_direct_answer(
        api=api,
        prompt="question",
        frame_paths=tuple(frame_paths),
        sheet_dir=tmp_path / "sheets",
    )

    assert parsed["answer"] == "A"
    assert raw.startswith("{")
    assert input_mode == "contact_sheets_32"
    assert len(submitted_paths) == 32
    assert [len(call["image_paths"]) for call in api.calls] == [512, 32]


def test_request_direct_answer_does_not_fallback_for_auth_error(tmp_path: Path) -> None:
    api = ScriptedApi((RuntimeError("HTTP 401: unauthorized"),))

    with pytest.raises(RuntimeError, match="401"):
        request_direct_answer(
            api=api,
            prompt="question",
            frame_paths=(tmp_path / "frame.jpg",),
            sheet_dir=tmp_path / "sheets",
        )

    assert len(api.calls) == 1


def test_summarize_results_reports_accuracy_modes_latency_and_failures() -> None:
    summary = summarize_results(
        (
            {"case_id": "a", "correct": True, "input_mode": "images_512", "latency_sec": 2.0, "error": ""},
            {"case_id": "b", "correct": False, "input_mode": "contact_sheets_32", "latency_sec": 4.0, "error": ""},
            {"case_id": "c", "correct": False, "input_mode": "", "latency_sec": 0.0, "error": "failed"},
        )
    )

    assert summary["case_count"] == 3
    assert summary["correct"] == 1
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["successful_cases"] == 2
    assert summary["mean_latency_sec"] == 3.0
    assert summary["input_modes"] == {"contact_sheets_32": 1, "images_512": 1}
    assert summary["failures"] == 1


def test_run_direct_case_writes_artifacts_and_scores_answer(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "video").mkdir(parents=True)
    (dataset_root / "subtitle").mkdir(parents=True)
    (dataset_root / "video" / "video-1.mp4").write_bytes(b"video")
    (dataset_root / "subtitle" / "video-1.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nvisible answer\n",
        encoding="utf-8",
    )
    row = {
        "question_id": "case-1",
        "videoID": "video-1",
        "question": "Which option is visible?",
        "options": ["A. First", "B. Second"],
        "answer": "B",
    }

    def fake_materializer(**kwargs: Any) -> tuple[dict[str, Any], ...]:
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, time_sec in enumerate((2.5, 7.5), start=1):
            path = out_dir / f"frame_{index:04d}.jpg"
            Image.new("RGB", (16, 9), color=(30, 50, 70)).save(path)
            rows.append({"frame_index": index, "time_sec": time_sec, "path": str(path)})
        (out_dir / "frame_manifest.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in rows),
            encoding="utf-8",
        )
        return tuple(rows)

    def fake_requester(**kwargs: Any) -> tuple[dict[str, Any], str, str, tuple[str, ...]]:
        del kwargs
        return (
            {"answer": "B", "rationale": "The second option is visible.", "evidence": ({"frame_index": 2},)},
            '{"answer":"B"}',
            "images_2",
            ("frame_0001.jpg", "frame_0002.jpg"),
        )

    ticks = iter((10.0, 12.5))
    result = _direct_runner.run_direct_case(
        row=row,
        dataset_root=dataset_root,
        out_root=tmp_path / "out",
        api=object(),
        frame_count=2,
        max_image_edge=512,
        duration_probe=lambda path: 10.0,
        frame_materializer=fake_materializer,
        requester=fake_requester,
        clock=lambda: next(ticks),
    )

    assert result["case_id"] == "case-1"
    assert result["answer"] == "B"
    assert result["correct"] is True
    assert result["latency_sec"] == 2.5
    assert result["input_mode"] == "images_2"
    case_dir = tmp_path / "out" / "cases" / "case-1"
    assert "visible answer" in (case_dir / "asr_prompt.txt").read_text()
    assert json.loads((case_dir / "result.json").read_text())["correct"] is True
    metadata = json.loads((case_dir / "request_metadata.json").read_text())
    assert metadata["frame_count"] == 2
    assert "api_key" not in metadata


def test_build_group_summary_preserves_case_order() -> None:
    rows = (
        {"case_id": "case-c", "correct": False, "input_mode": "images_512", "latency_sec": 2.0, "error": ""},
        {"case_id": "case-a", "correct": True, "input_mode": "images_512", "latency_sec": 1.0, "error": ""},
    )

    summary = _direct_runner.build_group_summary(rows, group_id="hard-v3")

    assert summary["group_id"] == "hard-v3"
    assert [row["case_id"] for row in summary["cases"]] == ["case-c", "case-a"]
    assert summary["correct"] == 1
