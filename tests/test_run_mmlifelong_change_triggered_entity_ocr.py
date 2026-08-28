from argparse import Namespace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import threading

from vcah.virtual_video import VirtualVideoSegment


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "run_mmlifelong_change_triggered_entity_ocr.py"
)
SPEC = importlib.util.spec_from_file_location(
    "change_triggered_entity_ocr", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _segment() -> VirtualVideoSegment:
    return VirtualVideoSegment(
        segment_id="segment-a",
        source_video_id="video-a",
        source_path="/existing/video.mp4",
        source_start_sec=10.0,
        source_end_sec=20.0,
        virtual_start_sec=100.0,
        virtual_end_sec=110.0,
    )


def _rows() -> tuple[dict, ...]:
    return (
        {
            "segment_id": "segment-a",
            "source_video_id": "video-a",
            "tier0_frame_index": 1,
            "source_time_sec": 11.0,
            "virtual_time_sec": 101.0,
            "selection_reason": "coverage_bin_peak",
        },
        {
            "segment_id": "segment-a",
            "source_video_id": "video-a",
            "tier0_frame_index": 4,
            "source_time_sec": 14.0,
            "virtual_time_sec": 104.0,
            "selection_reason": "ranked_change_peak",
        },
    )


def _args(*, resume: bool = False) -> Namespace:
    return Namespace(
        batch_size=1,
        workers=2,
        resume=resume,
        expected_model="model-a",
        max_image_edge=640,
        ffmpeg_executable="/opt/test/ffmpeg",
    )


def test_selected_frame_materialization_uses_only_requested_indexes(
    tmp_path: Path, monkeypatch
) -> None:
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        output_root = Path(command[-1]).parent
        for index in range(2):
            (output_root / f"frame_{index + 1:06d}.jpg").write_bytes(b"jpeg")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    paths = runner._materialize_selected_segment_frames(
        _segment(),
        _rows(),
        out_dir=tmp_path / "frames",
        max_image_edge=640,
        ffmpeg_executable="/opt/test/ffmpeg",
    )
    assert len(paths) == 2
    assert seen["command"][0] == "/opt/test/ffmpeg"
    filter_value = seen["command"][seen["command"].index("-vf") + 1]
    assert "eq(n\\,1)+eq(n\\,4)" in filter_value
    assert "scale=640:640" in filter_value
    assert seen["kwargs"]["capture_output"] is True


def test_worker_clients_are_created_inside_executor_threads(
    tmp_path: Path, monkeypatch
) -> None:
    main_thread = threading.get_ident()
    client_threads = []

    def fake_materialize(segment, rows, *, out_dir, max_image_edge, ffmpeg_executable):
        del segment, max_image_edge, ffmpeg_executable
        out_dir.mkdir(parents=True)
        paths = []
        for index, _ in enumerate(rows):
            path = out_dir / f"frame_{index + 1:06d}.jpg"
            path.write_bytes(b"jpeg")
            paths.append(path)
        return tuple(paths)

    def fake_batch(batch, *, client_for_worker, **kwargs):
        del batch, kwargs
        client_for_worker()
        return {"status": "success", "parse_status": "success"}

    def client_factory():
        client_threads.append(threading.get_ident())
        return object()

    monkeypatch.setattr(
        runner, "_materialize_selected_segment_frames", fake_materialize
    )
    monkeypatch.setattr(runner, "_run_batch", fake_batch)
    results = runner._run_segment_batches(
        _segment(),
        _rows(),
        arm="a2_change",
        result_root=tmp_path / "results",
        temp_root=tmp_path / "temporary",
        client_for_worker=client_factory,
        args=_args(),
    )
    assert len(results) == 2
    assert client_threads
    assert all(identifier != main_thread for identifier in client_threads)
    assert not any((tmp_path / "temporary").rglob("*.jpg"))


def test_resume_reuse_removes_stale_temporary_frames(
    tmp_path: Path,
) -> None:
    row = _rows()[0]
    batch = (row,)
    result_path = runner._batch_result_path(
        tmp_path / "results", arm="a1_uniform", batch=batch
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "status": "success",
                "parse_status": "success",
                "actual_model": "model-a",
                "selection_digest": runner._batch_selection_digest(batch),
                "frame_labels": [runner._frame_label(row)],
            }
        ),
        encoding="utf-8",
    )
    stale_root = tmp_path / "temporary" / runner._safe_name("segment-a")
    stale_root.mkdir(parents=True)
    (stale_root / "frame_000001.jpg").write_bytes(b"stale")
    results = runner._run_segment_batches(
        _segment(),
        batch,
        arm="a1_uniform",
        result_root=tmp_path / "results",
        temp_root=tmp_path / "temporary",
        client_for_worker=lambda: object(),
        args=_args(resume=True),
    )
    assert results[0]["resume_reused_success"] is True
    assert not stale_root.exists()
