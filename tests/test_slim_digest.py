from __future__ import annotations

from vcah.index import ColdIndex, TextIndex, VisualIndex
from vcah.model import ModelClient
from vcah.types import Beat, Chapter, IndexDiagnostics


def _cold_index() -> ColdIndex:
    text_index = TextIndex()
    text_index.add("bt00001", "red shipyard history", modality="asr")
    text_index.add("bt00002", "blue bridge closeup", modality="asr")
    model = ModelClient()
    return ColdIndex(
        video_path="/videos/demo.mp4",
        duration_sec=20.0,
        chapters=(
            Chapter("ch01", 0.0, 10.0, ("bt00001",)),
            Chapter("ch02", 10.0, 20.0, ("bt00002",)),
        ),
        beats=(
            Beat("bt00001", "ch01", 0.0, 10.0, "red.jpg", asr_text="red shipyard history"),
            Beat("bt00002", "ch02", 10.0, 20.0, "blue.jpg", asr_text="blue bridge closeup"),
        ),
        text_index=text_index,
        visual_index=VisualIndex(model),
        diagnostics=IndexDiagnostics(20.0, 2, 2, 10.0, 10.0, 0, 0.0, "test", "fast"),
    )


def test_timeline_digest_without_query_matches_legacy_output() -> None:
    cold = _cold_index()

    assert cold.timeline_digest() == (
        "2 chapters, 2 beats\n"
        "ch01 [00:00-00:10] 1 beats\n"
        "ch02 [00:10-00:20] 1 beats"
    )


def test_timeline_digest_marks_query_hits_hot_and_visited_seen() -> None:
    cold = _cold_index()

    digest = cold.timeline_digest(query="blue bridge", visited_beats=("bt00002",))
    lines = digest.splitlines()

    assert lines[1].startswith("[hot][seen] ch02")
    assert lines[2].startswith("[cold] ch01")
