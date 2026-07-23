from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from vcah.caption_schema import (
    CaptionAnchorV1,
    CaptionChunkV1,
    CaptionPassageV1,
)
from vcah.caption_store import (
    CaptionStore,
    activate_caption_config,
    resolve_caption_passages_path,
)
from tools.reprocess_caption_timestamps import reprocess_caption_timestamps


def _chunk(caption_id: str = "cap_000001") -> CaptionChunkV1:
    return CaptionChunkV1(
        caption_id=caption_id,
        subset="game",
        virtual_start_sec=0.0,
        virtual_end_sec=10.0,
        source_segments=("seg_0001",),
        wall_clock_begin=None,
        wall_clock_end=None,
        text_raw="[00:00:02] The player opens a door.",
        text_normalized="[00:00:02] The player opens a door.",
        timestamp_anchors=(CaptionAnchorV1("[00:00:02]", 2.0, 2.0, "The player opens a door."),),
        model="fixture-vlm",
        provider="fixture",
        prompt_digest="prompt",
        generation_config_digest="generation",
        source_manifest_digest="manifest",
        created_at="2026-07-21T00:00:00+00:00",
    )


def _passage(caption_id: str = "cap_000001") -> CaptionPassageV1:
    return CaptionPassageV1(
        passage_id=f"{caption_id}:p0000",
        caption_id=caption_id,
        text="The player opens a door.",
        virtual_start_sec=2.0,
        virtual_end_sec=10.0,
        anchor_virtual_sec=2.0,
        ordinal=0,
        metadata={"interval_precision": "anchor"},
    )


def test_caption_store_resume_keeps_success_chunk_unique(tmp_path: Path) -> None:
    store = CaptionStore(tmp_path, "digest-a")
    store.prepare("key-a", {"virtual_start_sec": 0.0})
    assert store.begin("key-a") is True
    store.mark_success("key-a", _chunk(), (_passage(),))

    resumed = CaptionStore(tmp_path, "digest-a")

    assert resumed.recover_interrupted() == 0
    assert resumed.begin("key-a") is False
    assert len(resumed.successful_chunks()) == 1
    assert len(resumed.successful_passages()) == 1
    assert len(resumed.chunks_path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(resumed.passages_path.read_text(encoding="utf-8").splitlines()) == 1


def test_caption_store_recovers_running_and_persists_failure_state(tmp_path: Path) -> None:
    store = CaptionStore(tmp_path, "digest-b")
    store.prepare("key-b", {"virtual_start_sec": 10.0})
    store.begin("key-b")

    resumed = CaptionStore(tmp_path, "digest-b")

    assert resumed.recover_interrupted() == 1
    assert resumed.record("key-b")["status"] == "pending"
    resumed.begin("key-b")
    resumed.mark_failed("key-b", "temporary failure")
    record = CaptionStore(tmp_path, "digest-b").record("key-b")
    assert record["status"] == "failed"
    assert record["attempt_count"] == 2
    assert record["last_error"] == "temporary failure"


def test_caption_store_can_defer_large_jsonl_exports_until_run_end(tmp_path: Path) -> None:
    store = CaptionStore(tmp_path, "digest-deferred", eager_exports=False)
    store.prepare("key-a", {"virtual_start_sec": 0.0})
    store.begin("key-a")
    store.mark_success("key-a", _chunk(), (_passage(),))

    assert not store.chunks_path.exists()
    assert not store.passages_path.exists()
    assert len(CaptionStore(tmp_path, "digest-deferred").successful_chunks()) == 1

    store.flush_exports()

    assert len(store.chunks_path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(store.passages_path.read_text(encoding="utf-8").splitlines()) == 1


def test_reprocess_caption_timestamps_rebuilds_anchors_and_passages(tmp_path: Path) -> None:
    store = CaptionStore(tmp_path, "digest-reprocess", eager_exports=False)
    base = _chunk()
    text = "[00:00:00] The player arrives. [00:08:00] The player leaves."
    malformed = CaptionChunkV1(
        **{
            **asdict(base),
            "text_raw": text,
            "text_normalized": text,
            "timestamp_anchors": (),
        }
    )
    store.prepare("key-a", {"virtual_start_sec": 0.0})
    store.begin("key-a")
    store.mark_success("key-a", malformed, ())

    summary = reprocess_caption_timestamps(tmp_path, "digest-reprocess")
    repaired = CaptionStore(tmp_path, "digest-reprocess").successful_chunks()[0]

    assert summary["reprocessed_chunks"] == 1
    assert summary["parse_status_counts"] == {
        "strict": 1,
        "filtered_invalid": 0,
        "chunk_fallback": 0,
    }
    assert [anchor.local_sec for anchor in repaired.timestamp_anchors] == [0.0, 8.0]
    assert repaired.metadata["timestamp_repair_count"] == 1
    assert len(CaptionStore(tmp_path, "digest-reprocess").successful_passages()) == 2


def test_active_caption_config_resolves_multiple_caches(tmp_path: Path) -> None:
    for digest in ("digest-a", "digest-b"):
        store = CaptionStore(tmp_path, digest)
        store.prepare(f"key-{digest}", {"virtual_start_sec": 0.0})
        store.begin(f"key-{digest}")
        store.mark_success(f"key-{digest}", _chunk(f"cap-{digest}"), (_passage(f"cap-{digest}"),))

    with pytest.raises(ValueError, match="Multiple caption passage caches"):
        resolve_caption_passages_path(tmp_path)

    active_path = activate_caption_config(
        tmp_path,
        "digest-b",
        metadata={"index_mode": "hybrid"},
    )
    passages_path, digest = resolve_caption_passages_path(tmp_path)
    active = json.loads(active_path.read_text(encoding="utf-8"))

    assert digest == "digest-b"
    assert passages_path.name == "passages.digest-b.jsonl"
    assert active["config_digest"] == "digest-b"
    assert active["metadata"]["index_mode"] == "hybrid"
