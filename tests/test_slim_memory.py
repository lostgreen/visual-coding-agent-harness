from __future__ import annotations

import json
from pathlib import Path

from vcah.memory import AgentMemory, EvidenceStore, TraceStore
from vcah.types import EvidenceRecord, ToolAction, ToolResult


def test_evidence_store_accepts_only_known_evidence_ids(tmp_path: Path) -> None:
    store = EvidenceStore.empty(tmp_path / "evidence.jsonl")
    record = EvidenceRecord(
        evidence_id="ev_0001",
        beat_id="bt00002",
        start_sec=10.0,
        end_sec=20.0,
        modality="asr",
        pointer="bt00002@10.000-20.000",
        verbatim="A blue sign is visible.",
        claim="A blue sign is visible.",
    )

    store.add(record)

    assert store.valid(("ev_0001",))
    assert not store.valid(("ev_9999",))
    payload = json.loads((tmp_path / "evidence.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert payload["evidence_id"] == "ev_0001"
    assert payload["modality"] == "asr"
    assert payload["verbatim"] == "A blue sign is visible."


def test_memory_and_trace_write_minimal_run_artifacts(tmp_path: Path) -> None:
    memory = AgentMemory.empty(tmp_path / "memory.json")
    trace = TraceStore(tmp_path / "trace.jsonl")

    memory.record_result(ToolResult(tool="search_text", beat_ids=("bt00001",), text="found transcript"))
    trace.append(
        ToolAction(type="search_text", query="shipyard"),
        ToolResult(tool="search_text", beat_ids=("bt00001",), text="found transcript"),
    )

    memory.save()

    assert json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))["visited_beats"] == ["bt00001"]
    assert json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0])["action"]["type"] == "search_text"


def test_memory_tracks_last_search_hits(tmp_path: Path) -> None:
    memory = AgentMemory.empty(tmp_path / "memory.json")

    memory.record_result(ToolResult(tool="search_visual", beat_ids=("bt00003", "bt00002"), text="visual hits"))
    memory.save()

    payload = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
    assert memory.last_hits == ["bt00003", "bt00002"]
    assert payload["last_hits"] == ["bt00003", "bt00002"]
    assert "last_hits=bt00003,bt00002" in memory.digest()


def test_evidence_pointer_keeps_stable_beat_modality_time_reference(tmp_path: Path) -> None:
    store = EvidenceStore.empty(tmp_path / "evidence.jsonl")
    store.add(
        EvidenceRecord(
            evidence_id=store.next_id(),
            beat_id="bt00002",
            start_sec=4.0,
            end_sec=8.0,
            modality="asr",
            pointer="bt00002@4.000-8.000",
            verbatim="a blue sign appears",
            claim="a blue sign appears",
        )
    )

    payload = json.loads((tmp_path / "evidence.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert payload["evidence_id"] == "ev_0001"
    assert payload["beat_id"] in payload["pointer"]
    assert payload["modality"] == "asr"
    assert payload["pointer"].endswith("@4.000-8.000")
