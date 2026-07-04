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
        text="A blue sign is visible.",
        source="focus_clip",
    )

    store.add(record)

    assert store.valid(("ev_0001",))
    assert not store.valid(("ev_9999",))
    assert json.loads((tmp_path / "evidence.jsonl").read_text(encoding="utf-8").splitlines()[0])["evidence_id"] == "ev_0001"


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
