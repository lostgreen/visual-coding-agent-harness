from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

from vcah.types import EvidenceRecord, ToolAction, ToolResult, to_jsonable


class AgentMemory:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.observations: list[str] = []
        self.failed_searches: list[str] = []
        self.visited_beats: list[str] = []
        self.last_hits: list[str] = []

    @classmethod
    def empty(cls, path: Path) -> "AgentMemory":
        return cls(path)

    def record_result(self, result: ToolResult) -> None:
        if result.text:
            self.observations.append(result.text)
        for beat_id in result.beat_ids:
            if beat_id not in self.visited_beats:
                self.visited_beats.append(beat_id)
        if result.tool.startswith("search"):
            self.last_hits = list(result.beat_ids)
        if result.tool.startswith("search") and not result.beat_ids:
            self.failed_searches.append(result.text or result.tool)

    def digest(self) -> str:
        parts = self.observations[-4:]
        if self.last_hits:
            parts.append(f"last_hits={','.join(self.last_hits[-8:])}")
        parts.append(f"visited={','.join(self.visited_beats[-8:])}")
        return "\n".join(part for part in parts if part)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "observations": self.observations,
                    "failed_searches": self.failed_searches,
                    "last_hits": self.last_hits,
                    "visited_beats": self.visited_beats,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


class EvidenceStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.records: list[EvidenceRecord] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    @classmethod
    def empty(cls, path: Path) -> "EvidenceStore":
        return cls(path)

    def add(self, record: EvidenceRecord) -> None:
        self.records.append(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")

    def next_id(self) -> str:
        return f"ev_{len(self.records) + 1:04d}"

    def valid(self, citations: Iterable[str]) -> bool:
        citations = tuple(citations)
        if not citations:
            return False
        known = {record.evidence_id for record in self.records}
        return all(citation in known for citation in citations)

    def digest(self) -> str:
        return " ".join(record.evidence_id for record in self.records)


class TraceStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def append(self, action: ToolAction, result: ToolResult) -> None:
        entry = {"action": to_jsonable(action), "result": to_jsonable(result)}
        for key in (
            "requested_windows",
            "actual_windows",
            "window_coverage_report",
            "fallback_used",
            "fallback_reason",
            "investigator_received_hypothesis",
            "investigator_evidence_table",
            "final_verification",
        ):
            if key in result.payload:
                entry[key] = to_jsonable(result.payload[key])
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
