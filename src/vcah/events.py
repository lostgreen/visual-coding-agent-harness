from __future__ import annotations

from dataclasses import dataclass

from vcah.types import CoverageSegment


@dataclass(frozen=True)
class EventObservation:
    event_id: str
    evidence_id: str
    start_sec: float
    end_sec: float
    predicate: str

    def __post_init__(self) -> None:
        start = float(self.start_sec)
        end = float(self.end_sec)
        if end < start:
            raise ValueError("EventObservation end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "event_id", str(self.event_id or "").strip())
        object.__setattr__(self, "evidence_id", str(self.evidence_id or "").strip())
        object.__setattr__(self, "start_sec", start)
        object.__setattr__(self, "end_sec", end)
        object.__setattr__(self, "predicate", str(self.predicate or "").strip())


@dataclass(frozen=True)
class EventOrderResult:
    ordered_event_ids: tuple[str, ...]
    parent_evidence_ids: tuple[str, ...]
    coverage_manifest: tuple[CoverageSegment, ...]


def order_events(events: tuple[EventObservation, ...], coverage_manifest: tuple[CoverageSegment, ...] = ()) -> EventOrderResult:
    ordered = tuple(sorted(events, key=lambda event: (event.start_sec, event.end_sec, event.event_id)))
    return EventOrderResult(
        ordered_event_ids=tuple(event.event_id for event in ordered),
        parent_evidence_ids=tuple(dict.fromkeys(event.evidence_id for event in ordered if event.evidence_id)),
        coverage_manifest=coverage_manifest,
    )
