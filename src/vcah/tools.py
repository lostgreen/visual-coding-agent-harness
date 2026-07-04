from __future__ import annotations

from pathlib import Path

from vcah.index import ColdIndex
from vcah.memory import EvidenceStore
from vcah.types import EvidenceRecord, ToolAction, ToolResult
from vcah.video import render_timeline_grid


class AgentTools:
    def __init__(self, index: ColdIndex, evidence: EvidenceStore, run_dir: Path) -> None:
        self.index = index
        self.evidence = evidence
        self.run_dir = Path(run_dir)

    def run(self, action: ToolAction) -> ToolResult:
        if action.type == "search_text":
            return self.search_text(action.query)
        if action.type == "search_visual":
            return self.search_visual(action.query)
        if action.type == "open_grid":
            return self.open_grid()
        if action.type == "focus_clip":
            return self.focus_clip(action.beat_id)
        if action.type == "answer":
            return ToolResult(tool="answer", evidence_ids=action.citations, text=action.answer)
        return ToolResult(tool=action.type or "unknown", text="unknown action")

    def search_text(self, query: str) -> ToolResult:
        hits = self.index.search_text(query)
        text = "; ".join(f"{hit.beat_id}:{hit.score:.2f}" for hit in hits[:5]) or "no text hits"
        return ToolResult(tool="search_text", beat_ids=tuple(hit.beat_id for hit in hits[:5]), text=text)

    def search_visual(self, query: str) -> ToolResult:
        hits = self.index.search_visual(query, k=5)
        text = "; ".join(f"{hit.beat_id}:{hit.score:.2f}" for hit in hits) or "no visual hits"
        return ToolResult(tool="search_visual", beat_ids=tuple(hit.beat_id for hit in hits), text=text)

    def open_grid(self) -> ToolResult:
        path = render_timeline_grid([beat.keyframe_path for beat in self.index.beats], self.run_dir / "timeline.jpg")
        return ToolResult(tool="open_grid", text=str(path), payload={"path": str(path)})

    def focus_clip(self, beat_id: str) -> ToolResult:
        if not beat_id:
            beat_id = self.index.beats[0].beat_id if self.index.beats else ""
        beat = self.index.get_beat(beat_id)
        evidence_text = " ".join([beat.asr_text, " ".join(beat.ocr_text)]).strip()
        if not evidence_text:
            evidence_text = f"Visual beat {beat.beat_id} spans {beat.start_sec:.1f}-{beat.end_sec:.1f}s."
        record = EvidenceRecord(
            evidence_id=self.evidence.next_id(),
            beat_id=beat.beat_id,
            start_sec=beat.start_sec,
            end_sec=beat.end_sec,
            text=evidence_text,
            source="focus_clip",
        )
        self.evidence.add(record)
        return ToolResult(tool="focus_clip", beat_ids=(beat.beat_id,), evidence_ids=(record.evidence_id,), text=evidence_text)
