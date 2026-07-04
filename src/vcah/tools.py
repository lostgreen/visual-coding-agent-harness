from __future__ import annotations

from pathlib import Path

from vcah.index import ColdIndex
from vcah.memory import AgentMemory, EvidenceStore
from vcah.types import EvidenceRecord, ToolAction, ToolResult
from vcah.video import render_timeline_grid


class AgentTools:
    def __init__(self, index: ColdIndex, memory: AgentMemory, evidence: EvidenceStore, run_dir: Path) -> None:
        self.index = index
        self.memory = memory
        self.evidence = evidence
        self.run_dir = Path(run_dir)

    def run(self, action: ToolAction) -> ToolResult:
        if action.type == "search_text":
            return self.search_text(action.query)
        if action.type == "search_visual":
            return self.search_visual(action.query)
        if action.type == "open_grid":
            return self.open_grid(action.beat_ids)
        if action.type == "focus_clip":
            return self.focus_clip(action.beat_id, action.beat_ids)
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

    def open_grid(self, beat_ids: tuple[str, ...] = ()) -> ToolResult:
        selected = self._selected_beat_ids(beat_ids)
        beats = [self.index.get_beat(beat_id) for beat_id in selected] if selected else list(self.index.beats)
        path = render_timeline_grid([beat.keyframe_path for beat in beats], self.run_dir / "grid.jpg")
        return ToolResult(
            tool="open_grid",
            beat_ids=tuple(beat.beat_id for beat in beats),
            text=str(path),
            payload={"path": str(path), "beat_ids": tuple(beat.beat_id for beat in beats)},
        )

    def focus_clip(self, beat_id: str, beat_ids: tuple[str, ...] = ()) -> ToolResult:
        if not beat_id:
            selected = self._selected_beat_ids(beat_ids)
            if not selected:
                return ToolResult(
                    tool="focus_clip",
                    text="No candidate beat selected for focus_clip.",
                    payload={"evidence_created": False, "reason": "needs_candidate"},
                )
            beat_id = selected[0]
        beat = self.index.get_beat(beat_id)
        if beat.asr_text.strip():
            record = EvidenceRecord(
                evidence_id=self.evidence.next_id(),
                beat_id=beat.beat_id,
                start_sec=beat.start_sec,
                end_sec=beat.end_sec,
                modality="asr",
                pointer=_pointer(beat.beat_id, beat.start_sec, beat.end_sec),
                verbatim=beat.asr_text.strip(),
                claim=beat.asr_text.strip(),
            )
            self.evidence.add(record)
            return ToolResult(tool="focus_clip", beat_ids=(beat.beat_id,), evidence_ids=(record.evidence_id,), text=record.verbatim)
        if beat.ocr_text:
            verbatim = " ".join(beat.ocr_text).strip()
            record = EvidenceRecord(
                evidence_id=self.evidence.next_id(),
                beat_id=beat.beat_id,
                start_sec=beat.start_sec,
                end_sec=beat.end_sec,
                modality="ocr",
                pointer=_pointer(beat.beat_id, beat.start_sec, beat.end_sec),
                verbatim=verbatim,
                claim=verbatim,
            )
            self.evidence.add(record)
            return ToolResult(tool="focus_clip", beat_ids=(beat.beat_id,), evidence_ids=(record.evidence_id,), text=record.verbatim)
        observation = f"Observed frames for {beat.beat_id} at {beat.start_sec:.1f}-{beat.end_sec:.1f}s; no verified ASR/OCR evidence."
        return ToolResult(
            tool="focus_clip",
            beat_ids=(beat.beat_id,),
            text=observation,
            payload={"evidence_created": False, "reason": "visual_only_requires_verification"},
        )

    def _selected_beat_ids(self, beat_ids: tuple[str, ...]) -> tuple[str, ...]:
        selected = tuple(beat_id for beat_id in beat_ids if beat_id)
        if not selected:
            selected = tuple(self.memory.last_hits)
        return selected


def _pointer(beat_id: str, start_sec: float, end_sec: float) -> str:
    return f"{beat_id}@{float(start_sec):.3f}-{float(end_sec):.3f}"
