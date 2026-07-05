from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from vcah.index import ColdIndex
from vcah.memory import AgentMemory, EvidenceStore
from vcah.types import EvidenceRecord, ToolAction, ToolResult, Window, window_overlap_ratio
from vcah.video import render_timeline_grid


@dataclass(frozen=True)
class _ResolvedWindow:
    requested: Window
    beats: tuple[object, ...]
    coverage: float


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
        if action.type == "inspect_window":
            return self.inspect_window(action.windows, action.modalities)
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

    def inspect_window(self, windows: tuple[Window, ...], modalities: tuple[str, ...] = ()) -> ToolResult:
        if not windows:
            return ToolResult(
                tool="inspect_window",
                text="No requested windows supplied.",
                payload={
                    "requested_windows": [],
                    "actual_windows": [],
                    "window_coverage_report": [],
                    "fallback_used": False,
                    "fallback_reason": None,
                    "error": "missing_requested_window",
                },
            )
        selected_modalities = set(modalities or ("asr", "ocr", "frames"))
        resolved: list[_ResolvedWindow] = []
        actual_windows: list[dict[str, object]] = []
        coverage_report: list[dict[str, object]] = []
        for requested in windows:
            beats = tuple(
                beat
                for beat in self.index.beats
                if beat.end_sec > requested.start_sec and beat.start_sec < requested.end_sec
            )
            actuals = tuple(Window(beat.start_sec, beat.end_sec) for beat in beats)
            coverage = window_overlap_ratio(requested, actuals)
            coverage_report.append(
                {
                    "requested": _window_payload(requested),
                    "coverage": coverage,
                    "passed": coverage >= 0.8,
                }
            )
            actual_windows.extend(
                {
                    "start_sec": beat.start_sec,
                    "end_sec": beat.end_sec,
                    "source": "beat",
                    "beat_id": beat.beat_id,
                    "requested_window": _window_payload(requested),
                }
                for beat in beats
            )
            resolved.append(_ResolvedWindow(requested=requested, beats=beats, coverage=coverage))

        if any(item.coverage < 0.8 for item in resolved):
            return ToolResult(
                tool="inspect_window",
                text="window_coverage_failed",
                payload={
                    "requested_windows": [_window_payload(window) for window in windows],
                    "actual_windows": actual_windows,
                    "window_coverage_report": coverage_report,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "error": "window_coverage_failed",
                },
            )

        beat_ids: list[str] = []
        evidence_ids: list[str] = []
        texts: list[str] = []

        for item in resolved:
            for beat in item.beats:
                if beat.beat_id not in beat_ids:
                    beat_ids.append(beat.beat_id)
                evidence_window = Window(
                    max(beat.start_sec, item.requested.start_sec),
                    min(beat.end_sec, item.requested.end_sec),
                )
                metadata = {
                    "requested_window": _window_payload(item.requested),
                    "actual_beat_window": {"start_sec": beat.start_sec, "end_sec": beat.end_sec},
                    "evidence_window": _window_payload(evidence_window),
                }
                if "asr" in selected_modalities and beat.asr_text.strip():
                    record = self._add_evidence(
                        beat_id=beat.beat_id,
                        start_sec=evidence_window.start_sec,
                        end_sec=evidence_window.end_sec,
                        modality="asr",
                        verbatim=beat.asr_text.strip(),
                    )
                    evidence_ids.append(record.evidence_id)
                    texts.append(record.verbatim)
                    actual_windows.append({**metadata, "source": "asr", "beat_id": beat.beat_id, "evidence_id": record.evidence_id})
                if "ocr" in selected_modalities and beat.ocr_text:
                    verbatim = " ".join(beat.ocr_text).strip()
                    record = self._add_evidence(
                        beat_id=beat.beat_id,
                        start_sec=evidence_window.start_sec,
                        end_sec=evidence_window.end_sec,
                        modality="ocr",
                        verbatim=verbatim,
                    )
                    evidence_ids.append(record.evidence_id)
                    texts.append(record.verbatim)
                    actual_windows.append({**metadata, "source": "ocr", "beat_id": beat.beat_id, "evidence_id": record.evidence_id})
                if "frames" in selected_modalities and beat.frame_paths:
                    verbatim = (
                        f"Frame evidence for {beat.beat_id} at "
                        f"{evidence_window.start_sec:.1f}-{evidence_window.end_sec:.1f}s: {', '.join(beat.frame_paths)}"
                    )
                    record = self._add_evidence(
                        beat_id=beat.beat_id,
                        start_sec=evidence_window.start_sec,
                        end_sec=evidence_window.end_sec,
                        modality="frame",
                        verbatim=verbatim,
                    )
                    evidence_ids.append(record.evidence_id)
                    texts.append(record.verbatim)
                    actual_windows.append({**metadata, "source": "frame", "beat_id": beat.beat_id, "evidence_id": record.evidence_id})

        return ToolResult(
            tool="inspect_window",
            beat_ids=tuple(beat_ids),
            evidence_ids=tuple(evidence_ids),
            text="\n".join(texts) or "No ASR/OCR/frame evidence found in requested windows.",
            payload={
                "requested_windows": [_window_payload(window) for window in windows],
                "actual_windows": actual_windows,
                "window_coverage_report": coverage_report,
                "fallback_used": False,
                "fallback_reason": None,
                "evidence_created": bool(evidence_ids),
            },
        )

    def _selected_beat_ids(self, beat_ids: tuple[str, ...]) -> tuple[str, ...]:
        selected = tuple(beat_id for beat_id in beat_ids if beat_id)
        if not selected:
            selected = tuple(self.memory.last_hits)
        return selected

    def _add_evidence(
        self,
        *,
        beat_id: str,
        start_sec: float,
        end_sec: float,
        modality: str,
        verbatim: str,
    ) -> EvidenceRecord:
        record = EvidenceRecord(
            evidence_id=self.evidence.next_id(),
            beat_id=beat_id,
            start_sec=start_sec,
            end_sec=end_sec,
            modality=modality,  # type: ignore[arg-type]
            pointer=_pointer(beat_id, start_sec, end_sec),
            verbatim=verbatim,
            claim=verbatim,
        )
        self.evidence.add(record)
        return record


def _pointer(beat_id: str, start_sec: float, end_sec: float) -> str:
    return f"{beat_id}@{float(start_sec):.3f}-{float(end_sec):.3f}"


def _window_payload(window: Window) -> dict[str, float]:
    return {"start_sec": window.start_sec, "end_sec": window.end_sec}
