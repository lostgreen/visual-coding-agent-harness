from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from vcah.index import ColdIndex
from vcah.memory import AgentMemory, EvidenceStore
from vcah.model import ATTESTATION_PROMPT
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
            return ToolResult(tool="focus_clip", beat_ids=(beat.beat_id,), evidence_ids=(record.evidence_id,), text=record.verbatim, n_new=1)
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
            return ToolResult(tool="focus_clip", beat_ids=(beat.beat_id,), evidence_ids=(record.evidence_id,), text=record.verbatim, n_new=1)
        records = self._attest_visual_evidence(
            beat_id=beat.beat_id,
            start_sec=beat.start_sec,
            end_sec=beat.end_sec,
            frame_refs=beat.frame_paths or ((beat.keyframe_path,) if beat.keyframe_path else ()),
        )
        return ToolResult(
            tool="focus_clip",
            beat_ids=(beat.beat_id,),
            evidence_ids=tuple(record.evidence_id for record in records),
            text="\n".join(record.verbatim for record in records) or "No visual observations attested.",
            payload={"evidence_created": bool(records), "reason": None if records else "visual_attestation_empty"},
            n_new=len(records),
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
                    verbatim, source_window = _clip_cue_text(beat.asr_cues, evidence_window, fallback=beat.asr_text.strip(), beat_window=Window(beat.start_sec, beat.end_sec))
                    if not verbatim:
                        continue
                    is_window_local = _contains_window(evidence_window, source_window)
                    if not is_window_local:
                        actual_windows.append(
                            {
                                **metadata,
                                "source": "asr",
                                "beat_id": beat.beat_id,
                                "verbatim_source_window": _window_payload(source_window),
                                "verbatim_is_window_local": False,
                                "skipped_reason": "non_window_local_verbatim",
                            }
                        )
                        continue
                    record = self._add_evidence(
                        beat_id=beat.beat_id,
                        start_sec=evidence_window.start_sec,
                        end_sec=evidence_window.end_sec,
                        modality="asr",
                        verbatim=verbatim,
                    )
                    evidence_ids.append(record.evidence_id)
                    texts.append(record.verbatim)
                    actual_windows.append(
                        {
                            **metadata,
                            "source": "asr",
                            "beat_id": beat.beat_id,
                            "evidence_id": record.evidence_id,
                            "verbatim_source_window": _window_payload(source_window),
                            "verbatim_is_window_local": is_window_local,
                        }
                    )
                if "ocr" in selected_modalities and beat.ocr_text:
                    verbatim, source_window = _clip_cue_text(
                        beat.ocr_cues,
                        evidence_window,
                        fallback=" ".join(beat.ocr_text).strip(),
                        beat_window=Window(beat.start_sec, beat.end_sec),
                    )
                    if not verbatim:
                        continue
                    is_window_local = _contains_window(evidence_window, source_window)
                    if not is_window_local:
                        actual_windows.append(
                            {
                                **metadata,
                                "source": "ocr",
                                "beat_id": beat.beat_id,
                                "verbatim_source_window": _window_payload(source_window),
                                "verbatim_is_window_local": False,
                                "skipped_reason": "non_window_local_verbatim",
                            }
                        )
                        continue
                    record = self._add_evidence(
                        beat_id=beat.beat_id,
                        start_sec=evidence_window.start_sec,
                        end_sec=evidence_window.end_sec,
                        modality="ocr",
                        verbatim=verbatim,
                    )
                    evidence_ids.append(record.evidence_id)
                    texts.append(record.verbatim)
                    actual_windows.append(
                        {
                            **metadata,
                            "source": "ocr",
                            "beat_id": beat.beat_id,
                            "evidence_id": record.evidence_id,
                            "verbatim_source_window": _window_payload(source_window),
                            "verbatim_is_window_local": is_window_local,
                        }
                    )
                if "frames" in selected_modalities and beat.frame_paths:
                    records = self._attest_visual_evidence(
                        beat_id=beat.beat_id,
                        start_sec=evidence_window.start_sec,
                        end_sec=evidence_window.end_sec,
                        frame_refs=beat.frame_paths,
                    )
                    for record in records:
                        evidence_ids.append(record.evidence_id)
                        texts.append(record.verbatim)
                        actual_windows.append({**metadata, "source": "visual", "beat_id": beat.beat_id, "evidence_id": record.evidence_id})

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
            n_new=len(evidence_ids),
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
        frame_refs: tuple[str, ...] = (),
        attestation_model: str = "",
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
            frame_refs=frame_refs,
            attestation_model=attestation_model,
        )
        self.evidence.add(record)
        return record

    def _attest_visual_evidence(
        self,
        *,
        beat_id: str,
        start_sec: float,
        end_sec: float,
        frame_refs: tuple[str, ...],
    ) -> tuple[EvidenceRecord, ...]:
        model = getattr(self.index.visual_index, "model", None)
        if model is None or not frame_refs:
            return ()
        observations = tuple(str(item).strip() for item in model.attest(frame_refs, ATTESTATION_PROMPT) if str(item).strip())
        records = []
        for observation in observations:
            records.append(
                self._add_evidence(
                    beat_id=beat_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    modality="visual",
                    verbatim=observation,
                    frame_refs=frame_refs,
                    attestation_model=str(getattr(model, "vision_model", "") or ""),
                )
            )
        return tuple(records)


def _pointer(beat_id: str, start_sec: float, end_sec: float) -> str:
    return f"{beat_id}@{float(start_sec):.3f}-{float(end_sec):.3f}"


def _window_payload(window: Window) -> dict[str, float]:
    return {"start_sec": window.start_sec, "end_sec": window.end_sec}


def _clip_cue_text(
    cues: tuple[object, ...],
    window: Window,
    *,
    fallback: str,
    beat_window: Window,
) -> tuple[str, Window]:
    lines = []
    source_start: float | None = None
    source_end: float | None = None
    for cue in cues:
        if not isinstance(cue, dict):
            continue
        cue_start = float(cue.get("start_sec", 0.0) or 0.0)
        cue_end = float(cue.get("end_sec", cue_start) or cue_start)
        if cue_end < window.start_sec or cue_start > window.end_sec:
            continue
        text = str(cue.get("text", "") or "").strip()
        if not text:
            continue
        lines.append(text)
        source_start = cue_start if source_start is None else min(source_start, cue_start)
        source_end = cue_end if source_end is None else max(source_end, cue_end)
    if lines:
        return " ".join(lines), Window(max(window.start_sec, source_start or window.start_sec), min(window.end_sec, source_end or window.end_sec))
    return str(fallback or "").strip(), beat_window


def _contains_window(container: Window, item: Window) -> bool:
    return item.start_sec >= container.start_sec and item.end_sec <= container.end_sec
