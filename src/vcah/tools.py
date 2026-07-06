from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

from vcah.index import ColdIndex
from vcah.memory import AgentMemory, EvidenceStore
from vcah.model import ATTESTATION_PROMPT
from vcah.types import CoverageSegment, EvidenceRecord, ToolAction, ToolResult, Window, window_overlap_ratio
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
            return self.focus_clip(action.beat_id, action.beat_ids, action.modalities)
        if action.type == "inspect_window":
            return self.inspect_window(
                action.windows,
                action.modalities,
                raw_window_count=action.raw_window_count,
                window_parse_errors=action.window_parse_errors,
            )
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

    def focus_clip(self, beat_id: str, beat_ids: tuple[str, ...] = (), modalities: tuple[str, ...] = ()) -> ToolResult:
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
        selected_modalities = set(modalities or ("asr", "ocr", "frames"))
        records: list[EvidenceRecord] = []
        if "asr" in selected_modalities and beat.asr_text.strip():
            records.append(
                self._add_evidence(
                    beat_id=beat.beat_id,
                    start_sec=beat.start_sec,
                    end_sec=beat.end_sec,
                    modality="asr",
                    verbatim=beat.asr_text.strip(),
                    temporal_scope="window",
                    evidence_kind="quote",
                    sampling_coverage="complete_for_manifest",
                    request_ids=("focus_clip",),
                    coverage_manifest=(CoverageSegment("focus_clip", beat.start_sec, beat.end_sec, "asr", 1.0),),
                )
            )
        if "ocr" in selected_modalities and beat.ocr_text:
            verbatim = " ".join(beat.ocr_text).strip()
            records.append(
                self._add_evidence(
                    beat_id=beat.beat_id,
                    start_sec=beat.start_sec,
                    end_sec=beat.end_sec,
                    modality="ocr",
                    verbatim=verbatim,
                    temporal_scope="window",
                    evidence_kind="quote",
                    sampling_coverage="complete_for_manifest",
                    request_ids=("focus_clip",),
                    coverage_manifest=(CoverageSegment("focus_clip", beat.start_sec, beat.end_sec, "ocr", 1.0),),
                )
            )
        if "frames" in selected_modalities:
            records.extend(
                self._attest_visual_evidence(
                    beat_id=beat.beat_id,
                    start_sec=beat.start_sec,
                    end_sec=beat.end_sec,
                    frame_refs=beat.frame_paths or ((beat.keyframe_path,) if beat.keyframe_path else ()),
                    request_ids=("focus_clip",),
                    coverage_manifest=(CoverageSegment("focus_clip", beat.start_sec, beat.end_sec, "visual", 1.0),),
                )
            )
        return ToolResult(
            tool="focus_clip",
            beat_ids=(beat.beat_id,),
            evidence_ids=tuple(record.evidence_id for record in records),
            text="\n".join(record.verbatim for record in records) or "No evidence found for selected modalities.",
            payload={"evidence_created": bool(records), "reason": None if records else "no_selected_modality_evidence"},
            n_new=len(records),
        )

    def inspect_window(
        self,
        windows: tuple[Window, ...],
        modalities: tuple[str, ...] = (),
        *,
        raw_window_count: int = 0,
        window_parse_errors: tuple[str, ...] = (),
    ) -> ToolResult:
        if not windows:
            raw_ids = _raw_request_ids(max(int(raw_window_count), len(windows)))
            return ToolResult(
                tool="inspect_window",
                text="No requested windows supplied.",
                payload={
                    "requested_windows": [],
                    "actual_windows": [],
                    "window_coverage_report": [],
                    "window_lineage": _lineage_payload(raw_ids, (), [], [], parse_errors=window_parse_errors),
                    "fallback_used": False,
                    "fallback_reason": None,
                    "error": "missing_requested_window",
                },
            )
        selected_modalities = set(modalities or ("asr", "ocr", "frames"))
        resolved: list[_ResolvedWindow] = []
        actual_windows: list[dict[str, object]] = []
        coverage_report: list[dict[str, object]] = []
        requested_ids = tuple(_request_id(index, requested) for index, requested in enumerate(windows, start=1))
        if window_parse_errors or int(raw_window_count) > len(windows):
            raw_ids = _raw_request_ids(max(int(raw_window_count), len(windows)))
        else:
            raw_ids = requested_ids
        executed_request_ids: list[str] = []
        materialized_request_ids: list[str] = []
        for ordinal, requested in enumerate(windows, start=1):
            request_id = _request_id(ordinal, requested)
            beats = tuple(
                beat
                for beat in self.index.beats
                if beat.end_sec > requested.start_sec and beat.start_sec < requested.end_sec
            )
            if beats:
                executed_request_ids.append(request_id)
            actuals = tuple(Window(beat.start_sec, beat.end_sec) for beat in beats)
            coverage = window_overlap_ratio(requested, actuals)
            coverage_report.append(
                {
                    "request_id": request_id,
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
                    "request_id": request_id,
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
                    "window_lineage": _lineage_payload(raw_ids, requested_ids, executed_request_ids, materialized_request_ids, parse_errors=window_parse_errors),
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
                    "request_id": _request_id(tuple(windows).index(item.requested) + 1, item.requested),
                    "requested_window": _window_payload(item.requested),
                    "actual_beat_window": {"start_sec": beat.start_sec, "end_sec": beat.end_sec},
                    "evidence_window": _window_payload(evidence_window),
                }
                request_id = str(metadata["request_id"])
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
                        temporal_scope="window",
                        evidence_kind="quote",
                        sampling_coverage="complete_for_manifest",
                        request_ids=(request_id,),
                        coverage_manifest=(CoverageSegment(request_id, evidence_window.start_sec, evidence_window.end_sec, "asr", item.coverage),),
                    )
                    evidence_ids.append(record.evidence_id)
                    if request_id not in materialized_request_ids:
                        materialized_request_ids.append(request_id)
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
                        temporal_scope="window",
                        evidence_kind="quote",
                        sampling_coverage="complete_for_manifest",
                        request_ids=(request_id,),
                        coverage_manifest=(CoverageSegment(request_id, evidence_window.start_sec, evidence_window.end_sec, "ocr", item.coverage),),
                    )
                    evidence_ids.append(record.evidence_id)
                    if request_id not in materialized_request_ids:
                        materialized_request_ids.append(request_id)
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
                    frame_refs = _frame_refs_in_window(beat.frame_paths, beat.frame_times, evidence_window)
                    if not frame_refs:
                        actual_windows.append({**metadata, "source": "visual", "beat_id": beat.beat_id, "skipped_reason": "no_in_window_frame_refs"})
                        continue
                    records = self._attest_visual_evidence(
                        beat_id=beat.beat_id,
                        start_sec=evidence_window.start_sec,
                        end_sec=evidence_window.end_sec,
                        frame_refs=frame_refs,
                        request_ids=(request_id,),
                        coverage_manifest=(CoverageSegment(request_id, evidence_window.start_sec, evidence_window.end_sec, "visual", item.coverage),),
                    )
                    for record in records:
                        evidence_ids.append(record.evidence_id)
                        if request_id not in materialized_request_ids:
                            materialized_request_ids.append(request_id)
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
                "window_lineage": _lineage_payload(raw_ids, requested_ids, executed_request_ids, materialized_request_ids, parse_errors=window_parse_errors),
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
        temporal_scope: str = "window",
        evidence_kind: str = "quote",
        observation_polarity: str = "unknown",
        sampling_coverage: str = "unknown",
        parent_evidence_ids: tuple[str, ...] = (),
        request_ids: tuple[str, ...] = (),
        coverage_manifest: tuple[CoverageSegment, ...] = (),
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
            temporal_scope=temporal_scope,  # type: ignore[arg-type]
            evidence_kind=evidence_kind,  # type: ignore[arg-type]
            observation_polarity=observation_polarity,  # type: ignore[arg-type]
            sampling_coverage=sampling_coverage,  # type: ignore[arg-type]
            parent_evidence_ids=parent_evidence_ids,
            request_ids=request_ids,
            coverage_manifest=coverage_manifest,
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
        request_ids: tuple[str, ...] = (),
        coverage_manifest: tuple[CoverageSegment, ...] = (),
    ) -> tuple[EvidenceRecord, ...]:
        model = getattr(self.index.visual_index, "model", None)
        if model is None or not frame_refs:
            return ()
        records = []
        for observation, polarity in _visual_observations(model.attest(frame_refs, ATTESTATION_PROMPT)):
            records.append(
                self._add_evidence(
                    beat_id=beat_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    modality="visual",
                    verbatim=observation,
                    frame_refs=frame_refs,
                    attestation_model=str(getattr(model, "vision_model", "") or ""),
                    temporal_scope="local_frame",
                    evidence_kind="visual_observation",
                    observation_polarity=polarity,
                    sampling_coverage="sparse",
                    request_ids=request_ids,
                    coverage_manifest=coverage_manifest,
                )
            )
        return tuple(records)


def _pointer(beat_id: str, start_sec: float, end_sec: float) -> str:
    return f"{beat_id}@{float(start_sec):.3f}-{float(end_sec):.3f}"


def _window_payload(window: Window) -> dict[str, object]:
    payload: dict[str, float | str] = {"start_sec": window.start_sec, "end_sec": window.end_sec}
    if window.request_id:
        payload["request_id"] = window.request_id
    return payload


def _request_id(ordinal: int, window: Window) -> str:
    if window.request_id:
        return window.request_id
    return f"win_{int(ordinal):04d}_{int(window.start_sec * 1000):010d}_{int(window.end_sec * 1000):010d}"


def _raw_request_ids(count: int) -> tuple[str, ...]:
    return tuple(f"raw_window_{index:04d}" for index in range(1, max(0, int(count)) + 1))


def _lineage_payload(
    raw_request_ids: tuple[str, ...],
    requested_ids: tuple[str, ...],
    executed_request_ids: list[str],
    materialized_request_ids: list[str],
    *,
    parse_errors: tuple[str, ...] = (),
) -> dict[str, object]:
    raw = list(raw_request_ids or requested_ids)
    requested = list(requested_ids)
    dispatched = requested
    executed = list(dict.fromkeys(executed_request_ids))
    materialized = list(dict.fromkeys(materialized_request_ids))
    error = _lineage_error(raw, requested, dispatched, executed, materialized, parse_errors)
    dropped_source = raw if error == "window_request_parse_loss" else requested
    dropped_target = requested if error == "window_request_parse_loss" else executed
    if error == "window_request_materialization_loss":
        dropped_source = executed
        dropped_target = materialized
    return {
        "raw_requested_ids": raw,
        "parsed_requested_ids": requested,
        "dispatched_request_ids": dispatched,
        "executed_request_ids": executed,
        "materialized_request_ids": materialized,
        "error": error,
        "parse_errors": list(parse_errors),
        "dropped_request_ids": [item for item in dropped_source if item not in dropped_target],
    }


def _lineage_error(
    raw: list[str],
    parsed: list[str],
    dispatched: list[str],
    executed: list[str],
    materialized: list[str],
    parse_errors: tuple[str, ...],
) -> str | None:
    if parse_errors or len(parsed) < len(raw):
        return "window_request_parse_loss"
    if set(parsed) - set(dispatched):
        return "window_request_dispatch_loss"
    if set(dispatched) - set(executed):
        return "window_request_execution_loss"
    if executed and (not materialized or set(executed) - set(materialized)):
        return "window_request_materialization_loss"
    if set(materialized) - set(executed):
        return "window_request_materialization_loss"
    return None


def _frame_refs_in_window(frame_refs: tuple[str, ...], frame_times: tuple[float, ...], window: Window) -> tuple[str, ...]:
    if not frame_times:
        return ()
    kept = []
    for ref, time_sec in zip(frame_refs, frame_times):
        if window.start_sec <= float(time_sec) <= window.end_sec:
            kept.append(ref)
    return tuple(kept)


def _visual_observations(raw_items: tuple[object, ...]) -> tuple[tuple[str, str], ...]:
    observations: list[tuple[str, str]] = []
    for item in raw_items:
        if isinstance(item, dict):
            text = str(item.get("observation") or item.get("text") or item.get("verbatim") or "").strip()
            polarity = _clean_polarity(item.get("polarity") or item.get("observation_polarity"))
            if text:
                observations.append((text, polarity))
            continue
        text = str(item or "").strip()
        if not text:
            continue
        parsed = _json_payload(text)
        if isinstance(parsed, dict):
            nested = parsed.get("observations") or parsed.get("evidence") or ()
            if isinstance(nested, list):
                observations.extend(_visual_observations(tuple(nested)))
                continue
            observation = str(parsed.get("observation") or parsed.get("text") or "").strip()
            if observation:
                observations.append((observation, _clean_polarity(parsed.get("polarity") or parsed.get("observation_polarity"))))
                continue
        observations.append((text, "unknown"))
    return tuple(observations)


def _json_payload(text: str) -> object:
    try:
        return json.loads(text)
    except Exception:
        return None


def _clean_polarity(value: object) -> str:
    text = str(value or "").strip().casefold()
    return text if text in {"positive", "negative", "unknown"} else "unknown"


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
