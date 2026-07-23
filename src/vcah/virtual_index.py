from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from vcah.index import ColdIndex, TextIndex, VisualIndex
from vcah.model import ModelClient
from vcah.types import Beat, Chapter, IndexDiagnostics
from vcah.video import render_timeline_grid
from vcah.virtual_video import VirtualFrameRef, VirtualVideoWorkspace, virtual_to_source_windows


@dataclass(frozen=True)
class VirtualBeat:
    beat_id: str
    start_sec: float
    end_sec: float
    thumbnail_grid_path: str
    thumbnail_grid_paths: tuple[str, ...]
    frame_refs: tuple[VirtualFrameRef, ...]
    asr_cues: tuple[Mapping[str, Any], ...]
    source_lineage: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class VirtualBeatIndexResult:
    cold_index: ColdIndex
    virtual_beats: tuple[VirtualBeat, ...]
    beat_index_path: Path
    timeline_grid_path: Path


def build_virtual_beat_index(
    workspace: VirtualVideoWorkspace,
    frame_refs: Sequence[VirtualFrameRef],
    *,
    model: ModelClient | None = None,
    beat_sec: float = 60.0,
) -> VirtualBeatIndexResult:
    model = model or ModelClient()
    beat_width = max(1.0, float(beat_sec))
    frames = tuple(sorted(frame_refs, key=lambda frame: (frame.virtual_time_sec, frame.frame_id)))
    asr_cues = workspace.read_asr_virtual_cues()
    virtual_beats: list[VirtualBeat] = []
    number = 1
    for segment in workspace.manifest.segments:
        start = float(segment.virtual_start_sec)
        while start < float(segment.virtual_end_sec):
            end = min(float(segment.virtual_end_sec), start + beat_width)
            beat_frames = tuple(frame for frame in frames if start <= frame.virtual_time_sec < end)
            if not beat_frames and frames:
                # Keep empty timeline regions indexable; use nearest existing frame as visual placeholder.
                nearest = min(frames, key=lambda frame: abs(frame.virtual_time_sec - ((start + end) / 2.0)))
                beat_frames = (nearest,)
            thumbs = build_beat_thumbnail_grids(
                beat_frames,
                workspace.asset_root / "beat_thumbnails" / f"bt{number:05d}",
            )
            beat_cues = tuple(cue for cue in asr_cues if _cue_overlaps(cue, start, end))
            virtual_beats.append(
                VirtualBeat(
                    beat_id=f"bt{number:05d}",
                    start_sec=round(start, 3),
                    end_sec=round(end, 3),
                    thumbnail_grid_path=str(thumbs[0]),
                    thumbnail_grid_paths=tuple(str(path) for path in thumbs),
                    frame_refs=beat_frames,
                    asr_cues=beat_cues,
                    source_lineage=_source_lineage(workspace, start, end),
                )
            )
            start = end
            number += 1

    beats = tuple(_to_runtime_beat(beat) for beat in virtual_beats)
    chapters = _chapters_for_beats(beats, workspace.manifest.duration_sec)
    text_index = TextIndex()
    for beat in beats:
        text_index.add(beat.beat_id, beat.asr_text, modality="asr")
        text_index.add(beat.beat_id, " ".join(beat.ocr_text), modality="ocr")
    visual_index = VisualIndex(model)
    visual_index.build(beats)
    diagnostics = _diagnostics(workspace.manifest.duration_sec, chapters, beats, visual_index, model)
    cold = ColdIndex(
        video_path=f"virtual://{workspace.workspace_id}",
        duration_sec=workspace.manifest.duration_sec,
        chapters=chapters,
        beats=beats,
        text_index=text_index,
        visual_index=visual_index,
        diagnostics=diagnostics,
    )
    cold.save(workspace.cold_index_dir)
    beat_index_path = workspace.asset_root / "beat_index.json"
    beat_index_path.write_text(
        json.dumps(
            {
                "workspace_id": workspace.workspace_id,
                "beat_sec": beat_width,
                "beats": [_virtual_beat_payload(beat) for beat in virtual_beats],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    timeline_grid_path = render_timeline_grid(
        [beat.thumbnail_grid_path for beat in virtual_beats],
        workspace.asset_root / "timeline_grid.jpg",
    )
    return VirtualBeatIndexResult(cold, tuple(virtual_beats), beat_index_path, timeline_grid_path)


def build_beat_thumbnail_grid(frame_refs: Sequence[VirtualFrameRef], out_path: Path, *, max_frames: int = 9) -> Path:
    selected = _select_grid_frames(tuple(frame_refs), max_frames=max_frames)
    return _render_frame_grid(selected, out_path, max_columns=3)


def build_beat_thumbnail_grids(frame_refs: Sequence[VirtualFrameRef], out_prefix: Path, *, max_frames: int = 16) -> tuple[Path, ...]:
    selected = _select_grid_frames(tuple(frame_refs), max_frames=max_frames)
    selected_groups = tuple(tuple(selected[index : index + 4]) for index in range(0, len(selected), 4))
    paths = []
    for index, group in enumerate(selected_groups):
        path = out_prefix.parent / f"{out_prefix.name}_q{index}.jpg"
        _render_frame_grid(group, path, max_columns=4)
        paths.append(path)
    return tuple(paths)


def _render_frame_grid(frame_refs: Sequence[VirtualFrameRef], out_path: Path, *, max_columns: int) -> Path:
    refs = tuple(frame for frame in frame_refs if Path(frame.path).exists())
    cell_size = (160, 90)
    cap = max(1, int(max_columns))
    if refs:
        rows = tuple(tuple(refs[index : index + cap]) for index in range(0, len(refs), cap))
        canvas_width = cell_size[0] * min(cap, len(refs))
    else:
        rows = ((),)
        canvas_width = cell_size[0]
    canvas = Image.new("RGB", (canvas_width, cell_size[1] * len(rows)), color=(18, 18, 18))
    for row_index, row in enumerate(rows):
        if not row:
            continue
        cell_width = max(1, canvas_width // len(row))
        for col_index, frame in enumerate(row):
            with Image.open(frame.path) as image:
                image = image.convert("RGB")
                image.thumbnail((cell_width, cell_size[1]))
                cell = Image.new("RGB", (cell_width, cell_size[1]), color=(12, 12, 12))
                cell.paste(image, ((cell_width - image.width) // 2, (cell_size[1] - image.height) // 2))
            canvas.paste(cell, (col_index * cell_width, row_index * cell_size[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=88)
    return out_path


def build_segment_overview_grid(frame_refs: Sequence[VirtualFrameRef], out_path: Path, *, max_frames: int = 16) -> Path:
    selected = _select_grid_frames(tuple(frame_refs), max_frames=max_frames)
    return _render_frame_grid(selected, out_path, max_columns=4)


def build_workspace_overview(
    workspace: VirtualVideoWorkspace,
    frame_refs: Sequence[VirtualFrameRef] | None = None,
    *,
    thumbnail_budget: int = 40,
) -> dict[str, Any]:
    budget = max(1, int(thumbnail_budget))
    frames = tuple(sorted(frame_refs if frame_refs is not None else workspace.read_frame_manifest(), key=lambda frame: frame.virtual_time_sec))
    segments = tuple(workspace.manifest.segments)
    if len(segments) <= budget:
        groups = tuple((segment,) for segment in segments)
    else:
        group_size = max(1, math.ceil(len(segments) / budget))
        groups = tuple(tuple(segments[index : index + group_size]) for index in range(0, len(segments), group_size))

    asr_cues = workspace.read_asr_virtual_cues()
    overviews: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        start = min(float(segment.virtual_start_sec) for segment in group)
        end = max(float(segment.virtual_end_sec) for segment in group)
        kind = "segment" if len(group) == 1 else "page"
        overview_id = group[0].segment_id if kind == "segment" else f"page_{index:04d}"
        group_frames = tuple(frame for frame in frames if start <= frame.virtual_time_sec < end)
        thumb = build_segment_overview_grid(
            group_frames,
            workspace.asset_root / "segment_overviews" / f"{overview_id}_overview.jpg",
        )
        row: dict[str, Any] = {
            "kind": kind,
            "overview_id": overview_id,
            "virtual_time_range": [round(start, 3), round(end, 3)],
            "duration_sec": round(max(0.0, end - start), 3),
            "overview_thumbnail_grid_path": str(thumb),
            "asr_short_summary": _asr_short_summary(asr_cues, start, end),
            "source_hint": "hidden_or_generic",
            "segment_ids": [segment.segment_id for segment in group],
        }
        if kind == "segment":
            row["segment_id"] = group[0].segment_id
        overviews.append(row)

    caption_navigation = _caption_navigation_available(workspace)
    return {
        "workspace_id": workspace.workspace_id,
        "workspace_duration_sec": workspace.manifest.duration_sec,
        "thumbnail_budget": budget,
        "thumbnail_count": len(overviews),
        "segment_overviews": overviews,
        "available_tools": [
            "open_segment",
            "inspect_window",
        ],
        "available_navigation": ["search_asr", *(["search_caption"] if caption_navigation else [])],
    }


def load_virtual_beats(path: Path) -> tuple[dict[str, Any], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(dict(item) for item in payload.get("beats", ()))


def _caption_navigation_available(workspace: VirtualVideoWorkspace) -> bool:
    captions_root = workspace.asset_root / "captions"
    return any(path.stat().st_size > 0 for path in captions_root.glob("passages.*.jsonl"))


def _to_runtime_beat(beat: VirtualBeat) -> Beat:
    return Beat(
        beat_id=beat.beat_id,
        chapter_id="",
        start_sec=beat.start_sec,
        end_sec=beat.end_sec,
        keyframe_path=beat.thumbnail_grid_path,
        asr_text=" ".join(str(cue.get("text", "")).strip() for cue in beat.asr_cues if str(cue.get("text", "")).strip()),
        ocr_text=(),
        frame_paths=tuple(frame.path for frame in beat.frame_refs),
        frame_times=tuple(frame.virtual_time_sec for frame in beat.frame_refs),
        asr_cues=tuple(_runtime_cue(cue) for cue in beat.asr_cues),
        ocr_cues=(),
    )


def _runtime_cue(cue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "start_sec": float(cue.get("start_sec", cue.get("start", 0.0)) or 0.0),
        "end_sec": float(cue.get("end_sec", cue.get("end", cue.get("start", 0.0))) or 0.0),
        "text": str(cue.get("text", "") or ""),
    }


def _chapters_for_beats(beats: Sequence[Beat], duration_sec: float) -> tuple[Chapter, ...]:
    if not beats:
        return ()
    return (
        Chapter(
            chapter_id="ch01",
            start_sec=0.0,
            end_sec=float(duration_sec),
            beat_ids=tuple(beat.beat_id for beat in beats),
            thumb_path=beats[0].keyframe_path,
        ),
    )


def _source_lineage(workspace: VirtualVideoWorkspace, start_sec: float, end_sec: float) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "segment_id": item.segment_id,
            "source_video_id": item.source_video_id,
            "source_path": item.source_path,
            "source_time_range": [item.source_start_sec, item.source_end_sec],
            "virtual_time_range": [item.virtual_start_sec, item.virtual_end_sec],
        }
        for item in virtual_to_source_windows(workspace.manifest, start_sec, end_sec)
    )


def _virtual_beat_payload(beat: VirtualBeat) -> dict[str, Any]:
    return {
        "beat_id": beat.beat_id,
        "virtual_time_range": [beat.start_sec, beat.end_sec],
        "thumbnail_grid_path": beat.thumbnail_grid_path,
        "thumbnail_grid_paths": list(beat.thumbnail_grid_paths),
        "frame_ids": [frame.frame_id for frame in beat.frame_refs],
        "source_lineage": [dict(item) for item in beat.source_lineage],
        "asr_cues": [dict(item) for item in beat.asr_cues],
    }


def _cue_overlaps(cue: Mapping[str, Any], start_sec: float, end_sec: float) -> bool:
    start = float(cue.get("start_sec", cue.get("start", 0.0)) or 0.0)
    end = float(cue.get("end_sec", cue.get("end", start)) or start)
    return min(end, end_sec) > max(start, start_sec)


def _asr_short_summary(cues: Sequence[Mapping[str, Any]], start_sec: float, end_sec: float, *, limit: int = 240) -> str:
    text = " ".join(str(cue.get("text", "")).strip() for cue in cues if _cue_overlaps(cue, start_sec, end_sec)).strip()
    return text[:limit]


def _select_grid_frames(frames: tuple[VirtualFrameRef, ...], *, max_frames: int) -> tuple[VirtualFrameRef, ...]:
    limit = max(1, int(max_frames))
    if len(frames) <= limit:
        return frames
    if limit == 1:
        return (frames[len(frames) // 2],)
    indexes = [round(i * (len(frames) - 1) / (limit - 1)) for i in range(limit)]
    return tuple(frames[int(index)] for index in indexes)


def _diagnostics(
    duration_sec: float,
    chapters: Sequence[Chapter],
    beats: Sequence[Beat],
    visual_index: VisualIndex,
    model: ModelClient,
) -> IndexDiagnostics:
    durations = tuple(max(0.0, beat.end_sec - beat.start_sec) for beat in beats)
    norms = np.linalg.norm(visual_index.embeddings, axis=1) if visual_index.embeddings.size else np.asarray([], dtype=np.float32)
    embedding_backend = str(getattr(model, "embed_model", model.__class__.__name__) or "unknown")
    return IndexDiagnostics(
        duration_sec=float(duration_sec),
        chapter_count=len(chapters),
        beat_count=len(beats),
        median_beat_sec=float(median(durations)) if durations else 0.0,
        max_beat_sec=float(max(durations)) if durations else 0.0,
        visual_index_dim=int(visual_index.embeddings.shape[1]) if visual_index.embeddings.ndim == 2 else 0,
        visual_embedding_norm_mean=float(norms.mean()) if norms.size else 0.0,
        embedding_backend=embedding_backend,
        index_mode="virtual-video",
        warnings=(),
    )
