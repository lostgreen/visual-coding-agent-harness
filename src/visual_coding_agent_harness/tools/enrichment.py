"""Tools that enrich the mutable VideoMap workspace."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..open_questions import exploration_question
from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import ToolRegistry, tool
from ..video_map import VideoMapStore
from ..workspace import EvidenceWorkspace
from .frame_cache import FrameSampler, frame_cache_artifact_ref
from .segments import ClipExtractor, _clip_output_path, _extract_clip_ffmpeg


def build_video_enrichment_registry(
    *,
    video_map_store: VideoMapStore,
    backend: VisionLanguageBackend,
    workspace: EvidenceWorkspace | None = None,
    extract_clips: bool = False,
    clip_extractor: ClipExtractor | None = None,
    frame_sampler: FrameSampler | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="caption_segments", description="Caption selected VideoMap segments and write captions back into the map.")
    def caption_segments(
        segment_ids: Sequence[str] = (),
        question: str = "Create a concise search caption for this segment.",
        nframes: int = 8,
        max_pixels: int = 360 * 420,
        fps: float = 0.0,
        max_segments: int = 3,
    ) -> Mapping[str, object]:
        selected_segments = _select_segments(video_map_store=video_map_store, segment_ids=segment_ids, max_segments=max_segments)
        prompt_question = exploration_question(question)
        regions = []
        for segment in selected_segments:
            media_path: str | None = video_map_store.current.video_path
            media_type = "video"
            frame_paths: tuple[str, ...] = ()
            input_artifact = f"{media_path}#t={float(segment.start_sec):.3f},{float(segment.end_sec):.3f}"
            metadata = {
                "segment_id": segment.segment_id,
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "nframes": int(nframes),
                "max_pixels": int(max_pixels),
                "question": prompt_question,
            }
            if fps > 0:
                metadata["fps"] = float(fps)
            if frame_sampler is not None:
                frame_paths = tuple(
                    frame_sampler(
                        video_map_store.current.video_path,
                        float(segment.start_sec),
                        float(segment.end_sec),
                        int(nframes),
                    )
                )
                if frame_paths:
                    media_path = None
                    media_type = "video"
                    input_artifact = frame_cache_artifact_ref(
                        video_path=video_map_store.current.video_path,
                        start_sec=float(segment.start_sec),
                        end_sec=float(segment.end_sec),
                        frame_count=len(frame_paths),
                    )
                    metadata["source_video_path"] = video_map_store.current.video_path
                    metadata["frame_cache_policy"] = "precomputed_2fps"
                    metadata["frame_count"] = len(frame_paths)
            if not frame_paths and extract_clips:
                if workspace is None:
                    raise ValueError("extract_clips=True requires an EvidenceWorkspace")
                output_path = _clip_output_path(
                    workspace=workspace,
                    segment_id=segment.segment_id,
                    start_sec=float(segment.start_sec),
                    end_sec=float(segment.end_sec),
                )
                extractor = clip_extractor or _extract_clip_ffmpeg
                media_path = extractor(
                    video_map_store.current.video_path,
                    str(output_path),
                    float(segment.start_sec),
                    float(segment.end_sec),
                )
                input_artifact = media_path
                metadata["source_video_path"] = video_map_store.current.video_path
                metadata["clip_path"] = media_path
            response = backend.generate(
                BackendRequest(
                    task="caption_segment",
                    prompt=_caption_segment_prompt(
                        segment_id=segment.segment_id,
                        start_sec=segment.start_sec,
                        end_sec=segment.end_sec,
                        question=prompt_question,
                    ),
                    media_path=media_path,
                    media_type=media_type,
                    frames=frame_paths,
                    max_new_tokens=192,
                    metadata=metadata,
                )
            )
            caption = response.text.strip()
            updated = video_map_store.update_segment(segment.segment_id, low_fps_caption=caption)
            regions.append(
                {
                    "segment_id": updated.segment_id,
                    "start_sec": updated.start_sec,
                    "end_sec": updated.end_sec,
                    "low_fps_caption": updated.low_fps_caption,
                    "nframes": int(nframes),
                    "max_pixels": int(max_pixels),
                    "input_artifact": input_artifact,
                    "frame_count": len(frame_paths),
                    "frame_cache_policy": metadata.get("frame_cache_policy", ""),
                    "clip_path": metadata.get("clip_path", ""),
                }
            )

        count = len(regions)
        return {
            "claim": f"Captioned {count} segment{'s' if count != 1 else ''} and updated VideoMap low_fps_caption.",
            "confidence": 0.7 if count else 0.0,
            "input_artifacts": [str(region["input_artifact"]) for region in regions],
            "regions": regions,
            "limitations": "VLM-generated coarse captions; use focused QA or OCR/ASR tools for precise claims.",
        }

    @tool(name="ingest_segment_metadata", description="Write external ASR/OCR/entities/caption results into one VideoMap segment.")
    def ingest_segment_metadata(
        segment_id: str,
        low_fps_caption: str = "",
        asr_text: str = "",
        ocr_text: str = "",
        entities: Sequence[str] = (),
    ) -> Mapping[str, object]:
        updated = video_map_store.update_segment(
            segment_id,
            low_fps_caption=low_fps_caption or None,
            asr_text=asr_text or None,
            ocr_text=ocr_text or None,
            entities=list(entities) if entities else None,
        )
        return {
            "claim": f"Updated {segment_id} with external metadata.",
            "confidence": 1.0,
            "input_artifacts": [video_map_store.current.video_path],
            "regions": [updated.to_dict()],
            "limitations": "Metadata ingest trusts the caller-provided external tool output.",
        }

    registry.register(caption_segments)
    registry.register(ingest_segment_metadata)
    return registry


def _select_segments(*, video_map_store: VideoMapStore, segment_ids: Sequence[str], max_segments: int):
    if segment_ids:
        return [video_map_store.current.get(segment_id) for segment_id in segment_ids[:max_segments]]
    return list(video_map_store.current.segments[:max_segments])


def _caption_segment_prompt(*, segment_id: str, start_sec: float, end_sec: float, question: str) -> str:
    return (
        "Caption task: create a concise searchable description using only visible evidence.\n"
        f"Target segment: {segment_id} [{start_sec:.3f}, {end_sec:.3f}] seconds.\n"
        "Avoid unsupported identities, OCR text, or temporal claims.\n"
        f"Question: {question}"
    )
