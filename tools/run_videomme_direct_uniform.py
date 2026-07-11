from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from vcah.direct_baseline import (
    align_direct_evidence_to_frames,
    annotate_uniform_frames,
    build_direct_prompt,
    format_timestamped_asr,
    materialize_uniform_frames,
    request_direct_answer,
    summarize_results,
)
from vcah.multiround import _score_answer
from vcah.video import probe_duration
from vcah.virtual_video import VirtualVideoSegment, load_srt_as_virtual_cues

from run_virtual_videomme_interactive import (
    OpenAICompatibleVisionClient,
    _load_case_group,
    _load_rows,
    _options_mapping,
    _run_case_batch,
)


DEFAULT_DATASET_ROOT = Path(
    "/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/"
    "datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b"
)
DEFAULT_OUT_ROOT = Path("/m2v_intern/xuboshen/zgw/VideoAgent/videomme_direct_512_v3")


def run_direct_case(
    *,
    row: Mapping[str, Any],
    dataset_root: Path,
    out_root: Path,
    api: Any,
    frame_count: int = 512,
    max_image_edge: int = 512,
    rebuild: bool = False,
    force_contact_sheets: bool = False,
    duration_probe: Callable[[str], float] = probe_duration,
    frame_materializer: Callable[..., tuple[dict[str, Any], ...]] = materialize_uniform_frames,
    requester: Callable[..., tuple[dict[str, Any], str, str, tuple[str, ...]]] = request_direct_answer,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    case_id = str(row.get("question_id", "") or "")
    video_id = str(row.get("videoID", "") or "")
    if not case_id or not video_id:
        raise ValueError("VideoMME row requires question_id and videoID")
    case_dir = Path(out_root) / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(dataset_root) / "video" / f"{video_id}.mp4"
    duration_sec = float(duration_probe(str(video_path)))
    frame_rows = frame_materializer(
        video_path=video_path,
        duration_sec=duration_sec,
        out_dir=case_dir / "frames",
        frame_count=int(frame_count),
        max_image_edge=int(max_image_edge),
        rebuild=bool(rebuild),
    )
    submitted_frame_rows = annotate_uniform_frames(
        frame_rows,
        case_dir / "labeled_frames",
        rebuild=bool(rebuild),
    )
    segment = VirtualVideoSegment(
        segment_id="direct_source",
        source_video_id=video_id,
        source_path=str(video_path),
        source_start_sec=0.0,
        source_end_sec=duration_sec,
        virtual_start_sec=0.0,
        virtual_end_sec=duration_sec,
        role="content",
    )
    cues = load_srt_as_virtual_cues(Path(dataset_root) / "subtitle" / f"{video_id}.srt", segment)
    asr_text = format_timestamped_asr(cues)
    (case_dir / "asr_prompt.txt").write_text(asr_text, encoding="utf-8")
    options = _options_mapping(row.get("options", ()))
    prompt = build_direct_prompt(
        question=str(row.get("question", "") or ""),
        options=options,
        frame_rows=frame_rows,
        asr_text=asr_text,
    )
    start = clock()
    parsed, raw, input_mode, submitted_paths = requester(
        api=api,
        prompt=prompt,
        frame_paths=tuple(Path(str(item["path"])) for item in submitted_frame_rows),
        sheet_dir=case_dir / "contact_sheets",
        force_contact_sheets=bool(force_contact_sheets),
        max_tokens=900,
    )
    parsed = align_direct_evidence_to_frames(parsed, frame_rows)
    latency_sec = round(float(clock() - start), 3)
    gold = str(row.get("answer", "") or "")
    answer = str(parsed.get("answer", "") or "")
    result = {
        "case_id": case_id,
        "video_id": video_id,
        "question": str(row.get("question", "") or ""),
        "options": options,
        "gold": gold,
        "answer": answer,
        "correct": _score_answer(answer, gold, options),
        "rationale": str(parsed.get("rationale", "") or ""),
        "evidence": list(parsed.get("evidence", ()) or ()),
        "input_mode": input_mode,
        "latency_sec": latency_sec,
        "error": "",
    }
    _write_json(case_dir / "response.json", {"raw": raw, "parsed": parsed})
    _write_json(
        case_dir / "request_metadata.json",
        {
            "case_id": case_id,
            "video_id": video_id,
            "duration_sec": round(duration_sec, 3),
            "frame_count": len(frame_rows),
            "submitted_image_count": len(submitted_paths),
            "input_mode": input_mode,
            "max_image_edge": int(max_image_edge),
            "frame_labels_burned": True,
            "asr_cue_count": len(cues),
            "asr_chars": len(asr_text),
            "prompt_chars": len(prompt),
            "model": str(getattr(api, "model", type(api).__name__)),
        },
    )
    _write_json(case_dir / "result.json", result)
    return result


def build_group_summary(results: Sequence[Mapping[str, Any]], *, group_id: str) -> dict[str, Any]:
    return {"group_id": str(group_id), **summarize_results(results)}


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    group = _load_case_group(Path(args.case_group)) if args.case_group else None
    case_ids = tuple(group["case_ids"]) if group else tuple(args.case_ids or ())
    if not case_ids:
        raise ValueError("Provide --case-group or --case-ids")
    group_id = str(group["group_id"] if group else "direct-uniform-custom")
    rows = _load_rows(dataset_root)
    by_id = {str(row.get("question_id", "") or ""): row for row in rows}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise KeyError(f"Unknown VideoMME case ids: {missing}")
    api = OpenAICompatibleVisionClient.from_yaml(Path(args.config))

    def run_one(case_id: str) -> Mapping[str, Any]:
        try:
            return run_direct_case(
                row=by_id[case_id],
                dataset_root=dataset_root,
                out_root=out_root,
                api=api,
                frame_count=int(args.frames),
                max_image_edge=int(args.max_image_edge),
                rebuild=bool(args.rebuild),
                force_contact_sheets=bool(args.force_contact_sheets),
            )
        except Exception as exc:
            result = {
                "case_id": case_id,
                "answer": "",
                "correct": False,
                "input_mode": "",
                "latency_sec": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
            case_dir = out_root / "cases" / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_json(case_dir / "result.json", result)
            return result

    results = _run_case_batch(case_ids, run_one, workers=int(args.workers))
    summary = build_group_summary(results, group_id=group_id)
    _write_json(out_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "group_id": group_id,
                "case_count": summary["case_count"],
                "correct": summary["correct"],
                "accuracy": summary["accuracy"],
                "failures": summary["failures"],
                "input_modes": summary["input_modes"],
                "summary_path": str(out_root / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct Gemini baseline on uniformly sampled VideoMME frames.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    cases = parser.add_mutually_exclusive_group(required=True)
    cases.add_argument("--case-group")
    cases.add_argument("--case-ids", nargs="+")
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--max-image-edge", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1, help="Parallel case workers, clamped to 1-16.")
    parser.add_argument("--force-contact-sheets", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
