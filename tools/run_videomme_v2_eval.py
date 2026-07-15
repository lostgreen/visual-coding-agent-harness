#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

from vcah.direct_baseline import (
    align_direct_evidence_to_frames,
    annotate_uniform_frames,
    bounded_uniform_frame_count,
    build_direct_prompt,
    format_timestamped_asr,
    materialize_uniform_frames,
    request_direct_answer,
)
from vcah.video import probe_duration
from vcah.videomme_v2 import (
    VideoMMEV2Question,
    load_case_group,
    load_questions,
    load_subtitle_cues,
    options_mapping,
    score_videomme_v2_answer,
    summarize_group_results,
)
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)

from run_virtual_videomme_interactive import (
    OpenAICompatibleVisionClient,
    _run_case_batch,
    ensure_index,
    load_role_clients,
    run_case,
)


DEFAULT_DATASET_ROOT = Path("/ytech_m2v5_hdd/workspace/kling_mm/Datasets/Video-MME-v2")
DEFAULT_OUT_ROOT = Path("/m2v_intern/xuboshen/zgw/VideoAgent/videomme_v2_diverse20_v1")


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    group = load_case_group(Path(args.case_group))
    questions = {item.question_id: item for item in load_questions(dataset_root)}
    missing = [case_id for case_id in group["case_ids"] if case_id not in questions]
    if missing:
        raise KeyError(f"Unknown VideoMME v2 case ids: {missing}")
    selected = tuple(questions[case_id] for case_id in group["case_ids"])
    video_context = _load_video_context(dataset_root, selected)

    if args.method == "direct":
        if not args.config and not args.investigator_config:
            raise ValueError("Direct mode requires --config or --investigator-config")
        api = OpenAICompatibleVisionClient.from_yaml(
            Path(args.investigator_config or args.config),
            section="investigator_api",
        )
        _prebuild_direct_frames(
            selected,
            video_context=video_context,
            cache_root=out_root / "cache" / "direct",
            max_frames=int(args.max_frames),
            max_fps=float(args.max_fps),
            max_image_edge=int(args.max_image_edge),
            rebuild=bool(args.rebuild),
            workers=min(5, int(args.workers)),
        )

        def run_one(case_id: str) -> Mapping[str, Any]:
            return _run_direct_case(
                questions[case_id],
                out_root=out_root,
                cache_root=out_root / "cache" / "direct",
                video_context=video_context,
                api=api,
                max_frames=int(args.max_frames),
                max_fps=float(args.max_fps),
                max_image_edge=int(args.max_image_edge),
                force_contact_sheets=bool(args.force_contact_sheets),
                resume=bool(args.resume),
            )

    else:
        reasoner_api, investigator_api = load_role_clients(
            shared_config=args.config,
            reasoner_config=args.reasoner_config,
            investigator_config=args.investigator_config,
        )

        def run_one(case_id: str) -> Mapping[str, Any]:
            return _run_agent_case(
                questions[case_id],
                out_root=out_root,
                video_context=video_context,
                reasoner_api=reasoner_api,
                investigator_api=investigator_api,
                segment_sec=float(args.segment_sec),
                low_fps=float(args.low_fps),
                beat_sec=float(args.beat_sec),
                max_rounds=int(args.max_rounds),
                max_investigations=int(args.max_investigations),
                rebuild=bool(args.rebuild),
                rebuild_index=bool(args.rebuild_index),
                resume=bool(args.resume),
            )

    results = _run_case_batch(group["case_ids"], run_one, workers=int(args.workers))
    summary = {
        "method": args.method,
        "dataset_root": str(dataset_root),
        "selection_policy": group.get("selection_policy", {}),
        **summarize_group_results(results, group=group),
    }
    _write_json(out_root / f"{args.method}_summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, ensure_ascii=False, indent=2, sort_keys=True))


def _load_video_context(
    dataset_root: Path,
    questions: Sequence[VideoMMEV2Question],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for video_id in dict.fromkeys(item.video_id for item in questions):
        video_path = Path(dataset_root) / "videos" / f"{video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        result[video_id] = {
            "video_path": video_path,
            "duration_sec": float(probe_duration(str(video_path))),
            "asr_cues": load_subtitle_cues(Path(dataset_root) / "subtitle.zip", video_id),
        }
    return result


def _prebuild_direct_frames(
    questions: Sequence[VideoMMEV2Question],
    *,
    video_context: Mapping[str, Mapping[str, Any]],
    cache_root: Path,
    max_frames: int,
    max_fps: float,
    max_image_edge: int,
    rebuild: bool,
    workers: int,
) -> None:
    video_ids = tuple(dict.fromkeys(item.video_id for item in questions))

    def build(video_id: str) -> Mapping[str, Any]:
        context = video_context[video_id]
        frame_count = bounded_uniform_frame_count(
            float(context["duration_sec"]),
            max_frames=max_frames,
            max_fps=max_fps,
        )
        frames = materialize_uniform_frames(
            video_path=Path(context["video_path"]),
            duration_sec=float(context["duration_sec"]),
            out_dir=Path(cache_root) / video_id / "frames",
            frame_count=frame_count,
            max_image_edge=max_image_edge,
            rebuild=rebuild,
        )
        annotate_uniform_frames(frames, Path(cache_root) / video_id / "labeled_frames", rebuild=rebuild)
        return {"video_id": video_id, "frame_count": frame_count}

    _run_case_batch(video_ids, build, workers=workers)


def _run_direct_case(
    question: VideoMMEV2Question,
    *,
    out_root: Path,
    cache_root: Path,
    video_context: Mapping[str, Mapping[str, Any]],
    api: OpenAICompatibleVisionClient,
    max_frames: int,
    max_fps: float,
    max_image_edge: int,
    force_contact_sheets: bool,
    resume: bool,
) -> dict[str, Any]:
    case_dir = Path(out_root) / "direct_cases" / question.question_id
    result_path = case_dir / "result.json"
    if resume and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    case_dir.mkdir(parents=True, exist_ok=True)
    context = video_context[question.video_id]
    duration_sec = float(context["duration_sec"])
    frame_count = bounded_uniform_frame_count(duration_sec, max_frames=max_frames, max_fps=max_fps)
    frame_rows = materialize_uniform_frames(
        video_path=Path(context["video_path"]),
        duration_sec=duration_sec,
        out_dir=Path(cache_root) / question.video_id / "frames",
        frame_count=frame_count,
        max_image_edge=max_image_edge,
    )
    submitted = annotate_uniform_frames(frame_rows, Path(cache_root) / question.video_id / "labeled_frames")
    options = options_mapping(question.options)
    asr_text = format_timestamped_asr(context["asr_cues"])
    prompt = build_direct_prompt(
        question=question.question,
        options=options,
        frame_rows=frame_rows,
        asr_text=asr_text,
    )
    started = time.time()
    try:
        parsed, raw, input_mode, submitted_paths = request_direct_answer(
            api=api,
            prompt=prompt,
            frame_paths=tuple(Path(str(item["path"])) for item in submitted),
            sheet_dir=case_dir / "contact_sheets",
            force_contact_sheets=force_contact_sheets,
            max_tokens=1400,
        )
        parsed = align_direct_evidence_to_frames(parsed, frame_rows)
        answer = str(parsed.get("answer", "") or "")
        error = ""
    except Exception as exc:  # pragma: no cover - remote runtime failure path
        parsed, raw, input_mode, submitted_paths = {}, "", "", ()
        answer = ""
        error = f"{type(exc).__name__}: {exc}"
        (case_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    result = {
        "case_id": question.question_id,
        "video_id": question.video_id,
        "question": question.question,
        "options": options,
        "gold": question.answer,
        "answer": answer,
        "correct": score_videomme_v2_answer(answer, question.answer),
        "rationale": str(parsed.get("rationale", "") or ""),
        "evidence": list(parsed.get("evidence", ()) or ()),
        "duration_sec": round(duration_sec, 3),
        "frame_count": frame_count,
        "effective_fps": round(frame_count / duration_sec, 6),
        "input_mode": input_mode,
        "submitted_image_count": len(submitted_paths),
        "latency_sec": round(time.time() - started, 3),
        "error": error,
        "metadata": dict(question.metadata),
    }
    _write_json(case_dir / "response.json", {"raw": raw, "parsed": parsed})
    _write_json(result_path, result)
    return result


def _run_agent_case(
    question: VideoMMEV2Question,
    *,
    out_root: Path,
    video_context: Mapping[str, Mapping[str, Any]],
    reasoner_api: OpenAICompatibleVisionClient,
    investigator_api: OpenAICompatibleVisionClient,
    segment_sec: float,
    low_fps: float,
    beat_sec: float,
    max_rounds: int,
    max_investigations: int,
    rebuild: bool,
    rebuild_index: bool,
    resume: bool,
) -> dict[str, Any]:
    workspace_root = Path(out_root) / "workspaces" / question.question_id
    summary_path = workspace_root / "run_summary.json"
    if resume and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    context = video_context[question.video_id]
    duration_sec = float(context["duration_sec"])
    if rebuild or not (workspace_root / "virtual_timeline.json").exists():
        segments = _source_segments(
            question.video_id,
            Path(context["video_path"]),
            duration_sec=duration_sec,
            segment_sec=segment_sec,
        )
        manifest = VirtualVideoManifest(workspace_id=question.question_id, segments=segments)
        case = VirtualVideoCase(
            case_id=question.question_id,
            question=question.question,
            options=options_mapping(question.options),
            gold=question.answer,
            target_segment_id=segments[0].segment_id,
            target_virtual_interval=(0.0, duration_sec),
            metadata={"dataset": "VideoMME-v2", "video_id": question.video_id, **dict(question.metadata)},
        )
        workspace = VirtualVideoWorkspace.create(workspace_root, manifest=manifest, case=case)
        workspace.write_asr_virtual_cues(tuple(context["asr_cues"]))
    else:
        workspace = VirtualVideoWorkspace.load(workspace_root)
    ensure_index(workspace, low_fps=low_fps, beat_sec=beat_sec, rebuild=rebuild_index)
    started = time.time()
    try:
        result = run_case(
            workspace,
            reasoner_api=reasoner_api,
            investigator_api=investigator_api,
            max_rounds=max_rounds,
            max_investigations=max_investigations,
        )
        error = ""
    except Exception as exc:  # pragma: no cover - remote runtime failure path
        error = f"{type(exc).__name__}: {exc}"
        (workspace_root / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {
            "case_id": question.question_id,
            "video_id": question.video_id,
            "correct": False,
            "answer": "",
            "selected_option": "",
            "verified": False,
            "error": error,
        }
    return {
        "case_id": result.case_id,
        "video_id": question.video_id,
        "question": question.question,
        "options": options_mapping(question.options),
        "gold": question.answer,
        "answer": result.answer,
        "selected_option": result.selected_option,
        "correct": result.correct,
        "verified": result.verified,
        "answer_mode": result.answer_mode,
        "grounding_status": result.grounding_status,
        "retrieval_status": result.retrieval_status,
        "verification_reason": result.verification_reason,
        "rounds": result.rounds,
        "accepted_investigations": result.accepted_investigations,
        "latency_sec": round(time.time() - started, 3),
        "workspace": str(workspace_root),
        "error": error,
        "metadata": dict(question.metadata),
    }


def _source_segments(
    video_id: str,
    video_path: Path,
    *,
    duration_sec: float,
    segment_sec: float,
) -> tuple[VirtualVideoSegment, ...]:
    width = max(60.0, float(segment_sec))
    segments = []
    start = 0.0
    while start < duration_sec - 1e-6:
        end = min(duration_sec, start + width)
        segments.append(
            VirtualVideoSegment(
                segment_id=f"seg_{len(segments) + 1:04d}",
                source_video_id=video_id,
                source_path=str(video_path),
                source_start_sec=start,
                source_end_sec=end,
                virtual_start_sec=start,
                virtual_end_sec=end,
                role="content",
            )
        )
        start = end
    return tuple(segments)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare direct uniform frames and the current VCAH Agent on VideoMME v2.")
    parser.add_argument("--method", choices=("direct", "agent"), required=True)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--case-group", required=True)
    parser.add_argument("--config", help="Shared or direct API config.")
    parser.add_argument("--reasoner-config")
    parser.add_argument("--investigator-config")
    parser.add_argument("--workers", type=int, default=5, help="Parallel case workers, clamped to 1-16.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--max-fps", type=float, default=1.0)
    parser.add_argument("--max-image-edge", type=int, default=512)
    parser.add_argument("--force-contact-sheets", action="store_true")
    parser.add_argument("--segment-sec", type=float, default=180.0)
    parser.add_argument("--low-fps", type=float, default=0.1)
    parser.add_argument("--beat-sec", type=float, default=60.0)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--max-investigations", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    main()
