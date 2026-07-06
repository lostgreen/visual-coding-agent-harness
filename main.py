from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcah.agent import VideoAgent
from vcah.evals import run_videomme_case
from vcah.xlebench import (
    LifeLogColdIndex,
    LifeLogColdIndexBuilder,
    LifeLogIndexConfig,
    diagnose_cold_recall,
    load_xlebench_manifest,
    write_diagnose_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the slim evidence-seeking video agent.")
    subparsers = parser.add_subparsers(dest="command")
    xle_index = subparsers.add_parser(
        "xle-index",
        help="Build an X-LeBench lifelog cold index.",
        description="Build an X-LeBench lifelog cold index.",
    )
    _add_xle_manifest_args(xle_index)
    _add_xle_index_args(xle_index, default_run_dir="runs/xle-index")

    xle_diagnose = subparsers.add_parser(
        "xle-diagnose",
        help="Run X-LeBench cold-recall diagnostics from an existing index.",
        description="Run X-LeBench cold-recall diagnostics from an existing index.",
    )
    _add_xle_manifest_args(xle_diagnose)
    xle_diagnose.add_argument("--run-dir", default="runs/xle-index", help="Directory containing lifelog_index.json.")
    xle_diagnose.add_argument("--top-k", type=int, nargs="+", default=[5, 20], help="Recall cutoffs.")
    xle_diagnose.add_argument("--build", action="store_true", help="Build or resume the index before diagnosing.")
    _add_xle_index_args(xle_diagnose, include_run_dir=False)
    parser.add_argument("--video", help="Path to a video file.")
    parser.add_argument("--question", help="Question to answer about the video.")
    parser.add_argument("--videomme-root", help="VideoMME root containing cases.json.")
    parser.add_argument("--case", help="VideoMME case id.")
    parser.add_argument("--run-dir", default="runs/default", help="Output directory for cold_index/ and run/.")
    parser.add_argument("--duration-sec", type=float, default=None, help="Optional duration override for smoke tests.")
    parser.add_argument("--index-mode", default="fast", help="Cold index mode label.")
    args = parser.parse_args()

    if args.command == "xle-index":
        run_dir = Path(args.run_dir)
        manifest = _load_xle_manifest_from_args(args)
        index = LifeLogColdIndexBuilder(manifest, _xle_index_config(args)).build(run_dir, resume=not args.no_resume)
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "segments": len(index.segments),
                    "beats": sum(len(segment.index.beats) for segment in index.segments),
                    "lifelog_index": str(run_dir / "lifelog_index.json"),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "xle-diagnose":
        run_dir = Path(args.run_dir)
        manifest = _load_xle_manifest_from_args(args)
        if args.build:
            index = LifeLogColdIndexBuilder(manifest, _xle_index_config(args)).build(run_dir, resume=not args.no_resume)
        else:
            index = LifeLogColdIndex.load(run_dir)
        report = diagnose_cold_recall(index, manifest.cases, top_ks=args.top_k)
        write_diagnose_report(report, run_dir / "xle_diagnose.json")
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    run_dir = Path(args.run_dir)
    if args.videomme_root and args.case:
        result = run_videomme_case(Path(args.videomme_root), args.case, run_dir=run_dir, duration_sec=args.duration_sec)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if not args.video or not args.question:
        parser.error("--video and --question are required unless --videomme-root and --case are provided")

    answer = VideoAgent().ask(
        args.video,
        args.question,
        run_dir=run_dir,
        duration_sec=args.duration_sec,
        index_mode=args.index_mode,
    )
    print(json.dumps({"answer": answer.answer, "citations": list(answer.citations)}, ensure_ascii=False, indent=2, sort_keys=True))


def _add_xle_manifest_args(command: argparse.ArgumentParser) -> None:
    command.add_argument("xle_root", help="X-LeBench root containing cases/annotations JSON or JSONL.")
    command.add_argument("--annotation-file", help="Explicit X-LeBench annotation JSON/JSONL path.")
    command.add_argument("--video-template", help="Video path template, e.g. /data/{video_uid}.mp4.")
    command.add_argument("--default-duration-sec", type=float, default=None, help="Fallback duration for records without metadata.")


def _add_xle_index_args(command: argparse.ArgumentParser, *, default_run_dir: str = "runs/xle-index", include_run_dir: bool = True) -> None:
    if include_run_dir:
        command.add_argument("--run-dir", default=default_run_dir, help="Output directory.")
    command.add_argument("--max-range-sec", type=float, default=60.0, help="Maximum cold index range length.")
    command.add_argument("--max-beat-sec", type=float, default=60.0, help="Maximum beat length.")
    command.add_argument("--no-resume", action="store_true", help="Rebuild segment indexes even when artifacts exist.")


def _load_xle_manifest_from_args(args: argparse.Namespace):
    return load_xlebench_manifest(
        Path(args.xle_root),
        video_template=args.video_template,
        annotation_file=Path(args.annotation_file) if args.annotation_file else None,
        default_duration_sec=args.default_duration_sec,
    )


def _xle_index_config(args: argparse.Namespace) -> LifeLogIndexConfig:
    return LifeLogIndexConfig(
        max_range_sec=args.max_range_sec,
        max_beat_sec=args.max_beat_sec,
        index_mode="xle-cold-mvp",
    )


if __name__ == "__main__":
    main()
