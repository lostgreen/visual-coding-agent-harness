from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcah.agent import VideoAgent
from vcah.evals import run_videomme_case
from vcah.multiround import VirtualVideoMultiRoundDriver
from vcah.virtual_index import build_virtual_beat_index
from vcah.virtual_video import VirtualVideoWorkspace, materialize_lowfps_frame_cache
from vcah.videomme_virtual import build_videomme_smoke_workspaces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the slim evidence-seeking video agent.")
    subparsers = parser.add_subparsers(dest="command")
    vv_build = subparsers.add_parser(
        "vv-build-videomme",
        help="Build VideoMME virtual-video smoke workspaces.",
        description="Build three VideoMME virtual-video smoke workspaces.",
    )
    vv_build.add_argument("--dataset-root", required=True, help="VideoMME snapshot root containing videomme/, video/, subtitle/.")
    vv_build.add_argument("--out-dir", required=True, help="Output directory for per-case virtual workspaces.")
    vv_build.add_argument("--seed", type=int, default=20260707)

    vv_index = subparsers.add_parser(
        "vv-index",
        help="Build a virtual-video cold index.",
        description="Materialize low-fps frames and build beat thumbnail cold index for one workspace.",
    )
    vv_index.add_argument("--workspace", required=True, help="VirtualVideoWorkspace directory.")
    vv_index.add_argument("--low-fps", type=float, default=0.5)
    vv_index.add_argument("--beat-sec", type=float, default=18.0)

    vv_run = subparsers.add_parser(
        "vv-run",
        help="Run virtual-video multi-round investigation for one workspace.",
        description="Run Reasoner/Investigator multi-round loop for one VirtualVideoWorkspace.",
    )
    vv_run.add_argument("--workspace", required=True)
    vv_run.add_argument("--max-rounds", type=int, default=4)
    vv_run.add_argument("--max-investigations", type=int, default=20)

    vv_run_all = subparsers.add_parser(
        "vv-run-all",
        help="Run virtual-video multi-round investigation for all child workspaces.",
        description="Run all child directories containing case.json and virtual_timeline.json.",
    )
    vv_run_all.add_argument("--workspace-root", required=True)
    vv_run_all.add_argument("--max-rounds", type=int, default=4)
    vv_run_all.add_argument("--max-investigations", type=int, default=20)
    parser.add_argument("--video", help="Path to a video file.")
    parser.add_argument("--question", help="Question to answer about the video.")
    parser.add_argument("--videomme-root", help="VideoMME root containing cases.json.")
    parser.add_argument("--case", help="VideoMME case id.")
    parser.add_argument("--run-dir", default="runs/default", help="Output directory for cold_index/ and run/.")
    parser.add_argument("--duration-sec", type=float, default=None, help="Optional duration override for smoke tests.")
    parser.add_argument("--index-mode", default="fast", help="Cold index mode label.")
    args = parser.parse_args()

    if args.command == "vv-build-videomme":
        workspaces = build_videomme_smoke_workspaces(Path(args.dataset_root), Path(args.out_dir), seed=args.seed)
        print(
            json.dumps(
                {
                    "workspace_count": len(workspaces),
                    "workspaces": [str(workspace.root_dir) for workspace in workspaces],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "vv-index":
        workspace = VirtualVideoWorkspace.load(Path(args.workspace))
        frames = materialize_lowfps_frame_cache(workspace, fps=args.low_fps)
        result = build_virtual_beat_index(workspace, frames, beat_sec=args.beat_sec)
        print(
            json.dumps(
                {
                    "workspace": str(workspace.root_dir),
                    "frames": len(frames),
                    "beats": len(result.virtual_beats),
                    "cold_index": str(workspace.cold_index_dir),
                    "beat_index": str(result.beat_index_path),
                    "timeline_grid": str(result.timeline_grid_path),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "vv-run":
        workspace = VirtualVideoWorkspace.load(Path(args.workspace))
        result = VirtualVideoMultiRoundDriver(max_rounds=args.max_rounds, max_investigations=args.max_investigations).run(workspace)
        print(json.dumps(_vv_result_payload(result), ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "vv-run-all":
        root = Path(args.workspace_root)
        summaries = []
        for child in sorted(path for path in root.iterdir() if path.is_dir()):
            if not (child / "case.json").exists() or not (child / "virtual_timeline.json").exists():
                continue
            workspace = VirtualVideoWorkspace.load(child)
            result = VirtualVideoMultiRoundDriver(max_rounds=args.max_rounds, max_investigations=args.max_investigations).run(workspace)
            summaries.append(_vv_result_payload(result))
        print(
            json.dumps(
                {
                    "case_count": len(summaries),
                    "correct": sum(1 for item in summaries if item["correct"]),
                    "cases": summaries,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
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


def _vv_result_payload(result) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "answer": result.answer,
        "citations": list(result.citations),
        "correct": result.correct,
        "verified": result.verified,
        "verification_reason": result.verification_reason,
        "rounds": result.rounds,
        "accepted_investigations": result.accepted_investigations,
    }


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_") or "case"


if __name__ == "__main__":
    main()
