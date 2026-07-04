from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcah.agent import VideoAgent
from vcah.evals import run_videomme_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the slim evidence-seeking video agent.")
    parser.add_argument("--video", help="Path to a video file.")
    parser.add_argument("--question", help="Question to answer about the video.")
    parser.add_argument("--videomme-root", help="VideoMME root containing cases.json.")
    parser.add_argument("--case", help="VideoMME case id.")
    parser.add_argument("--run-dir", default="runs/default", help="Output directory for cold_index/ and run/.")
    parser.add_argument("--duration-sec", type=float, default=None, help="Optional duration override for smoke tests.")
    parser.add_argument("--index-mode", default="fast", help="Cold index mode label.")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
