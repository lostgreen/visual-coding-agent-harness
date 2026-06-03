"""Compare direct full-video description against map-first video exploration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..comparison import DescriptionComparisonConfig, run_qwen_description_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--media-path", required=True)
    parser.add_argument("--question", default="Describe the video.")
    parser.add_argument("--duration-sec", required=True, type=float)
    parser.add_argument("--window-sec", default=30.0, type=float)
    parser.add_argument("--max-rounds", default=4, type=int)
    parser.add_argument("--direct-nframes", default=64, type=int)
    parser.add_argument("--max-pixels", default=151200, type=int)
    parser.add_argument("--extract-clips", action="store_true")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--run-id", default="qwen_description_comparison")
    args = parser.parse_args()

    result = run_qwen_description_comparison(
        DescriptionComparisonConfig(
            model_path=args.model_path,
            media_path=args.media_path,
            question=args.question,
            duration_sec=args.duration_sec,
            window_sec=args.window_sec,
            run_id=args.run_id,
            max_rounds=args.max_rounds,
            direct_nframes=args.direct_nframes,
            max_pixels=args.max_pixels,
            extract_clips=args.extract_clips,
        ),
        base_dir=Path(args.base_dir),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
