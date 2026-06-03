"""Run an iterative long-video tool-use smoke case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..iterative_smoke import IterativeSmokeConfig, run_qwen_iterative_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--media-path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--duration-sec", required=True, type=float)
    parser.add_argument("--window-sec", default=30.0, type=float)
    parser.add_argument("--max-rounds", default=4, type=int)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--run-id", default="qwen_vl_iterative_smoke")
    args = parser.parse_args()

    result = run_qwen_iterative_smoke(
        IterativeSmokeConfig(
            model_path=args.model_path,
            media_path=args.media_path,
            question=args.question,
            duration_sec=args.duration_sec,
            window_sec=args.window_sec,
            run_id=args.run_id,
            max_rounds=args.max_rounds,
        ),
        base_dir=Path(args.base_dir),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
