"""Run a single VLM-agent tool-use smoke case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..vlm_smoke import SmokeConfig, run_qwen_vl_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--media-path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--media-type", default="video", choices=["image", "video"])
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--run-id", default="qwen_vl_smoke")
    args = parser.parse_args()

    result = run_qwen_vl_smoke(
        SmokeConfig(
            model_path=args.model_path,
            media_path=args.media_path,
            question=args.question,
            media_type=args.media_type,
            run_id=args.run_id,
        ),
        base_dir=Path(args.base_dir),
    )
    print(
        json.dumps(
            {
                "answer": result.answer,
                "program": list(result.program),
                "observation_ids": list(result.program_result.observation_ids),
                "assignments": dict(result.program_result.assignments),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
