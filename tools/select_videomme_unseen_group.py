from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


def main() -> None:
    args = _parse_args()
    rows = _load_rows(Path(args.dataset_root))
    seen_case_ids, seen_video_ids = _seen_cases(tuple(Path(item) for item in args.run_root))
    selected = select_unseen_cases(
        rows,
        seen_case_ids=seen_case_ids,
        seen_video_ids=seen_video_ids,
        count=args.count,
        seed=args.seed,
    )
    payload = {
        "group_id": args.group_id,
        "description": "Video-MME long cases from unseen source videos, stratified by task type and domain.",
        "duration_category": "long",
        "construction": "source_only",
        "selection_seed": args.seed,
        "selection_policy": {
            "exclude_previously_run_case_ids": True,
            "exclude_previously_viewed_source_videos": True,
            "distinct_source_video_per_case": True,
            "stratify_by_task_type_and_domain": True,
        },
        "seen_case_count": len(seen_case_ids),
        "seen_source_video_count": len(seen_video_ids),
        "cases": [
            {
                "case_id": str(row["question_id"]),
                "domain": str(row.get("domain", "") or ""),
                "task_type": str(row.get("task_type", "") or ""),
                "source_video_id": str(row.get("videoID", "") or ""),
            }
            for row in selected
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "selected": len(selected), "seen_cases": len(seen_case_ids)}, sort_keys=True))


def select_unseen_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    seen_case_ids: set[str],
    seen_video_ids: set[str],
    count: int,
    seed: int,
) -> tuple[Mapping[str, Any], ...]:
    candidates = [
        row
        for row in rows
        if str(row.get("duration", "")).casefold() == "long"
        and str(row.get("question_id", "")) not in seen_case_ids
        and str(row.get("videoID", "")) not in seen_video_ids
    ]
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    selected: list[Mapping[str, Any]] = []
    used_videos: set[str] = set()
    task_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    task_types = sorted({str(row.get("task_type", "") or "") for row in candidates})

    for task_type in task_types:
        row = next(
            (
                item
                for item in candidates
                if str(item.get("task_type", "") or "") == task_type
                and str(item.get("videoID", "") or "") not in used_videos
            ),
            None,
        )
        if row is not None:
            _adopt(row, selected, used_videos, task_counts, domain_counts)
        if len(selected) >= count:
            return tuple(selected[:count])

    while len(selected) < count:
        pool = [row for row in candidates if str(row.get("videoID", "") or "") not in used_videos]
        if not pool:
            raise RuntimeError(f"Only found {len(selected)} unseen long-video sources; requested {count}")
        row = min(
            pool,
            key=lambda item: (
                task_counts.get(str(item.get("task_type", "") or ""), 0),
                domain_counts.get(str(item.get("domain", "") or ""), 0),
                str(item.get("question_id", "")),
            ),
        )
        _adopt(row, selected, used_videos, task_counts, domain_counts)
    return tuple(selected)


def _adopt(
    row: Mapping[str, Any],
    selected: list[Mapping[str, Any]],
    used_videos: set[str],
    task_counts: dict[str, int],
    domain_counts: dict[str, int],
) -> None:
    selected.append(row)
    used_videos.add(str(row.get("videoID", "") or ""))
    task = str(row.get("task_type", "") or "")
    domain = str(row.get("domain", "") or "")
    task_counts[task] = task_counts.get(task, 0) + 1
    domain_counts[domain] = domain_counts.get(domain, 0) + 1


def _seen_cases(run_roots: Sequence[Path]) -> tuple[set[str], set[str]]:
    case_ids: set[str] = set()
    video_ids: set[str] = set()
    for root in run_roots:
        if not root.exists():
            continue
        for path in root.rglob("case.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            case_id = str(payload.get("case_id", "") or "").strip()
            source_id = str(dict(payload.get("metadata", {}) or {}).get("source_video_id", "") or "").strip()
            if case_id:
                case_ids.add(case_id)
            if source_id:
                video_ids.add(source_id)
    return case_ids, video_ids


def _load_rows(dataset_root: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(dataset_root / "videomme" / "test-00000-of-00001.parquet")
    return [dict(row) for row in frame.to_dict("records")]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select reproducible unseen Video-MME long cases.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-root", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-id", default="videomme_long_unseen20_v5")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


if __name__ == "__main__":
    main()
