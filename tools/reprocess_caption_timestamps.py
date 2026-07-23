from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from vcah.caption_schema import (
    CaptionChunkV1,
    count_repaired_timestamp_tokens,
    count_timestamp_tokens,
    parse_timestamp_anchors,
    split_caption_passages,
)
from vcah.caption_store import CaptionStore


def reprocess_caption_timestamps(asset_root: Path, config_digest: str) -> dict[str, object]:
    store = CaptionStore(asset_root, config_digest, eager_exports=False)
    replacements = {}
    parse_counts = {"strict": 0, "filtered_invalid": 0, "chunk_fallback": 0}
    repair_count = 0
    anchor_count = 0
    for cache_key, chunk in store.successful_records():
        duration = chunk.virtual_end_sec - chunk.virtual_start_sec
        parse_status = "strict"
        warning = ""
        try:
            anchors = parse_timestamp_anchors(
                chunk.text_raw,
                chunk_start_sec=chunk.virtual_start_sec,
                chunk_end_sec=chunk.virtual_end_sec,
                strict=True,
                repair_duplicate_minute_hour=True,
                repair_short_timestamp=True,
            )
        except ValueError as exc:
            anchors = parse_timestamp_anchors(
                chunk.text_raw,
                chunk_start_sec=chunk.virtual_start_sec,
                chunk_end_sec=chunk.virtual_end_sec,
                strict=False,
                repair_duplicate_minute_hour=True,
                repair_short_timestamp=True,
            )
            parse_status = "filtered_invalid" if anchors else "chunk_fallback"
            warning = str(exc)[:300]
        repairs = count_repaired_timestamp_tokens(
            chunk.text_raw,
            chunk_duration_sec=duration,
        )
        metadata = {
            **dict(chunk.metadata),
            "timestamp_parse_status": parse_status,
            "timestamp_parse_warning": warning,
            "timestamp_token_count": count_timestamp_tokens(chunk.text_raw),
            "timestamp_repair_count": repairs,
            "valid_timestamp_anchor_count": len(anchors),
        }
        updated = CaptionChunkV1(
            **{
                **asdict(chunk),
                "timestamp_anchors": anchors,
                "metadata": metadata,
            }
        )
        replacements[cache_key] = (updated, split_caption_passages(updated))
        parse_counts[parse_status] += 1
        repair_count += repairs
        anchor_count += len(anchors)

    replaced = store.replace_successful_records(replacements)
    store.flush_exports()
    return {
        "asset_root": str(Path(asset_root)),
        "config_digest": str(config_digest),
        "reprocessed_chunks": replaced,
        "anchor_count": anchor_count,
        "timestamp_repair_count": repair_count,
        "parse_status_counts": parse_counts,
        "chunks_path": str(store.chunks_path),
        "passages_path": str(store.passages_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically rebuild ReMA Caption timestamps and passages from text_raw."
    )
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--config-digest", required=True)
    args = parser.parse_args()
    result = reprocess_caption_timestamps(Path(args.asset_root), args.config_digest)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
