#!/usr/bin/env python3
"""Freeze the WP17 Week-dev60 and Week-holdout140 query split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from vcah.wp17_week_split import build_week_query_manifests


def run(args: argparse.Namespace) -> dict[str, Path]:
    metadata_path = Path(args.week_metadata)
    rows = _load_week_rows(metadata_path)
    manifests = build_week_query_manifests(
        rows,
        dev_count=int(args.dev_count),
        expected_count=int(args.expected_count),
        seed=int(args.seed),
    )
    out_root = Path(args.out_root)
    paths = {
        "week_dev": out_root / f"week_dev{int(args.dev_count)}.json",
        "week_holdout": out_root
        / f"week_holdout{int(args.expected_count) - int(args.dev_count)}.json",
        "protocol": out_root / "week_query_split_protocol.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("WP17 Week split output already exists")
    _write_json(paths["week_dev"], manifests["week_dev"])
    _write_json(paths["week_holdout"], manifests["week_holdout"])
    protocol = dict(manifests["protocol"])
    protocol["source_metadata_sha256"] = _file_sha256(metadata_path)
    protocol["manifest_sha256"] = {
        name: _file_sha256(paths[name]) for name in ("week_dev", "week_holdout")
    }
    _write_json(paths["protocol"], protocol)
    print(
        "WP17_WEEK_QUERY_SPLIT_DONE "
        f"decision={protocol['decision']} dev={protocol['development_count']} "
        f"holdout={protocol['holdout_count']} "
        f"gate={str(protocol['structural_gate_passed']).lower()} model_calls=0",
        flush=True,
    )
    return paths


def _load_week_rows(path: Path) -> tuple[dict[str, str], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Week metadata must be a JSON list")
    rows = []
    for position, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Week metadata row {position} is not an object")
        source_index = raw.get("index", position)
        rows.append(
            {
                "case_id": _case_id(source_index),
                "question_type": str(raw.get("question_type") or "Unknown"),
                "case_sha256": _digest(raw),
            }
        )
    return tuple(rows)


def _case_id(source_index: Any) -> str:
    try:
        suffix = f"{int(source_index):04d}"
    except (TypeError, ValueError):
        suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", str(source_index)).strip("_")
        suffix = suffix or "unknown"
    return f"mmlifelong-week-test-{suffix}"


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-metadata", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--dev-count", type=int, default=60)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
