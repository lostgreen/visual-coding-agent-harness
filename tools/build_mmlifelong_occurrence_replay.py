#!/usr/bin/env python3
"""Rebuild frozen occurrence replay fixtures from completed A0 run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.occurrence_agent import (
    assert_no_oracle_packet,
    occurrence_replay_identity,
)


REQUIRED_PACKET_FIELDS = frozenset(
    {
        "config_digest",
        "hits",
        "index_digest",
        "occurrence_set",
        "query_fingerprint",
        "rendered",
    }
)


def build_replay_fixtures(
    run_root: Path,
    output_root: Path,
    *,
    caption_config_digest: str,
    expected_cases: int | None = None,
) -> dict[str, Any]:
    source_root = Path(run_root).resolve()
    destination = Path(output_root)
    prediction_paths = sorted(source_root.glob("cases/*/prediction.json"))
    if expected_cases is not None and len(prediction_paths) != expected_cases:
        raise ValueError(
            f"expected {expected_cases} completed cases, found {len(prediction_paths)}"
        )
    if not prediction_paths:
        raise ValueError("source run has no completed predictions")

    fixture_rows: list[dict[str, Any]] = []
    for prediction_path in prediction_paths:
        run_dir = prediction_path.parent
        prediction = _read_json(prediction_path)
        case_id = str(prediction.get("case_id", run_dir.name) or run_dir.name)
        config = _read_json(run_dir / "run_config.json")
        runtime = _read_json(run_dir / "runtime_summary.json")
        _validate_source_case(
            case_id,
            config=config,
            runtime=runtime,
            caption_config_digest=caption_config_digest,
        )
        packets = _ordered_caption_packets(
            run_dir,
            caption_config_digest=caption_config_digest,
        )
        fixture = {
            "schema_version": "MMLifelongOccurrenceReplayV1",
            "case_id": case_id,
            "caption_config_digest": caption_config_digest,
            "source_method_arm": "a0",
            "packets": [
                {
                    "ordinal": index,
                    "retrieval_identity_digest": occurrence_replay_identity(packet),
                    "packet": packet,
                }
                for index, packet in enumerate(packets, start=1)
            ],
        }
        assert_no_oracle_packet(fixture, surface="rebuilt_occurrence_replay")
        fixture_path = destination / "cases" / f"{case_id}.json"
        _write_json_idempotent(fixture_path, fixture)
        fixture_rows.append(
            {
                "case_id": case_id,
                "packet_count": len(packets),
                "sha256": _sha256(fixture_path),
            }
        )

    manifest = {
        "schema_version": "MMLifelongOccurrenceReplayManifestV1",
        "caption_config_digest": caption_config_digest,
        "case_count": len(fixture_rows),
        "packet_count": sum(row["packet_count"] for row in fixture_rows),
        "packet_count_min": min(row["packet_count"] for row in fixture_rows),
        "packet_count_max": max(row["packet_count"] for row in fixture_rows),
        "source_run_root": str(source_root),
        "cases": fixture_rows,
    }
    _write_json_idempotent(destination / "manifest.json", manifest)
    return manifest


def _validate_source_case(
    case_id: str,
    *,
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    caption_config_digest: str,
) -> None:
    if str(config.get("oracle_arm", "") or "") != "o0":
        raise ValueError(f"{case_id}: source oracle arm is not o0")
    if str(config.get("occurrence_method_arm", "") or "") != "a0":
        raise ValueError(f"{case_id}: source occurrence arm is not a0")
    if str(config.get("caption_config_digest", "") or "") != caption_config_digest:
        raise ValueError(f"{case_id}: source Caption digest mismatch")
    raw_gate = runtime.get("no_oracle_runtime_gate", {})
    gate = raw_gate if isinstance(raw_gate, Mapping) else {}
    if gate.get("no_oracle_runtime_gate_passed") is not True:
        raise ValueError(f"{case_id}: source no-oracle runtime gate failed")


def _ordered_caption_packets(
    run_dir: Path,
    *,
    caption_config_digest: str,
) -> tuple[dict[str, Any], ...]:
    observation_path = Path(run_dir) / "observation_log.jsonl"
    rows = _read_jsonl(observation_path)
    caption_root = (Path(run_dir) / "caption_search").resolve()
    seen_paths: set[Path] = set()
    packets: list[dict[str, Any]] = []
    for row in rows:
        if row.get("modality") != "caption_search":
            continue
        pointer = _raw_output_pointer(row.get("raw_output"))
        if not pointer:
            raise ValueError(f"{run_dir.name}: Caption observation missing packet pointer")
        packet_path = Path(pointer)
        if not packet_path.is_absolute():
            packet_path = Path(run_dir) / packet_path
        packet_path = packet_path.resolve()
        if packet_path.parent != caption_root:
            raise ValueError(f"{run_dir.name}: Caption packet escaped case directory")
        if packet_path in seen_paths:
            continue
        seen_paths.add(packet_path)
        packet = _read_json(packet_path)
        missing = sorted(REQUIRED_PACKET_FIELDS - set(packet))
        if missing:
            raise ValueError(
                f"{run_dir.name}: Caption packet missing fields: {', '.join(missing)}"
            )
        if str(packet.get("config_digest", "") or "") != caption_config_digest:
            raise ValueError(f"{run_dir.name}: Caption packet digest mismatch")
        assert_no_oracle_packet(packet, surface="rebuilt_occurrence_replay_packet")
        packets.append(packet)
    if not packets:
        raise ValueError(f"{run_dir.name}: no Caption packets available for priming")
    return tuple(packets)


def _raw_output_pointer(raw_output: Any) -> str:
    if isinstance(raw_output, Mapping):
        value = raw_output
    elif isinstance(raw_output, str) and raw_output.strip():
        parsed = json.loads(raw_output)
        value = parsed if isinstance(parsed, Mapping) else {}
    else:
        value = {}
    return str(value.get("raw_output_pointer", "") or "").strip()


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return tuple(rows)


def _write_json_idempotent(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    rendered = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    if target.is_file():
        if target.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite different fixture: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(target)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--caption-config-digest", required=True)
    parser.add_argument("--expected-cases", type=int)
    args = parser.parse_args()
    manifest = build_replay_fixtures(
        Path(args.run_root),
        Path(args.output_root),
        caption_config_digest=args.caption_config_digest,
        expected_cases=args.expected_cases,
    )
    print(
        json.dumps(
            {
                "case_count": manifest["case_count"],
                "packet_count": manifest["packet_count"],
                "packet_count_min": manifest["packet_count_min"],
                "packet_count_max": manifest["packet_count_max"],
                "output_root": str(Path(args.output_root)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
