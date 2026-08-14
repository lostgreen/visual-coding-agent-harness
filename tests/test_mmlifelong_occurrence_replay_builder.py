from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_mmlifelong_occurrence_replay import build_replay_fixtures


CAPTION_DIGEST = "caption-digest"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _packet(*, query: str, occurrence_id: str) -> dict[str, object]:
    return {
        "queries": [query],
        "time_range": None,
        "segment_ids": [],
        "source_video_ids": [],
        "top_k": 12,
        "expand_neighbors": 0,
        "index_mode": "hybrid",
        "config_digest": CAPTION_DIGEST,
        "index_digest": "index",
        "query_fingerprint": query,
        "hits": [{"passage_id": f"p-{query}"}],
        "occurrence_set": {
            "candidates": [{"occurrence_id": occurrence_id}]
        },
        "rendered": "bounded caption",
    }


def _source_case(root: Path, case_id: str = "case-1") -> Path:
    run_dir = root / "cases" / case_id
    _write_json(run_dir / "prediction.json", {"case_id": case_id})
    _write_json(
        run_dir / "run_config.json",
        {
            "oracle_arm": "o0",
            "occurrence_method_arm": "a0",
            "caption_config_digest": CAPTION_DIGEST,
        },
    )
    _write_json(
        run_dir / "runtime_summary.json",
        {"no_oracle_runtime_gate": {"no_oracle_runtime_gate_passed": True}},
    )
    first = run_dir / "caption_search" / "first.json"
    second = run_dir / "caption_search" / "second.json"
    _write_json(first, _packet(query="first", occurrence_id="occ-1"))
    _write_json(second, _packet(query="second", occurrence_id="occ-2"))
    rows = [
        {
            "modality": "caption_search",
            "raw_output": json.dumps({"raw_output_pointer": str(first)}),
        },
        {
            "modality": "caption_search",
            "raw_output": json.dumps({"raw_output_pointer": str(first)}),
        },
        {
            "modality": "caption_search",
            "raw_output": json.dumps({"raw_output_pointer": str(second)}),
        },
    ]
    (run_dir / "observation_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return run_dir


def test_build_replay_fixtures_preserves_first_use_order_and_deduplicates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "fixtures"
    _source_case(source)

    manifest = build_replay_fixtures(
        source,
        output,
        caption_config_digest=CAPTION_DIGEST,
        expected_cases=1,
    )

    fixture = json.loads(
        (output / "cases" / "case-1.json").read_text(encoding="utf-8")
    )
    assert manifest["case_count"] == 1
    assert manifest["packet_count"] == 2
    assert [row["packet"]["queries"] for row in fixture["packets"]] == [
        ["first"],
        ["second"],
    ]
    assert fixture["packets"][0]["retrieval_identity_digest"]
    assert build_replay_fixtures(
        source,
        output,
        caption_config_digest=CAPTION_DIGEST,
        expected_cases=1,
    ) == manifest


def test_build_replay_fixtures_rejects_packet_outside_case(tmp_path: Path) -> None:
    source = tmp_path / "source"
    run_dir = _source_case(source)
    outside = tmp_path / "outside.json"
    _write_json(outside, _packet(query="outside", occurrence_id="occ-x"))
    (run_dir / "observation_log.jsonl").write_text(
        json.dumps(
            {
                "modality": "caption_search",
                "raw_output": json.dumps({"raw_output_pointer": str(outside)}),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escaped case directory"):
        build_replay_fixtures(
            source,
            tmp_path / "fixtures",
            caption_config_digest=CAPTION_DIGEST,
            expected_cases=1,
        )
