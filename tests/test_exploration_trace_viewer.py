from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


def _load_viewer_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "tools" / "build_exploration_trace_viewer.py"
    spec = importlib.util.spec_from_file_location("exploration_trace_viewer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VIEWER = _load_viewer_module()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_viewer_bundles_complete_exploration_story(tmp_path: Path) -> None:
    workspace = tmp_path / "case-0117"
    asset_root = tmp_path / "day-assets"
    frame_path = workspace / "observations" / "frames" / "observed.png"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )

    _write_json(
        workspace / "case.json",
        {
            "case_id": "mmlifelong-game-test-0117",
            "question": "What changes after the player opens the door?",
            "gold_answer": "The light turns green.",
            "gold_clue_intervals": [[36000, 36020]],
            "asset_ref": str(asset_root),
        },
    )
    _write_json(
        asset_root / "virtual_timeline.json",
        {
            "duration_sec": 90000,
            "segments": [
                {
                    "segment_id": "segment-001",
                    "source_video_id": "video-a",
                    "virtual_start_sec": 0,
                    "virtual_end_sec": 45000,
                },
                {
                    "segment_id": "segment-002",
                    "source_video_id": "video-b",
                    "virtual_start_sec": 45000,
                    "virtual_end_sec": 90000,
                },
            ],
        },
    )
    _write_json(
        workspace / "run_config.json",
        {
            "caption_config_digest": "caption-digest",
            "caption_index_digests": ["index-digest"],
            "models": {"reasoner": "reasoner-model", "investigator": "vision-model"},
        },
    )
    _write_jsonl(
        asset_root / "captions" / "passages.caption-digest.jsonl",
        [
            {
                "passage_id": "passage-1",
                "text": "A red door opens and the indicator light changes from red to green.",
            }
        ],
    )
    _write_jsonl(
        workspace / "observation_log.jsonl",
        [
            {
                "round_id": "round_1",
                "task_id": "caption-query",
                "attempt_id": "attempt-1",
                "sampling_config": {
                    "mode": "search_caption",
                    "index_mode": "hybrid",
                    "top_k": 3,
                    "queries": ["door light state change"],
                    "hits": [
                        {"passage_id": "passage-1", "range": [35990, 36030], "score": 0.93}
                    ],
                },
            },
            {
                "round_id": "round_2",
                "task_id": "visual-query",
                "attempt_id": "attempt-2",
                "modality": "visual",
                "inspected_ranges": [[36000, 36020]],
                "sampling_config": {"mode": "window", "fps": 1},
                "raw_output": "The observed frame confirms that the indicator turns green.",
            },
        ],
    )
    _write_jsonl(
        workspace / "observations" / "window_frame_manifest.jsonl",
        [
            {
                "path": str(frame_path),
                "query_id": "visual-query",
                "virtual_time_sec": 36005,
                "source_video_id": "video-a",
                "source_time_sec": 605,
            }
        ],
    )
    _write_jsonl(workspace / "exploration_ledger.jsonl", [{"range": [36000, 36020]}])
    _write_jsonl(
        workspace / "interactions.jsonl",
        [
            {
                "type": "investigator_observation",
                "query_id": "visual-query",
                "parsed": {
                    "summary": "The indicator changes from red to green.",
                    "observations": [
                        {"time_sec": 36005, "description": "The indicator is green."}
                    ],
                    "events": [
                        {"time_range": [36000, 36020], "description": "The light changes color."}
                    ],
                    "uncertainties": ["The exact switch frame is between two samples."],
                },
            }
        ],
    )
    _write_jsonl(
        workspace / "workspace_ops.jsonl",
        [
            {
                "round_id": "3",
                "operations": [
                    {
                        "op": "add_claim",
                        "claim": {
                            "claim_id": "c1",
                            "text": "The indicator changes from red to green.",
                            "confidence": "high",
                            "source": "observation",
                            "cites": ["attempt-2"],
                            "time_anchor": [36000, 36020],
                        },
                    }
                ],
                "result": {"accepted": True, "applied_count": 1, "revision": 1},
            }
        ],
    )
    _write_json(
        workspace / "run_summary.json",
        {
            "answer": "The light turns green.",
            "answer_present": True,
            "reference_valid": True,
            "reference_reason": "visual evidence supports the answer",
            "rounds": 3,
            "investigation_count": 2,
            "supporting_intervals": [[36000, 36020]],
            "citations": ["evidence-1"],
            "evidence": [
                {
                    "evidence_id": "evidence-1",
                    "virtual_time_range": [36000, 36020],
                    "attempt_id": "attempt-2",
                    "pointer": "observation-2",
                    "modality": "visual",
                    "summary": "The light changes from red to green.",
                }
            ],
            "evaluation": {"accuracy_score": 1.0},
            "trace": [
                {
                    "type": "reasoner_decision",
                    "round": 1,
                    "action": "investigate",
                    "remaining_budget": 4,
                    "workspace_revision": 0,
                    "tasks": [
                        {
                            "query_id": "caption-query",
                            "inspection_mode": "search_caption",
                            "caption_queries": ["door light state change"],
                            "index_mode": "hybrid",
                            "top_k": 3,
                        }
                    ],
                },
                {
                    "type": "investigator_batch",
                    "round": 1,
                    "outcomes": [
                        {
                            "query_id": "caption-query",
                            "status": "completed",
                            "attempt_ids": ["attempt-1"],
                            "evidence_ids": [],
                            "consumes_budget": True,
                        }
                    ],
                },
                {
                    "type": "reasoner_decision",
                    "round": 2,
                    "action": "investigate",
                    "remaining_budget": 3,
                    "workspace_revision": 0,
                    "tasks": [
                        {
                            "query_id": "visual-query",
                            "inspection_mode": "window",
                            "segment_id": "segment-001",
                            "time_range": [36000, 36020],
                            "sampling_floor_fps": 1,
                        }
                    ],
                },
                {
                    "type": "investigator_batch",
                    "round": 2,
                    "outcomes": [
                        {
                            "query_id": "visual-query",
                            "status": "completed",
                            "attempt_ids": ["attempt-2"],
                            "evidence_ids": ["evidence-1"],
                            "consumes_budget": True,
                        }
                    ],
                },
                {
                    "type": "reasoner_decision",
                    "round": 3,
                    "action": "answer",
                    "remaining_budget": 2,
                    "workspace_revision": 1,
                    "supporting_claim_ids": ["c1"],
                    "tasks": [],
                },
                {
                    "type": "reference_integrity_check",
                    "round": 3,
                    "passed": True,
                    "reason": "supported",
                },
                {"type": "answer_outcome", "answer": "The light turns green."},
            ],
        },
    )

    out_dir = tmp_path / "viewer"
    result = VIEWER.build_viewer(workspace, out_dir, max_frames=8, title="Trace Demo")
    manifest = json.loads((out_dir / "viewer_manifest.json").read_text(encoding="utf-8"))
    html = (out_dir / "index.html").read_text(encoding="utf-8")
    embedded = html.split(
        '<script id="trace-data" type="application/json">', 1
    )[1].split("</script>", 1)[0]
    payload = json.loads(embedded)

    assert result["case_id"] == "mmlifelong-game-test-0117"
    assert result["frame_count"] == 1
    assert manifest["event_count"] >= 4
    assert manifest["schema_version"] == 2
    assert manifest["step_count"] == 7
    assert manifest["round_count"] == 3
    assert manifest["duration_sec"] == 90000
    assert (out_dir / "assets" / "frame-001.png").is_file()
    assert "What changes after the player opens the door?" in html
    assert "A red door opens" in html
    assert "任务级探索链路" in html
    assert payload["rounds"][0]["tasks"][0]["id"] == "caption-query"
    assert payload["rounds"][0]["tasks"][0]["result"]["hitCount"] == 1
    assert payload["rounds"][0]["tasks"][0]["result"]["topHits"][0]["selectedNext"] is True
    assert payload["rounds"][1]["tasks"][0]["result"]["frameCount"] == 1
    assert payload["rounds"][1]["tasks"][0]["result"]["uncertainties"] == [
        "The exact switch frame is between two samples."
    ]
    assert payload["rounds"][1]["tasks"][0]["evidenceIds"] == ["evidence-1"]
    assert payload["rounds"][2]["claims"][0]["id"] == "c1"
    assert "__TRACE_DATA__" not in html
