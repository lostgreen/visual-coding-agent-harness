#!/usr/bin/env python3
"""Freeze the zero-call WP17-3 120-second slot-memory manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.occurrence_negative_sidecar import file_sha256
from vcah.wp17_slot_protocol import build_wp17_3_protocol_manifest


def run(args: argparse.Namespace) -> Path:
    protocol_path = Path(args.protocol)
    timeline_path = Path(args.timeline_manifest)
    dense_root = Path(args.dense_root)
    dense_report_path = dense_root / "wp17_dense_ocr_report.json"
    dense_audit_path = Path(args.dense_audit)
    input_sha256 = {
        "timeline": file_sha256(timeline_path),
        "dense_report": file_sha256(dense_report_path),
        "dense_audit": file_sha256(dense_audit_path),
    }
    manifest = build_wp17_3_protocol_manifest(
        _read_json(protocol_path),
        timeline=_read_json(timeline_path),
        dense_report=_read_json(dense_report_path),
        dense_audit=_read_json(dense_audit_path),
        input_sha256=input_sha256,
        source_commit=str(args.source_commit),
    )
    manifest["provenance"]["protocol_sha256"] = file_sha256(protocol_path)
    out_root = Path(args.out_root)
    report_path = out_root / "wp17_3_slot_protocol_manifest.json"
    markdown_path = out_root / "wp17_3_slot_protocol_manifest.md"
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError("WP17-3 slot protocol output already exists")
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, manifest)
    markdown_path.write_text(_render_markdown(manifest), encoding="utf-8")
    counts = dict(manifest["counts"])
    print(
        "WP17_3_SLOT_PROTOCOL_FREEZE_DONE "
        f"decision={manifest['decision']} segments={counts['segments']} "
        f"base_calls={counts['base_model_calls']} canary_calls={counts['canary_base_model_calls']} "
        f"gate={str(manifest['structural_gate_passed']).lower()} model_calls=0",
        flush=True,
    )
    return report_path


def _render_markdown(manifest: Mapping[str, Any]) -> str:
    counts = dict(manifest["counts"])
    return "\n".join(
        (
            "# MM-Lifelong WP17-3 120s Slot Protocol Freeze",
            "",
            f"- Decision: `{manifest['decision']}`",
            f"- Structural gate: `{str(manifest['structural_gate_passed']).lower()}`",
            f"- Windows / 120s segments: `{counts['windows']} / {counts['segments']}`",
            f"- Three-arm base calls / hard cap: `{counts['base_model_calls']} / {counts['model_call_hard_cap']}`",
            f"- Consecutive canary segments / base calls / hard cap: `{counts['canary_segments']} / {counts['canary_base_model_calls']} / {counts['canary_model_call_hard_cap']}`",
            "- This step freezes inputs, state semantics, schemas, budgets, arm order, and gates with zero model calls.",
            "",
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--timeline-manifest", required=True)
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--dense-audit", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
