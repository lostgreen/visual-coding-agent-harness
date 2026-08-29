#!/usr/bin/env python3
"""Freeze the question-blind WP17 local construction timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.occurrence_negative_sidecar import file_sha256
from vcah.virtual_video import VirtualVideoWorkspace
from vcah.wp17_dense_ocr import build_local_timeline


def run(args: argparse.Namespace) -> Path:
    spec_path = Path(args.protocol_spec)
    if file_sha256(spec_path) != str(args.expected_protocol_sha256):
        raise ValueError("WP17-1 protocol SHA mismatch")
    spec = _read_json(spec_path)
    workspace = VirtualVideoWorkspace.load(Path(args.workspace_root))
    timeline = build_local_timeline(spec, manifest=workspace.manifest)
    if not timeline["structural_gate_passed"]:
        raise RuntimeError("WP17-1 timeline structural gate failed")
    report = {
        **timeline,
        "provenance": {
            "source_commit": str(args.source_commit),
            "protocol_sha256": file_sha256(spec_path),
            "workspace_id": workspace.workspace_id,
            "question_visible_to_preparation": False,
            "options_visible_to_preparation": False,
            "gold_answer_visible_to_preparation": False,
            "target_entity_aliases_visible_to_preparation": False,
            "day_test140_accessed": False,
            "week_accessed": False,
        },
    }
    out_root = Path(args.out_root)
    report_path = out_root / "timeline_manifest.json"
    markdown_path = out_root / "timeline_manifest.md"
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError("WP17-1 timeline output already exists")
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    counts = report["counts"]
    print(
        "WP17_TIMELINE_DONE "
        f"cases={counts['cases']} windows={counts['merged_windows']} "
        f"slices={counts['timeline_slices']} duration_sec={counts['scoped_duration_sec']} "
        f"sample_points={counts['expected_sample_points']} gate=true",
        flush=True,
    )
    return report_path


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


def _render_markdown(report: Mapping[str, Any]) -> str:
    counts = dict(report["counts"])
    hours = float(counts["scoped_duration_sec"]) / 3600.0
    return "\n".join(
        (
            "# MM-Lifelong WP17-1 Local Timeline",
            "",
            f"- Cases: `{counts['cases']}`",
            f"- Merged windows / source slices: `{counts['merged_windows']} / {counts['timeline_slices']}`",
            f"- Scoped duration: `{hours:.3f} h`",
            f"- Expected 1fps points: `{counts['expected_sample_points']}`",
            f"- Structural gate: `{str(report['structural_gate_passed']).lower()}`",
            "- Development annotations only define scope and are not supplied to the OCR reader.",
            "- No question, options, answer, target aliases, source paths, frames, Day-test140, or Week data are persisted.",
            "",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-spec", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
