#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw


def main() -> None:
    args = _parse_args()
    workspace = Path(args.workspace).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    query_ids = tuple(args.query_ids)

    run_summary = _read_json(workspace / "run_summary.json")
    case = _read_json(workspace / "case.json")
    trace = _read_jsonl(workspace / "interactions.jsonl")
    frame_rows = _read_jsonl(workspace / "observations" / "window_frame_manifest.jsonl")

    selected = {}
    for query_id in query_ids:
        rows = [row for row in frame_rows if row.get("query_id") == query_id]
        rows.sort(key=lambda row: float(row.get("virtual_time_sec") or 0.0))
        if not rows:
            continue
        selected[query_id] = rows
        _write_contact_sheet(rows, out_dir / f"{query_id}_contact_sheet.jpg", max_frames=int(args.max_sheet_frames))
        _copy_keyframes(rows, out_dir / "keyframes" / query_id, max_frames=int(args.max_keyframes))

    (out_dir / "summary.md").write_text(
        _summary_markdown(case=case, run_summary=run_summary, trace=trace, selected=selected),
        encoding="utf-8",
    )
    (out_dir / "trace_excerpt.json").write_text(
        json.dumps(_trace_excerpt(trace), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "query_ids": list(selected)}, ensure_ascii=False, indent=2))


def _write_contact_sheet(rows: Sequence[Mapping[str, Any]], out_path: Path, *, max_frames: int) -> None:
    picked = _sample(rows, max_frames=max_frames)
    thumb = (224, 126)
    caption_h = 34
    cols = min(4, max(1, len(picked)))
    rows_count = max(1, math.ceil(len(picked) / cols))
    canvas = Image.new("RGB", (cols * thumb[0], rows_count * (thumb[1] + caption_h)), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(picked):
        x = (idx % cols) * thumb[0]
        y = (idx // cols) * (thumb[1] + caption_h)
        path = Path(str(row.get("path", "")))
        if path.exists():
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail(thumb)
                cell = Image.new("RGB", thumb, color=(20, 20, 20))
                cell.paste(image, ((thumb[0] - image.width) // 2, (thumb[1] - image.height) // 2))
                canvas.paste(cell, (x, y))
        label = f"vt {float(row.get('virtual_time_sec') or 0):.1f}s | src {float(row.get('source_time_sec') or 0):.1f}s"
        draw.text((x + 4, y + thumb[1] + 4), label, fill=(20, 20, 20))
        draw.text((x + 4, y + thumb[1] + 18), str(row.get("segment_id", "")), fill=(80, 80, 80))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=90)


def _copy_keyframes(rows: Sequence[Mapping[str, Any]], out_dir: Path, *, max_frames: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(_sample(rows, max_frames=max_frames), start=1):
        src = Path(str(row.get("path", "")))
        if not src.exists():
            continue
        dst = out_dir / f"{idx:02d}_vt{float(row.get('virtual_time_sec') or 0):.1f}_src{float(row.get('source_time_sec') or 0):.1f}.jpg"
        shutil.copy2(src, dst)


def _summary_markdown(
    *,
    case: Mapping[str, Any],
    run_summary: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    lines = [
        f"# {case.get('case_id', '')}: {case.get('question', '')}",
        "",
        f"- Answer: {run_summary.get('answer')}",
        f"- Gold: {case.get('gold')}",
        f"- Correct: {run_summary.get('correct')}",
        f"- Rounds: {run_summary.get('rounds')}",
        f"- Investigations: {run_summary.get('investigation_count')}",
        f"- References valid: {run_summary.get('reference_valid')}",
        f"- Citations: {', '.join(run_summary.get('citations') or [])}",
        "",
        "## Reasoner Behavior",
    ]
    for idx, row in enumerate(trace, start=1):
        if row.get("type") == "reasoner_workspace" and (row.get("parsed") or {}).get("action") == "investigate":
            tasks = (row.get("parsed") or {}).get("tasks") or []
            lines.append(f"- Step {idx}: Reasoner requested {len(tasks)} investigation task(s).")
            for task in tasks:
                lines.append(
                    f"  - {task.get('query_id')}: segment={task.get('segment_id')} time_range={task.get('time_range')} goal={task.get('goal')}"
                )
        elif row.get("type") == "reasoner_answer":
            parsed = row.get("parsed") or {}
            lines.append(f"- Step {idx}: Reasoner answered {parsed.get('answer')} citing {parsed.get('citations')}.")

    lines.append("")
    lines.append("## Evidence Summaries")
    for item in run_summary.get("evidence", ()) or ():
        lines.append(f"- {item.get('evidence_id')} | range={item.get('virtual_time_range')}")
        lines.append(f"  - {item.get('summary')}")

    lines.append("")
    lines.append("## Frame Assets")
    for query_id, rows in selected.items():
        start = min(float(row.get("virtual_time_sec") or 0.0) for row in rows)
        end = max(float(row.get("virtual_time_sec") or 0.0) for row in rows)
        source = rows[0].get("source_video_id", "")
        lines.append(f"- `{query_id}_contact_sheet.jpg`: {len(rows)} frames, virtual {start:.1f}-{end:.1f}s, source `{source}`.")

    lines.append("")
    lines.append("## Suggested PPT Message")
    lines.append(
        "- This run did not use a structured scholar identity table. It answered correctly by multi-round exploration plus natural-language visual summarization."
    )
    lines.append(
        "- The strongest cited evidence is `ev_r1_t1_001` and `ev_r2_t1_001`; however, the scholar descriptions are partly inconsistent, so this is a good slide for motivating explicit entity tracking/dedup."
    )
    return "\n".join(lines) + "\n"


def _trace_excerpt(trace: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in trace:
        parsed = row.get("parsed") or {}
        rows.append(
            {
                "type": row.get("type"),
                "query_id": row.get("query_id"),
                "window": row.get("window"),
                "tasks": parsed.get("tasks"),
                "selected_window": {key: parsed.get(key) for key in ("start_sec", "end_sec", "reason") if key in parsed},
                "summary": parsed.get("summary"),
                "answer": parsed.get("answer"),
                "citations": parsed.get("citations"),
            }
        )
    return rows


def _sample(rows: Sequence[Mapping[str, Any]], *, max_frames: int) -> tuple[Mapping[str, Any], ...]:
    if len(rows) <= max_frames:
        return tuple(rows)
    count = max(1, int(max_frames))
    return tuple(rows[round(idx * (len(rows) - 1) / (count - 1))] for idx in range(count))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PPT-friendly assets for one virtual video case.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--query-ids", nargs="+", required=True)
    parser.add_argument("--max-sheet-frames", type=int, default=24)
    parser.add_argument("--max-keyframes", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    main()
