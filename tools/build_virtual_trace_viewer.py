#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    workspaces = _workspace_dirs(run_root, args.case_ids)
    cases_html = []
    summary_rows = []
    for workspace in workspaces:
        bundle = AssetBundler(run_root=run_root, assets_dir=assets_dir, case_id=workspace.name)
        case_html, summary = _render_case(workspace, bundle)
        cases_html.append(case_html)
        summary_rows.append(summary)

    index_path = out_dir / "index.html"
    index_path.write_text(_page_html(run_root, summary_rows, cases_html), encoding="utf-8")
    _write_manifest(out_dir, run_root, summary_rows)
    zip_path = Path(args.zip_path).resolve() if args.zip_path else out_dir.with_suffix(".zip")
    _zip_dir(out_dir, zip_path)
    print(json.dumps({"viewer": str(index_path), "zip": str(zip_path), "case_count": len(workspaces)}, ensure_ascii=False, indent=2))


class AssetBundler:
    def __init__(self, *, run_root: Path, assets_dir: Path, case_id: str) -> None:
        self.run_root = run_root
        self.assets_dir = assets_dir
        self.case_id = case_id
        self._seen: dict[str, str] = {}

    def add(self, path: str | Path | None, *, label: str = "") -> str:
        if not path:
            return ""
        src = Path(str(path))
        if not src.exists() or not src.is_file():
            return ""
        key = str(src.resolve())
        if key in self._seen:
            return self._seen[key]
        suffix = src.suffix.lower() or ".jpg"
        safe_label = _safe_name(label or src.stem)
        rel = Path("assets") / self.case_id / f"{len(self._seen) + 1:04d}_{safe_label}{suffix}"
        dst = self.assets_dir / self.case_id / rel.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self._seen[key] = rel.as_posix()
        return rel.as_posix()


def _render_case(workspace: Path, bundle: AssetBundler) -> tuple[str, dict[str, Any]]:
    case = _read_json(workspace / "case.json")
    timeline = _read_json(workspace / "virtual_timeline.json")
    run_summary = _read_json(workspace / "run_summary.json")
    trace_rows = _read_jsonl(workspace / "interactions.jsonl")
    beat_index = _read_json(workspace / "beat_index.json")
    window_rows = _read_jsonl(workspace / "observations" / "window_frame_manifest.jsonl")
    evidence_rows = _read_jsonl(workspace / "evidence.jsonl")
    ledger_rows = _read_jsonl(workspace / "exploration_ledger.jsonl")

    segments = list(timeline.get("segments", ()))
    beats = list(beat_index.get("beats", ()))
    beats_by_segment = _beats_by_segment(beats)
    frames_by_query = _frames_by_query(window_rows)
    overview_images = _segment_overviews(workspace, bundle)
    evidence_by_query = _evidence_by_query(run_summary.get("evidence", ()))

    summary = {
        "case_id": workspace.name,
        "question": case.get("question", ""),
        "gold": case.get("gold", ""),
        "answer": run_summary.get("answer", ""),
        "correct": run_summary.get("correct"),
        "rounds": run_summary.get("rounds"),
        "accepted_investigations": run_summary.get("accepted_investigations"),
        "trace_lines": len(trace_rows),
        "evidence_records": len(evidence_rows),
        "exploration_visits": len(ledger_rows),
        "duration_sec": timeline.get("duration_sec"),
    }

    body = [
        f"<section class='case' id='{_id(workspace.name)}'>",
        f"<h2>{_e(workspace.name)} · {_e(case.get('question', ''))}</h2>",
        _summary_card(case, run_summary, timeline),
        "<h3>Reasoner Initial Input: Segment Overview</h3>",
        _overview_grid(overview_images, segments),
    ]

    select_events = [row for row in trace_rows if row.get("type") == "investigator_select_window"]
    preview_events = [row for row in trace_rows if row.get("type") == "investigator_preview"]
    evidence_events = [row for row in trace_rows if row.get("type") == "investigator_evidence"]
    select_by_query = {str(row.get("query_id")): row for row in select_events}
    preview_by_query = {str(row.get("query_id")): row for row in preview_events}
    evidence_event_by_query = {str(row.get("query_id")): row for row in evidence_events}

    reasoner_events = [row for row in trace_rows if row.get("type") in {"reasoner_investigate", "reasoner_answer"}]
    for event_index, reasoner_event in enumerate(reasoner_events, start=1):
        round_id = int(reasoner_event.get("round") or event_index)
        parsed = dict(reasoner_event.get("parsed") or {})
        is_answer = reasoner_event.get("type") == "reasoner_answer"
        body.append("<section class='round'>")
        body.append(f"<h3>Reasoner Round {round_id} · {'Answer' if is_answer else 'Investigate'}</h3>")
        body.append(_details("Reasoner Prompt", _pre(reasoner_event.get("prompt", ""))))
        body.append(_details("Reasoner Raw Output", _pre(reasoner_event.get("raw", ""))))
        body.append(_json_block(parsed))

        for task in parsed.get("tasks", ()) or ():
            query_id = str(task.get("query_id", ""))
            segment_id = str(task.get("segment_id", ""))
            body.append("<section class='task'>")
            body.append(f"<h4>Investigator Task {_e(query_id)} · Segment {_e(segment_id)}</h4>")
            body.append(_json_block(task))
            body.append("<h4>Investigator Input: open_segment Beat Thumbnails</h4>")
            body.append(_beat_grid(beats_by_segment.get(segment_id, ()), bundle))
            select = select_by_query.get(query_id, {})
            body.append("<h4>Investigator Window Selection</h4>")
            body.append(_json_block(select.get("parsed") or {}))
            body.append(_details("Window Selection Raw Output", _pre(select.get("raw", ""))))

            preview_event = preview_by_query.get(query_id, {})
            evidence_event = evidence_event_by_query.get(query_id, {})
            observation_id = str(evidence_event.get("observation_id") or preview_event.get("observation_id") or query_id)
            preview_query_id = str(
                evidence_event.get("preview_query_id") or preview_event.get("preview_query_id") or f"{query_id}_preview"
            )
            detail_query_id = str(evidence_event.get("detail_query_id") or query_id)
            body.append("<h4>inspect_window Frames</h4>")
            body.append(_frame_group("Preview 0.5fps", frames_by_query.get(preview_query_id, ()), bundle))
            if detail_query_id:
                body.append(_frame_group("Detail frames", frames_by_query.get(detail_query_id, ()), bundle))
            region_rows = tuple(
                {"path": path, "frame_id": Path(str(path)).stem, "fps_level": "region crop"}
                for path in evidence_event.get("region_frame_paths", ()) or ()
            )
            if region_rows:
                body.append(_frame_group("Region crops", region_rows, bundle))
            body.append(_details("Preview Observation Raw Output", _pre(preview_event.get("raw", ""))))
            body.append("<h4>Evidence Report</h4>")
            body.append(_json_block(evidence_by_query.get(observation_id) or evidence_by_query.get(query_id) or {}))
            body.append(_details("Investigator Evidence Raw Output", _pre(evidence_event.get("raw", ""))))
            body.append("</section>")
        body.append("</section>")

    body.extend(
        [
            _details("Structured Evidence Store", _json_block(evidence_rows)),
            _details("Exploration Ledger", _json_block(ledger_rows)),
            _details("Driver Run Summary", _json_block(run_summary)),
            "</section>",
        ]
    )
    return "\n".join(body), summary


def _summary_card(case: Mapping[str, Any], run_summary: Mapping[str, Any], timeline: Mapping[str, Any]) -> str:
    options = case.get("options") or {}
    option_lines = "".join(f"<li><b>{_e(k)}</b>: {_e(v)}</li>" for k, v in sorted(dict(options).items()))
    return (
        "<div class='summary'>"
        f"<div><b>Gold</b>: {_e(case.get('gold', ''))}</div>"
        f"<div><b>Answer</b>: {_e(run_summary.get('answer', ''))}</div>"
        f"<div><b>Correct</b>: {_e(run_summary.get('correct', ''))}</div>"
        f"<div><b>Rounds</b>: {_e(run_summary.get('rounds', ''))}</div>"
        f"<div><b>Accepted investigations</b>: {_e(run_summary.get('accepted_investigations', ''))}</div>"
        f"<div><b>Virtual duration</b>: {_fmt_time(float(timeline.get('duration_sec') or 0.0))}</div>"
        f"<div class='wide'><b>Options</b><ul>{option_lines}</ul></div>"
        "</div>"
    )


def _overview_grid(images: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]]) -> str:
    by_segment = {str(item.get("segment_id")): item for item in segments}
    cards = []
    for item in images:
        segment_id = str(item.get("segment_id", ""))
        segment = by_segment.get(segment_id, {})
        role = str(segment.get("role", "hidden"))
        cards.append(
            "<figure>"
            f"<img src='{_e(item['rel'])}' loading='lazy'>"
            f"<figcaption>{_e(segment_id)} · {_fmt_range(segment.get('virtual_start_sec'), segment.get('virtual_end_sec'))} · source {_e(segment.get('source_video_id', ''))} · role hidden in prompt ({_e(role)})</figcaption>"
            "</figure>"
        )
    return f"<div class='grid overview'>{''.join(cards)}</div>"


def _beat_grid(beats: Sequence[Mapping[str, Any]], bundle: AssetBundler) -> str:
    cards = []
    for beat in beats:
        paths = list(beat.get("thumbnail_grid_paths") or [beat.get("thumbnail_grid_path", "")])
        imgs = []
        for index, path in enumerate(paths):
            rel = bundle.add(path, label=f"{beat.get('beat_id', 'beat')}_{index}")
            if rel:
                imgs.append(f"<img src='{_e(rel)}' loading='lazy'>")
        cards.append(
            "<figure>"
            f"<div class='thumb-strip'>{''.join(imgs)}</div>"
            f"<figcaption>{_e(beat.get('beat_id', ''))} · {_fmt_range(*(beat.get('virtual_time_range') or ('', '')))}</figcaption>"
            "</figure>"
        )
    return f"<div class='grid beats'>{''.join(cards) or '<p>No beat thumbnails found.</p>'}</div>"


def _frame_group(title: str, frames: Sequence[Mapping[str, Any]], bundle: AssetBundler) -> str:
    cards = []
    for frame in frames:
        rel = bundle.add(frame.get("path"), label=str(frame.get("frame_id") or "frame"))
        if not rel:
            continue
        cards.append(
            "<figure>"
            f"<img src='{_e(rel)}' loading='lazy'>"
            f"<figcaption>vt={_e(frame.get('virtual_time_sec', ''))} · src={_e(frame.get('source_video_id', ''))}@{_e(frame.get('source_time_sec', ''))} · {_e(frame.get('fps_level', ''))}</figcaption>"
            "</figure>"
        )
    return f"<details open><summary>{_e(title)} ({len(cards)} frames)</summary><div class='grid frames'>{''.join(cards) or '<p>No frames found.</p>'}</div></details>"


def _page_html(run_root: Path, summaries: Sequence[Mapping[str, Any]], cases_html: Sequence[str]) -> str:
    rows = "".join(
        "<tr>"
        f"<td><a href='#{_id(row['case_id'])}'>{_e(row['case_id'])}</a></td>"
        f"<td>{_e(row.get('correct'))}</td>"
        f"<td>{_e(row.get('answer'))}</td>"
        f"<td>{_e(row.get('rounds'))}</td>"
        f"<td>{_e(row.get('accepted_investigations'))}</td>"
        f"<td>{_e(row.get('trace_lines'))}</td>"
        "</tr>"
        for row in summaries
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Virtual Video Interaction Trace Viewer</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#f7f7f4;color:#202124}"
        "h1,h2,h3,h4{letter-spacing:0} .case{border-top:2px solid #333;padding-top:24px;margin-top:28px}"
        ".summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px 18px;background:#fff;border:1px solid #ddd;padding:12px;border-radius:8px}"
        ".summary .wide{grid-column:1/-1}.grid{display:grid;gap:12px}.overview{grid-template-columns:repeat(auto-fill,minmax(260px,1fr))}.beats{grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}.frames{grid-template-columns:repeat(auto-fill,minmax(180px,1fr))}"
        "figure{margin:0;background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}img{max-width:100%;height:auto;display:block;border-radius:4px}figcaption{font-size:12px;color:#555;margin-top:6px;word-break:break-word}"
        ".thumb-strip{display:grid;grid-template-columns:repeat(2,1fr);gap:4px}.task{background:#fff;border:1px solid #d8d8d8;border-radius:8px;padding:14px;margin:18px 0}"
        "pre{white-space:pre-wrap;word-break:break-word;background:#171717;color:#eee;padding:12px;border-radius:6px;max-height:520px;overflow:auto}details{margin:10px 0}summary{cursor:pointer;font-weight:600}table{border-collapse:collapse;background:#fff}td,th{border:1px solid #ddd;padding:6px 8px;text-align:left}"
        "</style></head><body>"
        "<h1>Virtual Video Multi-Round Interaction Trace Viewer</h1>"
        f"<p>Run root: <code>{_e(str(run_root))}</code></p>"
        "<table><thead><tr><th>Case</th><th>Correct</th><th>Answer</th><th>Rounds</th><th>Accepted</th><th>Trace Lines</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        + "\n".join(cases_html)
        + "</body></html>"
    )


def _segment_overviews(workspace: Path, bundle: AssetBundler) -> list[dict[str, str]]:
    rows = []
    for path in sorted((workspace / "segment_overviews").glob("*_overview.jpg")):
        segment_id = path.name.split("_overview.jpg")[0]
        rows.append({"segment_id": segment_id, "rel": bundle.add(path, label=segment_id)})
    return rows


def _beats_by_segment(beats: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for beat in beats:
        for lineage in beat.get("source_lineage", ()) or ():
            segment_id = str(lineage.get("segment_id", ""))
            if segment_id:
                result.setdefault(segment_id, []).append(beat)
    return result


def _frames_by_query(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if query_id:
            result.setdefault(query_id, []).append(row)
    for frames in result.values():
        frames.sort(key=lambda item: float(item.get("virtual_time_sec") or 0.0))
    return result


def _evidence_by_query(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        evidence_id = str(row.get("evidence_id", ""))
        match = re.match(r"ev_(.+)_001$", evidence_id)
        if match:
            result[match.group(1)] = row
    return result


def _workspace_dirs(run_root: Path, case_ids: Sequence[str] | None) -> list[Path]:
    base = run_root / "workspaces"
    wanted = set(case_ids or ())
    rows = [path for path in sorted(base.iterdir()) if path.is_dir() and (not wanted or path.name in wanted)]
    return [path for path in rows if (path / "case.json").exists()]


def _write_manifest(out_dir: Path, run_root: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    (out_dir / "viewer_manifest.json").write_text(
        json.dumps({"run_root": str(run_root), "cases": list(summaries)}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(src_dir.parent))


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
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _details(title: str, body: str) -> str:
    return f"<details><summary>{_e(title)}</summary>{body}</details>"


def _json_block(value: Any) -> str:
    return _pre(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _pre(value: Any) -> str:
    return f"<pre>{_e(value)}</pre>"


def _fmt_range(start: Any = "", end: Any = "") -> str:
    try:
        return f"{float(start):.1f}-{float(end):.1f}s"
    except (TypeError, ValueError):
        return ""


def _fmt_time(seconds: float) -> str:
    return f"{seconds:.1f}s / {seconds / 3600.0:.2f}h"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:80] or "asset"


def _id(value: Any) -> str:
    return _safe_name(str(value))


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a self-contained HTML viewer for virtual video interaction traces.")
    parser.add_argument("--run-root", required=True, help="Root containing workspaces/ and summary JSON.")
    parser.add_argument("--out-dir", required=True, help="Output directory for index.html and copied assets.")
    parser.add_argument("--zip-path", help="Optional zip output path. Defaults to <out-dir>.zip.")
    parser.add_argument("--case-ids", nargs="*", help="Optional subset of case IDs.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
