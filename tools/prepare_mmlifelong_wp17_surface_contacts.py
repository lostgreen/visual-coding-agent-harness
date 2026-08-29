#!/usr/bin/env python3
"""Prepare compact, zero-API contact sheets for WP17-0 pixel review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps

from vcah.wp17_preflight import select_surface_review_rows


def run(args: argparse.Namespace) -> Path:
    a3_root = Path(args.a3_root)
    manifest = _read_json(a3_root / "run_manifest.json")
    selection_path = Path(str(manifest["selection_path"]))
    selection_rows = _read_jsonl(selection_path)
    batch_results = tuple(
        _read_json(path)
        for path in sorted((a3_root / "batch_results").rglob("*.json"))
    )
    parsed_rows = tuple(
        dict(row)
        for result in batch_results
        for row in tuple(result.get("parsed_rows", ()) or ())
        if isinstance(row, Mapping)
    )
    case_ids = tuple(dict.fromkeys(args.case_ids))
    if not case_ids:
        raise ValueError("at least one case ID is required")

    out_root = Path(args.out_root)
    manifest_path = out_root / "surface_contact_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"contact manifest already exists: {manifest_path}")
    out_root.mkdir(parents=True, exist_ok=True)

    case_outputs = []
    with tempfile.TemporaryDirectory(prefix="wp17-frames-", dir=out_root) as tmp:
        frame_root = Path(tmp)
        for case_id in case_ids:
            rows = select_surface_review_rows(
                case_id=case_id,
                selection_rows=selection_rows,
                parsed_rows=parsed_rows,
                max_frames=int(args.max_frames),
            )
            images = []
            public_rows = []
            for index, row in enumerate(rows):
                frame_path = frame_root / f"{case_id}-{index:02d}.jpg"
                _extract_frame(
                    ffmpeg=args.ffmpeg,
                    source_path=Path(str(row["source_path"])),
                    source_time_sec=float(row["source_time_sec"]),
                    out_path=frame_path,
                )
                images.append(Image.open(frame_path).convert("RGB"))
                public_rows.append(
                    {
                        "frame_label": row["frame_label"],
                        "source_video_id": str(row["source_video_id"]),
                        "source_time_sec": round(float(row["source_time_sec"]), 3),
                        "virtual_time_sec": round(float(row["virtual_time_sec"]), 3),
                        "surface_review_score": int(row["surface_review_score"]),
                    }
                )
            outputs = {}
            for view in ("full", "top", "bottom"):
                target = out_root / f"{case_id}-{view}.jpg"
                _write_contact_sheet(images, public_rows, view=view, target=target)
                outputs[view] = target.name
            for image in images:
                image.close()
            case_outputs.append(
                {
                    "case_id": case_id,
                    "selected_frame_count": len(rows),
                    "frames": public_rows,
                    "outputs": outputs,
                }
            )

    payload = {
        "schema_version": "MMLifelongWP17SurfaceContactsV1",
        "diagnostic_only": True,
        "model_calls": 0,
        "question_visible": False,
        "options_visible": False,
        "answer_visible": False,
        "source_paths_persisted": False,
        "case_count": len(case_outputs),
        "cases": case_outputs,
    }
    _write_json(manifest_path, payload)
    print(
        "WP17_CONTACTS_DONE "
        f"cases={len(case_outputs)} frames="
        f"{sum(row['selected_frame_count'] for row in case_outputs)}",
        flush=True,
    )
    return manifest_path


def _extract_frame(
    *, ffmpeg: str, source_path: Path, source_time_sec: float, out_path: Path
) -> None:
    result = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{source_time_sec:.3f}",
            "-i",
            str(source_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=1280:-2:force_original_aspect_ratio=decrease",
            "-q:v",
            "2",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if result.returncode != 0 or not out_path.is_file():
        fingerprint = (result.stderr or "ffmpeg did not produce a frame").strip()
        raise RuntimeError(f"frame extraction failed: {fingerprint[:240]}")


def _write_contact_sheet(
    images: Sequence[Image.Image],
    rows: Sequence[Mapping[str, Any]],
    *,
    view: str,
    target: Path,
) -> None:
    if view not in {"full", "top", "bottom"}:
        raise ValueError(f"unsupported contact view: {view}")
    tile_width = 640
    tile_height = 360 if view == "full" else 230
    label_height = 34
    columns = min(4, len(images))
    rows_count = math.ceil(len(images) / columns)
    sheet = Image.new(
        "RGB", (columns * tile_width, rows_count * (tile_height + label_height)), "black"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (image, metadata) in enumerate(zip(images, rows)):
        crop = _crop_view(image, view)
        fitted = ImageOps.contain(crop, (tile_width, tile_height))
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        sheet.paste(
            fitted,
            (x + (tile_width - fitted.width) // 2, y + (tile_height - fitted.height) // 2),
        )
        draw.text(
            (x + 8, y + tile_height + 8),
            "src %.3fs | vt %.3fs"
            % (metadata["source_time_sec"], metadata["virtual_time_sec"]),
            fill="white",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="JPEG", quality=92, optimize=True)


def _crop_view(image: Image.Image, view: str) -> Image.Image:
    if view == "full":
        return image.copy()
    width, height = image.size
    if view == "top":
        return image.crop((0, 0, width, max(1, int(height * 0.45))))
    return image.crop((0, min(height - 1, int(height * 0.55)), width, height))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"expected JSON object row: {path}")
            rows.append(dict(payload))
    return tuple(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a3-root", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
