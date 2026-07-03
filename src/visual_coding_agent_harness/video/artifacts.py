"""Image artifact helpers for the multi_v3 video index."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from .index import Frame, Scene, Shot


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def is_image_path(path: str | Path, *, must_exist: bool = False) -> bool:
    candidate = Path(str(path))
    if candidate.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    return candidate.exists() if must_exist else True


def compose_shot_grid(
    frames: Sequence[Frame],
    out_path: Path,
    *,
    cols: int = 3,
    cell_size: tuple[int, int] = (320, 180),
) -> Path:
    """Compose a low-resolution JPG grid from existing frame thumbnails."""

    sources = [Path(frame.thumb_path) for frame in frames if frame.thumb_path and is_image_path(frame.thumb_path, must_exist=True)]
    if not sources:
        raise ValueError("compose_shot_grid requires at least one readable image frame")
    cells = [_fit_image(source, cell_size=cell_size) for source in sources]
    return _write_grid(cells, out_path=out_path, cols=cols, cell_size=cell_size)


def compose_scene_thumb(
    scene: Scene,
    shots: Sequence[Shot],
    out_path: Path,
    *,
    cell_size: tuple[int, int] = (320, 180),
) -> Path:
    """Create a scene thumbnail from the first shot grid or a placeholder."""

    source = _first_existing_image(
        [shot.lowres_grid_path for shot in shots]
        + [frame.thumb_path for shot in shots for frame in shot.frames]
    )
    if source is None:
        image = _placeholder(cell_size, scene.scene_id, scene.title or scene.summary)
    else:
        image = _fit_image(source, cell_size=cell_size)
    return _save_jpeg(image, out_path)


def compose_scene_timeline_grid(
    scenes: Sequence[Scene],
    out_path: Path,
    *,
    cols: int = 8,
    cell_size: tuple[int, int] = (320, 180),
) -> Path:
    cells = []
    for scene in scenes:
        source = _first_existing_image((scene.scene_thumb_path,))
        if source is None:
            cells.append(_placeholder(cell_size, scene.scene_id, scene.title or scene.summary))
        else:
            cells.append(_fit_image(source, cell_size=cell_size))
    if not cells:
        cells = [_placeholder(cell_size, "empty", "No scenes indexed")]
    return _write_grid(cells, out_path=out_path, cols=cols, cell_size=cell_size)


def _first_existing_image(paths: Sequence[str]) -> Path | None:
    for raw_path in paths:
        if is_image_path(raw_path, must_exist=True):
            return Path(raw_path)
    return None


def _fit_image(path: Path, *, cell_size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(cell_size)
        canvas = Image.new("RGB", cell_size, color=(18, 18, 18))
        x = (cell_size[0] - image.width) // 2
        y = (cell_size[1] - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas


def _placeholder(cell_size: tuple[int, int], label: str, detail: str = "") -> Image.Image:
    image = Image.new("RGB", cell_size, color=(42, 48, 56))
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), str(label)[:32], fill=(240, 240, 240))
    if detail:
        draw.text((12, 34), str(detail)[:72], fill=(200, 210, 220))
    return image


def _write_grid(
    cells: Sequence[Image.Image],
    *,
    out_path: Path,
    cols: int,
    cell_size: tuple[int, int],
) -> Path:
    cols = max(1, min(int(cols), len(cells)))
    rows = max(1, ceil(len(cells) / cols))
    canvas = Image.new("RGB", (cols * cell_size[0], rows * cell_size[1]), color=(12, 12, 12))
    for index, cell in enumerate(cells):
        x = (index % cols) * cell_size[0]
        y = (index // cols) * cell_size[1]
        canvas.paste(cell.convert("RGB"), (x, y))
    return _save_jpeg(canvas, out_path)


def _save_jpeg(image: Image.Image, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path, format="JPEG", quality=88)
    return out_path
