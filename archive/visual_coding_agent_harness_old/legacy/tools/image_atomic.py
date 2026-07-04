"""Traditional image manipulation tools.

These tools are deterministic pixel operations. They should stay separate from
VLM-based inspection tools so the harness can audit what was actually changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageEnhance, ImageFilter


def crop_region(image_path: str, bbox: Sequence[int], output_path: str) -> Mapping[str, object]:
    with Image.open(image_path) as image:
        pixel_bbox = _normalized_bbox_to_pixels(bbox, image.size)
        cropped = image.crop(pixel_bbox)
        _ensure_parent(output_path)
        cropped.save(output_path)

    return {
        "claim": "Cropped region saved.",
        "confidence": 1.0,
        "input_artifacts": [image_path],
        "output_artifacts": [output_path],
        "regions": [{"bbox": list(bbox), "pixel_bbox": list(pixel_bbox)}],
        "limitations": "Deterministic crop; semantic correctness depends on caller-selected bbox.",
    }


def zoom_region(
    image_path: str,
    bbox: Sequence[int],
    output_path: str,
    scale: float = 2.0,
) -> Mapping[str, object]:
    with Image.open(image_path) as image:
        pixel_bbox = _normalized_bbox_to_pixels(bbox, image.size)
        cropped = image.crop(pixel_bbox)
        width, height = cropped.size
        zoomed = cropped.resize((int(width * scale), int(height * scale)))
        _ensure_parent(output_path)
        zoomed.save(output_path)

    return {
        "claim": "Region cropped and zoomed.",
        "confidence": 1.0,
        "input_artifacts": [image_path],
        "output_artifacts": [output_path],
        "regions": [{"bbox": list(bbox), "pixel_bbox": list(pixel_bbox)}],
        "limitations": "Deterministic zoom; may amplify blur or compression artifacts.",
    }


def threshold_image(image_path: str, output_path: str, threshold: int = 128) -> Mapping[str, object]:
    with Image.open(image_path) as image:
        gray = image.convert("L")
        thresholded = gray.point(lambda value: 255 if value >= threshold else 0)
        _ensure_parent(output_path)
        thresholded.save(output_path)

    return {
        "claim": f"Thresholded image at value {threshold}.",
        "confidence": 1.0,
        "input_artifacts": [image_path],
        "output_artifacts": [output_path],
        "limitations": "Deterministic threshold; useful for OCR pre-processing, not semantic interpretation.",
    }


def enhance_image(
    image_path: str,
    output_path: str,
    sharpness: float = 1.0,
    contrast: float = 1.0,
) -> Mapping[str, object]:
    with Image.open(image_path) as image:
        enhanced = ImageEnhance.Sharpness(image).enhance(sharpness)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)
        _ensure_parent(output_path)
        enhanced.save(output_path)

    return {
        "claim": f"Enhanced image with sharpness={sharpness} and contrast={contrast}.",
        "confidence": 1.0,
        "input_artifacts": [image_path],
        "output_artifacts": [output_path],
        "limitations": "Deterministic enhancement; may distort colors or edges.",
    }


def edge_detect(image_path: str, output_path: str) -> Mapping[str, object]:
    with Image.open(image_path) as image:
        edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
        _ensure_parent(output_path)
        edges.save(output_path)

    return {
        "claim": "Edge image saved.",
        "confidence": 1.0,
        "input_artifacts": [image_path],
        "output_artifacts": [output_path],
        "limitations": "Deterministic edge detection; not an object detector.",
    }


def _normalized_bbox_to_pixels(bbox: Sequence[int], image_size: Sequence[int]) -> Sequence[int]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain [x1, y1, x2, y2]")
    width, height = image_size
    x1, y1, x2, y2 = bbox
    pixels = [
        round(width * x1 / 1000),
        round(height * y1 / 1000),
        round(width * x2 / 1000),
        round(height * y2 / 1000),
    ]
    if pixels[0] >= pixels[2] or pixels[1] >= pixels[3]:
        raise ValueError("bbox must have x1 < x2 and y1 < y2")
    return [
        max(0, min(width, pixels[0])),
        max(0, min(height, pixels[1])),
        max(0, min(width, pixels[2])),
        max(0, min(height, pixels[3])),
    ]


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
