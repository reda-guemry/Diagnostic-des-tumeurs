"""YOLO annotation parsing and geometric helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

Box = tuple[int, float, float, float, float]


def parse_yolo_label(path: str | Path, num_classes: int) -> tuple[list[Box], list[str]]:
    """Return ``(boxes, errors)`` for one YOLO ``.txt`` file.

    Boxes are ``(class_id, x_center, y_center, width, height)`` with normalized
    coordinates.  An empty file is a valid background image.
    """
    path = Path(path)
    boxes: list[Box] = []
    errors: list[str] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes, errors

    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 values, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            x, y, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"line {line_number}: non-numeric value")
            continue

        if not 0 <= class_id < num_classes:
            errors.append(
                f"line {line_number}: class id {class_id} out of range"
            )
            continue
        if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            errors.append(f"line {line_number}: coordinates not in [0, 1]")
            continue
        if width <= 0 or height <= 0:
            errors.append(f"line {line_number}: width/height must be positive")
            continue
        if (
            x - width / 2 < -1e-6
            or x + width / 2 > 1 + 1e-6
            or y - height / 2 < -1e-6
            or y + height / 2 > 1 + 1e-6
        ):
            errors.append(f"line {line_number}: box crosses the image border")
            continue

        boxes.append((class_id, x, y, width, height))
    return boxes, errors


def read_label_boxes(path: str | Path) -> list[Box]:
    """Read a label file without validation, for plotting and error analysis."""
    path = Path(path)
    if not path.exists():
        return []
    boxes: list[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, width, height = line.split()
        boxes.append(
            (int(class_id), float(x), float(y), float(width), float(height))
        )
    return boxes


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def label_classes(boxes: list[Box]) -> list[int]:
    return sorted({int(box[0]) for box in boxes})


def relative(path: str | Path, root: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def results_to_records(result: Any) -> list[dict[str, Any]]:
    """Convert one Ultralytics result to JSON-friendly detections."""
    boxes = result.boxes
    if boxes is None:
        return []
    records: list[dict[str, Any]] = []
    for xyxy, confidence, class_id in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.cls.cpu().numpy().astype(int),
    ):
        records.append(
            {
                "class_id": int(class_id),
                "class_name": result.names[int(class_id)],
                "confidence": float(confidence),
                "xyxy": [float(value) for value in xyxy],
            }
        )
    return records
