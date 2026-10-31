"""Plotting helpers for the YOLO workflow."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnostic_des_tumeurs.yolo.helpers.annotations import read_label_boxes

CLASS_COLORS = {
    0: "#e74c3c",
    1: "#3498db",
    2: "#2ecc71",
    3: "#9b59b6",
}
_FALLBACK_COLORS = ["#f39c12", "#1abc9c", "#34495e", "#e84393", "#6c5ce7"]


def class_color(class_id: int) -> str:
    return CLASS_COLORS.get(
        int(class_id), _FALLBACK_COLORS[int(class_id) % len(_FALLBACK_COLORS)]
    )


def _read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_boxes(
    axis: plt.Axes,
    image_path: str | Path,
    boxes: Iterable[tuple[int, float, float, float, float]],
    class_names: list[str],
) -> None:
    """Draw normalized YOLO boxes on ``axis``."""
    image = _read_rgb(image_path)
    height, width = image.shape[:2]
    axis.imshow(image)
    axis.axis("off")
    for class_id, x, y, box_width, box_height in boxes:
        x1 = (x - box_width / 2) * width
        y1 = (y - box_height / 2) * height
        color = class_color(class_id)
        axis.add_patch(
            patches.Rectangle(
                (x1, y1),
                box_width * width,
                box_height * height,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
        )
        axis.text(
            x1,
            max(y1 - 4, 8),
            class_names[class_id],
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "pad": 2, "edgecolor": "none"},
        )


def draw_detections(
    axis: plt.Axes,
    image_path: str | Path,
    detections: Iterable[dict[str, Any]],
) -> None:
    """Draw absolute ``xyxy`` detections returned by Ultralytics."""
    image = _read_rgb(image_path)
    axis.imshow(image)
    axis.axis("off")
    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"]
        class_id = int(detection["class_id"])
        color = class_color(class_id)
        axis.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
        )
        axis.text(
            x1,
            max(y1 - 4, 8),
            f"{detection['class_name']} {detection['confidence']:.2f}",
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "pad": 2, "edgecolor": "none"},
        )


def save_class_distribution(
    image_pivot: pd.DataFrame,
    instance_pivot: pd.DataFrame,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    image_pivot.plot(kind="bar", ax=axes[0], color=["#4c72b0", "#dd8452"])
    axes[0].set_title("Paired images per folder")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Images")
    axes[0].tick_params(axis="x", rotation=20)

    instance_pivot.plot(kind="bar", ax=axes[1], color=["#4c72b0", "#dd8452"])
    axes[1].set_title("Bounding boxes per class")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Boxes")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_resolution_plot(size_counts: pd.Series, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top = size_counts.head(10)
    labels = [f"{int(width)}x{int(height)}" for width, height in top.index]
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar(labels, top.values, color="#55a868")
    axis.set_title("Most common image resolutions")
    axis.set_xlabel("Resolution")
    axis.set_ylabel("Images")
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_sample_grid(
    records: list[dict[str, Any]],
    root: str | Path,
    class_names: list[str],
    path: str | Path,
    seed: int = 42,
) -> Path:
    """Save two annotated archive samples for every class."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fig, axes = plt.subplots(len(class_names), 2, figsize=(9, 4 * len(class_names)))
    axes = np.atleast_2d(axes)

    for class_id, class_name in enumerate(class_names):
        candidates = [
            record
            for record in records
            if record["valid"]
            and record["boxes"]
            and record["folder_class"] == class_id
        ]
        chosen = rng.sample(candidates, min(2, len(candidates)))
        for column in range(2):
            axis = axes[class_id, column]
            if column < len(chosen):
                record = chosen[column]
                draw_boxes(
                    axis,
                    Path(root) / record["image"],
                    record["boxes"],
                    class_names,
                )
                axis.set_title(
                    f"{class_name} — {Path(record['image']).name}", fontsize=9
                )
            else:
                axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_dataset_samples(
    dataset_dir: str | Path,
    splits: Iterable[str],
    class_names: list[str],
    path: str | Path,
) -> Path:
    """Save one annotated image per split and class from the prepared dataset."""
    dataset_dir = Path(dataset_dir)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = list(splits)

    fig, axes = plt.subplots(
        len(splits), len(class_names), figsize=(14, 10)
    )
    axes = np.atleast_2d(axes)
    for row, split in enumerate(splits):
        labels = sorted((dataset_dir / "labels" / split).glob("*.txt"))
        for column, class_id in enumerate(range(len(class_names))):
            axis = axes[row, column]
            label_path = next(
                (
                    candidate
                    for candidate in labels
                    if any(
                        box[0] == class_id
                        for box in read_label_boxes(candidate)
                    )
                ),
                None,
            )
            if label_path is None:
                axis.axis("off")
                axis.set_title(
                    f"{split} — {class_names[class_id]} (none)", fontsize=9
                )
                continue
            image_path = next(
                (dataset_dir / "images" / split).glob(f"{label_path.stem}.*")
            )
            draw_boxes(
                axis,
                image_path,
                read_label_boxes(label_path),
                class_names,
            )
            axis.set_title(
                f"{split} — {class_names[class_id]}", fontsize=9
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_training_curves(history: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for column in (
        "train/box_loss",
        "val/box_loss",
        "train/cls_loss",
        "val/cls_loss",
        "train/dfl_loss",
        "val/dfl_loss",
    ):
        if column in history:
            axes[0].plot(history["epoch"], history[column], label=column)
    axes[0].set_title("Detection losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    for column in (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ):
        if column in history:
            axes[1].plot(history["epoch"], history[column], label=column)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_validation_vs_test(
    validation_table: pd.DataFrame,
    test_table: pd.DataFrame,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validation = validation_table.set_index("class")
    test = test_table.set_index("class")
    classes = [name for name in validation.index if name in test.index]
    positions = np.arange(len(classes))
    width = 0.38

    fig, axis = plt.subplots(figsize=(9, 4))
    axis.bar(
        positions - width / 2,
        validation.loc[classes, "mAP50"],
        width,
        label="validation",
    )
    axis.bar(
        positions + width / 2,
        test.loc[classes, "mAP50"],
        width,
        label="test",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(classes)
    axis.set_ylabel("mAP@0.5")
    axis.set_title("Per-class mAP@0.5 — validation vs test")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_prediction_samples(
    image_paths: list[Path],
    label_paths: list[Path],
    predictions: list[Any],
    class_names: list[str],
    path: str | Path,
) -> Path:
    """Ground truth on the left, predictions on the right."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = max(1, len(image_paths))
    fig, axes = plt.subplots(count, 2, figsize=(9, 4 * count))
    axes = np.atleast_2d(axes)

    for row, (image_path, label_path, result) in enumerate(
        zip(image_paths, label_paths, predictions)
    ):
        draw_boxes(
            axes[row, 0],
            image_path,
            read_label_boxes(label_path),
            class_names,
        )
        axes[row, 0].set_title(f"Ground truth — {image_path.name}", fontsize=9)

        detections = []
        if result.boxes is not None:
            for xyxy, confidence, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                detections.append(
                    {
                        "class_id": int(class_id),
                        "class_name": result.names[int(class_id)],
                        "confidence": float(confidence),
                        "xyxy": [float(value) for value in xyxy],
                    }
                )
        draw_detections(axes[row, 1], image_path, detections)
        axes[row, 1].set_title(
            f"Prediction — {len(detections)} box(es)", fontsize=9
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
