"""Evaluate the trained YOLO model on the held-out test split."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from diagnostic_des_tumeurs.config.hyperparameters import (
    YOLO_CLASS_NAMES,
    YOLO_IMAGE_EXTENSIONS,
    YoloConfig,
)
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import load_json, save_json
from diagnostic_des_tumeurs.yolo.functions.training import per_class_table
from diagnostic_des_tumeurs.yolo.helpers.annotations import box_iou, read_label_boxes
from diagnostic_des_tumeurs.yolo.helpers.runtime import disable_native_mlflow
from diagnostic_des_tumeurs.yolo.helpers.visualization import (
    save_prediction_samples,
    save_validation_vs_test,
)

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
SEED = 42


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    per_class: pd.DataFrame
    error_table: pd.DataFrame
    comparison: pd.DataFrame
    summary: dict[str, Any]
    run_dir: Path
    verdict: str
    figure_paths: dict[str, Path] = field(default_factory=dict)


def _find_weights(paths: Paths) -> Path:
    if paths.yolo_saved_model.exists():
        return paths.yolo_saved_model
    best = paths.yolo_runs_dir / "brain_tumor_yolov8n" / "weights" / "best.pt"
    if best.exists():
        return best
    raise FileNotFoundError(
        "No trained model found. Run the training step first, or place a "
        f"checkpoint at {paths.yolo_saved_model}."
    )


def _sample_predictions(
    model: YOLO,
    paths: Paths,
    class_names: list[str],
    imgsz: int,
) -> tuple[list[Path], list[Path], list[Any]]:
    label_dir = paths.yolo_dataset_dir / "labels" / "test"
    image_dir = paths.yolo_dataset_dir / "images" / "test"
    labels = sorted(label_dir.glob("*.txt"))
    chosen: list[Path] = []
    for class_id in range(len(class_names)):
        for label_path in labels:
            if any(
                box[0] == class_id for box in read_label_boxes(label_path)
            ):
                chosen.append(label_path)
                break
    image_paths = [
        next(image_dir.glob(f"{label.stem}.*")) for label in chosen
    ]
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=imgsz,
        verbose=False,
    )
    return image_paths, chosen, predictions


def _error_analysis(
    model: YOLO,
    paths: Paths,
    class_names: list[str],
    imgsz: int,
) -> pd.DataFrame:
    image_dir = paths.yolo_dataset_dir / "images" / "test"
    label_dir = paths.yolo_dataset_dir / "labels" / "test"
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in YOLO_IMAGE_EXTENSIONS
    )
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=imgsz,
        verbose=False,
    )
    counts = {
        class_id: {"tp": 0, "fp": 0, "fn": 0}
        for class_id in range(len(class_names))
    }

    for image_path, result in zip(image_paths, predictions):
        height, width = result.orig_shape
        label_path = label_dir / f"{image_path.stem}.txt"
        ground_truth = []
        for class_id, x, y, box_width, box_height in read_label_boxes(label_path):
            ground_truth.append(
                (
                    class_id,
                    np.array(
                        [
                            (x - box_width / 2) * width,
                            (y - box_height / 2) * height,
                            (x + box_width / 2) * width,
                            (y + box_height / 2) * height,
                        ]
                    ),
                )
            )

        predicted = []
        if result.boxes is not None:
            for xyxy, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                predicted.append((int(class_id), xyxy))

        matched: set[int] = set()
        for class_id, predicted_box in predicted:
            best_index, best_iou = None, 0.0
            for index, (true_class, true_box) in enumerate(ground_truth):
                if index in matched or true_class != class_id:
                    continue
                overlap = box_iou(predicted_box, true_box)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index is not None and best_iou >= IOU_THRESHOLD:
                matched.add(best_index)
                counts[class_id]["tp"] += 1
            else:
                counts[class_id]["fp"] += 1

        for index, (true_class, _) in enumerate(ground_truth):
            if index not in matched:
                counts[true_class]["fn"] += 1

    rows = []
    for class_id, values in counts.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "class": class_names[class_id],
                "true_boxes": tp + fn,
                "predicted_boxes": tp + fp,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision@0.5": precision,
                "recall@0.5": recall,
            }
        )
    return pd.DataFrame(rows)


def evaluate(paths: Paths, config: YoloConfig | None = None) -> EvaluationResult:
    """Run test metrics, per-class tables, error analysis and val/test comparison."""
    config = config or YoloConfig()
    paths.ensure_output_dirs()
    disable_native_mlflow()

    weights = _find_weights(paths)
    training_summary = (
        load_json(paths.yolo_training_summary)
        if paths.yolo_training_summary.exists()
        else {}
    )
    imgsz = int(
        config.imgsz
        or training_summary.get("imgsz")
        or (320 if not torch.cuda.is_available() else 640)
    )
    batch = int(
        config.batch
        or training_summary.get("batch")
        or (8 if not torch.cuda.is_available() else 32)
    )
    device: str | int = (
        config.device
        if config.device is not None
        else (0 if torch.cuda.is_available() else "cpu")
    )

    model = YOLO(str(weights))
    test_metrics = model.val(
        data=str(paths.yolo_data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(paths.yolo_runs_dir),
        name="brain_tumor_yolov8n_test",
        exist_ok=True,
        plots=True,
    )
    metrics = {
        key: float(value) for key, value in test_metrics.results_dict.items()
    }
    table = per_class_table(test_metrics)
    error_table = _error_analysis(model, paths, YOLO_CLASS_NAMES, imgsz)

    validation_metrics = training_summary.get("validation_metrics", {})
    comparison_rows = []
    for key, test_value in metrics.items():
        validation_value = validation_metrics.get(key)
        comparison_rows.append(
            {
                "metric": key,
                "validation": validation_value,
                "test": float(test_value),
                "gap (test - val)": (
                    float(test_value) - float(validation_value)
                    if validation_value is not None
                    else None
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    key_metric = "metrics/mAP50(B)"
    gap = (
        metrics[key_metric] - float(validation_metrics[key_metric])
        if key_metric in validation_metrics
        else None
    )
    if gap is None:
        verdict = "validation metrics not available"
    elif gap >= -0.05:
        verdict = "good generalization — test and validation scores are close"
    elif gap >= -0.15:
        verdict = "moderate gap — check the weakest classes"
    else:
        verdict = "large gap — the model likely over-fits; add data or regularize"

    figure_paths: dict[str, Path] = {}
    image_paths, label_paths, predictions = _sample_predictions(
        model, paths, YOLO_CLASS_NAMES, imgsz
    )
    if image_paths:
        figure_paths["predictions"] = save_prediction_samples(
            image_paths,
            label_paths,
            predictions,
            YOLO_CLASS_NAMES,
            paths.yolo_evaluation_dir / "predictions.png",
        )

    if training_summary.get("per_class_validation"):
        validation_table = pd.DataFrame(
            training_summary["per_class_validation"]
        )
        figure_paths["validation_vs_test"] = save_validation_vs_test(
            validation_table,
            table,
            paths.yolo_evaluation_dir / "validation_vs_test.png",
        )

    test_run_dir = Path(test_metrics.save_dir)
    if (test_run_dir / "results.csv").exists():
        shutil.copy2(
            test_run_dir / "results.csv",
            paths.yolo_evaluation_dir / "test_results.csv",
        )
    plots_dir = paths.yolo_evaluation_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for plot_name in (
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "val_batch0_pred.jpg",
        "val_batch0_labels.jpg",
    ):
        plot_path = test_run_dir / plot_name
        if plot_path.exists():
            shutil.copy2(plot_path, plots_dir / plot_name)
    figure_paths["plots"] = plots_dir

    summary = {
        "created_by": "diagnostic_des_tumeurs.yolo.functions.evaluation",
        "weights": weights.relative_to(paths.root).as_posix(),
        "test_run_dir": test_run_dir.relative_to(paths.root).as_posix(),
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "test_metrics": metrics,
        "validation_metrics": validation_metrics,
        "per_class_test": table.to_dict(orient="records"),
        "error_analysis": error_table.to_dict(orient="records"),
        "validation_vs_test": comparison.to_dict(orient="records"),
        "verdict": verdict,
        "mAP50_gap": gap,
        "figures": {
            key: value.as_posix() for key, value in figure_paths.items()
        },
    }
    save_json(paths.yolo_evaluation_dir / "test_evaluation.json", summary)

    return EvaluationResult(
        metrics=metrics,
        per_class=table,
        error_table=error_table,
        comparison=comparison,
        summary=summary,
        run_dir=test_run_dir,
        verdict=verdict,
        figure_paths=figure_paths,
    )
