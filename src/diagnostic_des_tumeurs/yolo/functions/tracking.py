"""MLflow logging for the YOLOv8 workflow."""

from __future__ import annotations

from dataclasses import asdict

import mlflow
import pandas as pd

from diagnostic_des_tumeurs.config.hyperparameters import (
    YOLO_REGISTERED_MODEL,
    YoloConfig,
)
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import save_json
from diagnostic_des_tumeurs.helpers.mlflow import (
    compare_runs,
    configure_experiment,
    log_artifacts,
    log_metrics,
    log_params,
    metric_key,
    register_best_run,
)
from diagnostic_des_tumeurs.yolo.functions.evaluation import EvaluationResult
from diagnostic_des_tumeurs.yolo.functions.training import TrainingResult
from diagnostic_des_tumeurs.yolo.helpers.mlflow_model import log_yolo_model

TRAIN_STAGE_FILTER = "params.stage = 'train'"
EVALUATION_METRIC = "val_mAP50_95"

_METRIC_ALIASES = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
    "fitness": "fitness",
}


def _clean(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}{_METRIC_ALIASES.get(key, metric_key(key))}": value
        for key, value in metrics.items()
    }


def _log_history(history: pd.DataFrame) -> None:
    if "epoch" not in history.columns:
        return
    for _, row in history.iterrows():
        step = int(row["epoch"]) + 1
        values = {
            column: row[column]
            for column in history.columns
            if column != "epoch"
        }
        log_metrics(values, step=step)


def log_training(
    result: TrainingResult,
    config: YoloConfig,
    paths: Paths,
    experiment: str,
) -> str:
    """Log hyperparameters, curves, validation metrics and the model."""
    configure_experiment(experiment, paths)

    with mlflow.start_run(run_name="yolo-train") as run:
        mlflow.set_tags(
            {
                "pipeline": "yolo",
                "stage": "train",
                "model": "yolov8n",
                "dataset": "data/yolo",
            }
        )
        params = asdict(config)
        split_counts = result.summary.get("split_counts", {})
        params.update(
            {
                "stage": "train",
                "epochs": result.train_args["epochs"],
                "imgsz": result.train_args["imgsz"],
                "batch": result.train_args["batch"],
                "patience": result.train_args["patience"],
                "learning_rate": result.train_args["lr0"],
                "workers": result.train_args["workers"],
                "device": result.train_args["device"],
                "fraction": result.train_args["fraction"],
                "classes": ",".join(result.summary["classes"]),
                "train_images": split_counts.get("train"),
                "valid_images": split_counts.get("valid"),
                "test_images": split_counts.get("test"),
                "epochs_completed": result.summary["epochs_completed"],
                "device_name": result.device_name,
            }
        )
        log_params(params)

        log_metrics(_clean(result.validation_metrics, "val_"))
        _log_history(result.history)
        log_artifacts(
            [
                paths.yolo_training_summary,
                paths.yolo_training_args,
                paths.yolo_training_results,
                paths.yolo_training_plots,
                paths.yolo_data_yaml,
            ]
        )
        log_yolo_model(
            result.weights,
            imgsz=result.train_args["imgsz"],
            conf=0.25,
            iou=0.5,
        )
        return run.info.run_id


def log_evaluation(
    result: EvaluationResult,
    paths: Paths,
    experiment: str,
) -> str:
    """Log the held-out test metrics and evaluation artefacts."""
    configure_experiment(experiment, paths)

    with mlflow.start_run(run_name="yolo-test") as run:
        mlflow.set_tags(
            {
                "pipeline": "yolo",
                "stage": "evaluation",
                "model": "yolov8n",
                "dataset": "data/yolo",
                "split": "test",
            }
        )
        log_params(
            {
                "stage": "evaluation",
                "confidence_threshold": result.summary["confidence_threshold"],
                "iou_threshold": result.summary["iou_threshold"],
                "weights": result.summary["weights"],
                "mAP50_gap": result.summary.get("mAP50_gap"),
                "verdict": result.verdict,
            }
        )
        log_metrics(_clean(result.metrics, "test_"))
        for row in result.per_class.to_dict(orient="records"):
            prefix = f"test_{metric_key(str(row['class']))}_"
            log_metrics(
                {
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "mAP50": row["mAP50"],
                    "mAP50_95": row["mAP50-95"],
                },
                prefix=prefix,
            )
        artifacts = [
            paths.yolo_evaluation_dir / "test_evaluation.json",
            paths.yolo_evaluation_dir / "test_results.csv",
            paths.yolo_evaluation_dir / "plots",
        ]
        for artifact in (
            result.figure_paths.get("predictions"),
            result.figure_paths.get("validation_vs_test"),
        ):
            if artifact is not None:
                artifacts.append(artifact)
        log_artifacts(artifacts)
        return run.info.run_id


def save_comparison(experiment: str, paths: Paths) -> pd.DataFrame:
    comparison = compare_runs(
        experiment,
        EVALUATION_METRIC,
        extra_filter=TRAIN_STAGE_FILTER,
    )
    if not comparison.empty:
        comparison.to_csv(paths.yolo_comparison_csv, index=False)
        save_json(
            paths.yolo_comparison_json,
            comparison.to_dict(orient="records"),
        )
    return comparison


def register_best(experiment: str, paths: Paths) -> dict[str, object]:
    info = register_best_run(
        experiment=experiment,
        metric=EVALUATION_METRIC,
        model_name=YOLO_REGISTERED_MODEL,
        description=(
            "Best YOLOv8n training run selected by validation mAP50-95. "
            "Load with mlflow.pyfunc.load_model('models:/brain-tumor-yolov8@best')."
        ),
        extra_filter=TRAIN_STAGE_FILTER,
    )
    save_json(paths.yolo_registered_model, info)
    return info
