"""Train YOLOv8n and save the reusable artefacts."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from ultralytics import YOLO

from diagnostic_des_tumeurs.config.hyperparameters import YoloConfig
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import save_json
from diagnostic_des_tumeurs.yolo.helpers.runtime import disable_native_mlflow
from diagnostic_des_tumeurs.yolo.helpers.visualization import save_training_curves

RUN_NAME = "brain_tumor_yolov8n"


@dataclass
class TrainingResult:
    model: YOLO
    run_dir: Path
    weights: Path
    history: pd.DataFrame
    validation_metrics: dict[str, float]
    per_class: pd.DataFrame
    summary: dict[str, Any]
    train_args: dict[str, Any]
    device_name: str
    figure_paths: dict[str, Path] = field(default_factory=dict)


def per_class_table(metrics: Any) -> pd.DataFrame:
    rows = []
    for index, class_id in enumerate(metrics.box.ap_class_index):
        rows.append(
            {
                "class": metrics.names[int(class_id)],
                "precision": float(metrics.box.p[index]),
                "recall": float(metrics.box.r[index]),
                "mAP50": float(metrics.box.ap50[index]),
                "mAP50-95": float(metrics.box.ap[index]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mAP50-95", ascending=False, ignore_index=True
    )


def _ensure_pretrained(paths: Paths) -> Path:
    if paths.yolo_pretrained.exists():
        return paths.yolo_pretrained
    from ultralytics.utils.downloads import attempt_download_asset

    downloaded = Path(attempt_download_asset("yolov8n.pt"))
    shutil.copy2(downloaded, paths.yolo_pretrained)
    return paths.yolo_pretrained


def _resolve_device(config: YoloConfig) -> tuple[str | int, str, bool]:
    cuda_available = torch.cuda.is_available()
    if config.device is None:
        device: str | int = 0 if cuda_available else "cpu"
    else:
        device = config.device
    if isinstance(device, int):
        device_name = (
            torch.cuda.get_device_name(device) if cuda_available else f"GPU {device}"
        )
    elif str(device).isdigit():
        index = int(str(device))
        device = index
        device_name = (
            torch.cuda.get_device_name(index) if cuda_available else f"GPU {index}"
        )
    else:
        device_name = "CPU" if str(device).lower() == "cpu" else str(device)
    use_gpu = str(device).lower() != "cpu"
    return device, device_name, use_gpu


def resolve_training_args(
    paths: Paths, config: YoloConfig
) -> tuple[dict[str, Any], str]:
    """Apply device defaults and build the Ultralytics training arguments."""
    if not paths.yolo_data_yaml.exists():
        raise FileNotFoundError(
            f"{paths.yolo_data_yaml} not found. Run the preprocessing step first."
        )
    device, device_name, use_gpu = _resolve_device(config)
    epochs = config.epochs if config.epochs is not None else (30 if use_gpu else 10)
    imgsz = config.imgsz if config.imgsz is not None else (640 if use_gpu else 320)
    batch = config.batch if config.batch is not None else (32 if use_gpu else 8)
    patience = (
        config.patience if config.patience is not None else (20 if use_gpu else 5)
    )

    args: dict[str, Any] = {
        "data": str(paths.yolo_data_yaml),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": os.cpu_count() or 1,
        "patience": int(patience),
        "device": device,
        "project": str(paths.yolo_runs_dir),
        "name": RUN_NAME,
        "exist_ok": True,
        "seed": config.seed,
        "lr0": config.learning_rate,
        "weight_decay": config.weight_decay,
        "fraction": config.fraction,
        "plots": True,
    }
    return args, device_name


def train(
    paths: Paths,
    config: YoloConfig | None = None,
) -> TrainingResult:
    """Train, validate and save the best checkpoint plus the standard artefacts."""
    config = config or YoloConfig()
    paths.ensure_output_dirs()
    disable_native_mlflow()
    train_args, device_name = resolve_training_args(paths, config)
    pretrained = _ensure_pretrained(paths)

    model = YOLO(str(pretrained))
    model.train(**train_args)
    run_dir = Path(model.trainer.save_dir)

    history = pd.read_csv(run_dir / "results.csv")
    history.columns = [str(column).strip() for column in history.columns]

    validation = model.val(
        data=str(paths.yolo_data_yaml),
        split="val",
        imgsz=train_args["imgsz"],
        batch=train_args["batch"],
        device=train_args["device"],
        project=str(paths.yolo_runs_dir),
        name=f"{RUN_NAME}_val",
        exist_ok=True,
        plots=True,
    )
    validation_metrics = {
        key: float(value) for key, value in validation.results_dict.items()
    }
    table = per_class_table(validation)

    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = run_dir / "weights" / "last.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"No weights found in {run_dir / 'weights'}")
    shutil.copy2(best_weights, paths.yolo_saved_model)
    shutil.copy2(run_dir / "args.yaml", paths.yolo_training_args)
    shutil.copy2(run_dir / "results.csv", paths.yolo_training_results)

    paths.yolo_training_plots.mkdir(parents=True, exist_ok=True)
    for plot_name in (
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "labels.jpg",
    ):
        plot_path = run_dir / plot_name
        if plot_path.exists():
            shutil.copy2(plot_path, paths.yolo_training_plots / plot_name)

    curves_path = save_training_curves(
        history, paths.yolo_training_plots / "curves.png"
    )
    split_counts = {
        split: len(
            [
                path
                for path in (paths.yolo_dataset_dir / "images" / split).iterdir()
                if path.is_file()
            ]
        )
        for split in ("train", "valid", "test")
    }
    class_names = [
        validation.names[index] for index in sorted(validation.names)
    ]
    summary = {
        "created_by": "diagnostic_des_tumeurs.yolo.functions.training",
        "model": "yolov8n",
        "weights": paths.yolo_saved_model.relative_to(paths.root).as_posix(),
        "run_dir": run_dir.relative_to(paths.root).as_posix(),
        "dataset": paths.yolo_data_yaml.relative_to(paths.root).as_posix(),
        "classes": class_names,
        "device": device_name,
        "epochs_requested": train_args["epochs"],
        "epochs_completed": int(history["epoch"].max()) + 1,
        "imgsz": train_args["imgsz"],
        "batch": train_args["batch"],
        "learning_rate": train_args["lr0"],
        "weight_decay": train_args["weight_decay"],
        "fraction": train_args["fraction"],
        "seed": train_args["seed"],
        "split_counts": split_counts,
        "validation_metrics": validation_metrics,
        "per_class_validation": table.to_dict(orient="records"),
    }
    save_json(paths.yolo_training_summary, summary)

    return TrainingResult(
        model=model,
        run_dir=run_dir,
        weights=paths.yolo_saved_model,
        history=history,
        validation_metrics=validation_metrics,
        per_class=table,
        summary=summary,
        train_args=train_args,
        device_name=device_name,
        figure_paths={
            "curves": curves_path,
            "results": paths.yolo_training_plots / "results.png",
        },
    )
