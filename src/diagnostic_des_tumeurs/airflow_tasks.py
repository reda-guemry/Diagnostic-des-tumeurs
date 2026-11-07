"""Task callables used by the Airflow DAG.

The DAG stays thin: these functions call the existing pipeline functions
without changing their training or evaluation logic. Heavy imports (TensorFlow,
Ultralytics, MLflow) are kept inside the callables so the Airflow scheduler can
parse the DAG quickly.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from diagnostic_des_tumeurs.config.hyperparameters import (
    INCEPTION_EXPERIMENT,
    YOLO_EXPERIMENT,
    YOLO_TARGET_SPLITS,
    InceptionConfig,
    YoloConfig,
)
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import json_safe, load_json, save_json

LOGGER = logging.getLogger(__name__)

ARTIFACTS_MANIFEST = "airflow_artifacts.json"


class DatasetValidationError(RuntimeError):
    """Raised when the prepared datasets are missing or inconsistent."""


def project_root() -> Path:
    """Return the project root, optionally overridden by ``BTD_PROJECT_ROOT``."""
    configured = os.environ.get("BTD_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not _is_project_root(root):
            raise RuntimeError(
                f"BTD_PROJECT_ROOT={configured!r} does not contain "
                "pyproject.toml and src/diagnostic_des_tumeurs."
            )
        return root

    candidates = [
        Path(__file__).resolve().parents[2],
        Path.cwd(),
    ]
    for candidate in candidates:
        root = candidate.resolve()
        if _is_project_root(root):
            return root

    raise RuntimeError(
        "Cannot locate the project root. Run Airflow from the project directory "
        "or set BTD_PROJECT_ROOT."
    )


def _is_project_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").is_file()
        and (path / "src" / "diagnostic_des_tumeurs").is_dir()
    )


def _paths() -> Paths:
    paths = Paths.from_root(project_root())
    paths.ensure_output_dirs()
    return paths


def _load_classification_data(paths: Paths):
    """Rebuild the lightweight ``ClassificationData`` object from disk."""
    import pandas as pd

    from diagnostic_des_tumeurs.inception.functions.data import ClassificationData

    splits = load_json(paths.splits_path)
    return ClassificationData(
        classes=list(splits["classes"]),
        class_to_id={
            name: int(index) for name, index in splits["class_to_id"].items()
        },
        id_to_class={
            int(index): name for index, name in splits["id_to_class"].items()
        },
        metadata=pd.read_csv(paths.image_metadata_path),
        splits=splits,
    )


def prepare_classification_data() -> dict[str, Any]:
    """Validate/copy the classification split and build the balanced manifest."""
    from diagnostic_des_tumeurs.inception.functions.data import build_splits
    from diagnostic_des_tumeurs.inception.functions.preprocessing import (
        build_balanced_manifest,
    )

    paths = _paths()
    config = InceptionConfig()
    LOGGER.info("Preparing classification data from %s", paths.data_dir)

    data = build_splits(
        paths,
        seed=config.seed,
        image_size=(config.image_size, config.image_size),
    )
    balanced = build_balanced_manifest(paths, data.splits, seed=config.seed)
    summary = {
        "classes": data.classes,
        "splits": {
            split: len(data.splits[split]["paths"])
            for split in ("train", "val", "test")
        },
        "balanced_train": len(balanced["paths"]),
        "splits_path": paths.splits_path.as_posix(),
        "balanced_path": paths.balanced_path.as_posix(),
    }
    LOGGER.info("Classification data ready: %s", summary["splits"])
    return json_safe(summary)


def prepare_detection_data() -> dict[str, Any]:
    """Validate the YOLO archive and rebuild ``data/yolo``."""
    from diagnostic_des_tumeurs.yolo.functions.dataset import prepare_dataset
    from diagnostic_des_tumeurs.yolo.functions.inspection import build_manifest

    paths = _paths()
    config = YoloConfig()
    LOGGER.info("Preparing YOLO data from %s", paths.archive_dir)

    inspection = build_manifest(paths, seed=config.seed)
    dataset = prepare_dataset(paths, seed=config.seed)
    summary = {
        "paired_per_split": inspection.totals,
        "invalid_annotations": len(inspection.invalid_df),
        "mixed_class_images": int(
            (~inspection.consistency_df["consistent"]).sum()
        ),
        "dataset": dataset.summary(),
    }
    LOGGER.info("YOLO data ready: %s", summary["dataset"]["images_per_split"])
    return json_safe(summary)


def validate_datasets() -> dict[str, Any]:
    """Check the structure, labels and image/label pairing of both datasets."""
    import pandas as pd
    import yaml

    paths = _paths()
    required = [
        paths.splits_path,
        paths.balanced_path,
        paths.image_metadata_path,
        paths.yolo_manifest_path,
        paths.yolo_data_yaml,
    ]
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        raise DatasetValidationError(
            "Missing prepared data files: " + ", ".join(missing)
        )

    splits = load_json(paths.splits_path)
    balanced = load_json(paths.balanced_path)
    metadata = pd.read_csv(paths.image_metadata_path)
    classes = list(splits.get("classes", []))
    if len(classes) < 2:
        raise DatasetValidationError(
            f"Expected at least two classes, found {len(classes)}"
        )

    split_sets: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        entry = splits.get(split, {})
        split_paths = list(entry.get("paths", []))
        labels = [int(value) for value in entry.get("labels", [])]
        if not split_paths:
            raise DatasetValidationError(f"Classification split {split!r} is empty")
        if len(split_paths) != len(labels):
            raise DatasetValidationError(
                f"Classification split {split!r} has {len(split_paths)} paths "
                f"but {len(labels)} labels"
            )
        invalid_labels = [
            label for label in labels if label < 0 or label >= len(classes)
        ]
        if invalid_labels:
            raise DatasetValidationError(
                f"Classification split {split!r} has labels outside "
                f"0..{len(classes) - 1}: {invalid_labels[:5]}"
            )
        split_sets[split] = set(split_paths)

    for left, right in combinations(split_sets, 2):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise DatasetValidationError(
                f"Classification splits {left!r} and {right!r} overlap: "
                f"{len(overlap)} files"
            )

    balanced_labels = [int(value) for value in balanced.get("labels", [])]
    balanced_counts = Counter(balanced_labels)
    if set(balanced_counts) != set(range(len(classes))):
        raise DatasetValidationError(
            "Balanced training manifest does not cover every class"
        )
    if len(set(balanced_counts.values())) != 1:
        raise DatasetValidationError(
            "Balanced training manifest has unequal class counts: "
            f"{dict(sorted(balanced_counts.items()))}"
        )

    unreadable = metadata.loc[
        ~metadata["readable"].fillna(False).astype(bool), "path"
    ].tolist()
    if unreadable:
        raise DatasetValidationError(
            f"{len(unreadable)} classification images are unreadable: "
            f"{unreadable[:5]}"
        )

    manifest = load_json(paths.yolo_manifest_path)
    invalid_annotations = int(
        manifest.get("counts", {}).get("invalid_annotations", 0)
    )
    if invalid_annotations:
        raise DatasetValidationError(
            f"YOLO manifest contains {invalid_annotations} invalid annotations"
        )
    unreadable_yolo = manifest.get("image_checks", {}).get("unreadable", [])
    if unreadable_yolo:
        raise DatasetValidationError(
            f"{len(unreadable_yolo)} YOLO images are unreadable: "
            f"{unreadable_yolo[:5]}"
        )

    for split in ("Train", "Val"):
        valid_records = [
            record
            for record in manifest.get("splits", {}).get(split, [])
            if record.get("valid")
        ]
        if not valid_records:
            raise DatasetValidationError(
                f"YOLO manifest split {split!r} has no valid records"
            )

    data_config = yaml.safe_load(paths.yolo_data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data_config, dict):
        raise DatasetValidationError(f"{paths.yolo_data_yaml} is not valid YAML")
    manifest_classes = list(manifest.get("classes", []))
    if int(data_config.get("nc", -1)) != len(manifest_classes):
        raise DatasetValidationError(
            "data.yaml class count does not match the YOLO manifest"
        )
    if list(data_config.get("names", [])) != manifest_classes:
        raise DatasetValidationError(
            "data.yaml class names do not match the YOLO manifest"
        )

    yolo_images: dict[str, set[str]] = {}
    for split in YOLO_TARGET_SPLITS:
        image_dir = paths.yolo_dataset_dir / "images" / split
        label_dir = paths.yolo_dataset_dir / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise DatasetValidationError(
                f"YOLO directory missing for split {split!r}"
            )
        images = {path.stem for path in image_dir.iterdir() if path.is_file()}
        labels = {path.stem for path in label_dir.iterdir() if path.is_file()}
        if not images:
            raise DatasetValidationError(f"YOLO split {split!r} has no images")
        if images != labels:
            raise DatasetValidationError(
                f"YOLO split {split!r} has {len(images.symmetric_difference(labels))} "
                "unpaired image/label files"
            )
        yolo_images[split] = images

    for left, right in combinations(yolo_images, 2):
        overlap = yolo_images[left] & yolo_images[right]
        if overlap:
            raise DatasetValidationError(
                f"YOLO splits {left!r} and {right!r} share {len(overlap)} stems"
            )

    expected_total = sum(
        1
        for split in ("Train", "Val")
        for record in manifest.get("splits", {}).get(split, [])
        if record.get("valid")
    )
    actual_total = sum(len(stems) for stems in yolo_images.values())
    if actual_total != expected_total:
        raise DatasetValidationError(
            f"YOLO dataset has {actual_total} images but the manifest has "
            f"{expected_total} valid pairs"
        )

    summary = {
        "classification": {
            "classes": classes,
            "splits": {
                split: len(split_sets[split]) for split in ("train", "val", "test")
            },
            "balanced_train": len(balanced_labels),
        },
        "detection": {
            "classes": manifest_classes,
            "images": {
                split: len(stems) for split, stems in yolo_images.items()
            },
            "invalid_annotations": invalid_annotations,
            "mixed_class_images": int(
                sum(
                    not record.get("consistent", True)
                    for record in (
                        manifest.get("splits", {}).get("Train", [])
                        + manifest.get("splits", {}).get("Val", [])
                    )
                )
            ),
        },
    }
    LOGGER.info("Dataset validation passed: %s", summary)
    return json_safe(summary)


def train_classification_frozen() -> dict[str, Any]:
    """Train the frozen InceptionV3 backbone + head and log the stage."""
    from diagnostic_des_tumeurs.inception.functions.preprocessing import (
        build_datasets,
    )
    from diagnostic_des_tumeurs.inception.functions.tracking import log_stage
    from diagnostic_des_tumeurs.inception.functions.training import train_frozen

    paths = _paths()
    config = InceptionConfig()
    data = _load_classification_data(paths)
    balanced = load_json(paths.balanced_path)
    datasets = build_datasets(paths, config, data.splits, balanced=balanced)

    LOGGER.info("Training the frozen InceptionV3 backbone + head")
    result = train_frozen(paths, config, datasets, data)
    run_id = log_stage(
        result, config, data, datasets, paths, INCEPTION_EXPERIMENT
    )
    summary = {
        "model": result.model_path.as_posix(),
        "run_id": run_id,
        "epochs_completed": len(result.history.get("loss", [])),
        "test_accuracy": result.evaluation.accuracy,
        "test_metrics": result.evaluation.metrics,
    }
    LOGGER.info("Frozen training complete: accuracy=%s", summary["test_accuracy"])
    return json_safe(summary)


def fine_tune_classification() -> dict[str, Any]:
    """Fine-tune the last InceptionV3 layers and log the stage."""
    from diagnostic_des_tumeurs.inception.functions.preprocessing import (
        build_datasets,
    )
    from diagnostic_des_tumeurs.inception.functions.tracking import log_stage
    from diagnostic_des_tumeurs.inception.functions.training import fine_tune

    paths = _paths()
    config = InceptionConfig()
    data = _load_classification_data(paths)
    datasets = build_datasets(paths, config, data.splits)

    LOGGER.info("Fine-tuning the last %s InceptionV3 layers", config.unfreeze_last_n)
    result = fine_tune(paths, config, datasets, data)
    run_id = log_stage(
        result, config, data, datasets, paths, INCEPTION_EXPERIMENT
    )
    summary = {
        "model": result.model_path.as_posix(),
        "run_id": run_id,
        "epochs_completed": len(result.history.get("loss", [])),
        "test_accuracy": result.evaluation.accuracy,
        "test_metrics": result.evaluation.metrics,
    }
    LOGGER.info("Fine-tuning complete: accuracy=%s", summary["test_accuracy"])
    return json_safe(summary)


def train_detection() -> dict[str, Any]:
    """Train YOLOv8n and log the training run."""
    from diagnostic_des_tumeurs.yolo.functions.tracking import log_training
    from diagnostic_des_tumeurs.yolo.functions.training import train

    paths = _paths()
    config = YoloConfig()

    LOGGER.info("Training YOLOv8n")
    result = train(paths, config)
    run_id = log_training(result, config, paths, YOLO_EXPERIMENT)
    summary = {
        "weights": result.weights.as_posix(),
        "run_dir": result.run_dir.as_posix(),
        "run_id": run_id,
        "device": result.device_name,
        "epochs_completed": result.summary["epochs_completed"],
        "validation_metrics": result.validation_metrics,
    }
    LOGGER.info(
        "YOLO training complete: val mAP50-95=%s",
        summary["validation_metrics"].get("metrics/mAP50-95(B)"),
    )
    return json_safe(summary)


def evaluate_classification() -> dict[str, Any]:
    """Compare classification runs and register the best one in MLflow."""
    from diagnostic_des_tumeurs.helpers.mlflow import configure_experiment
    from diagnostic_des_tumeurs.inception.functions.tracking import (
        register_best,
        save_comparison,
    )

    paths = _paths()
    configure_experiment(INCEPTION_EXPERIMENT, paths)

    LOGGER.info("Comparing Inception runs and registering the best model")
    comparison = save_comparison(INCEPTION_EXPERIMENT, paths)
    registered = register_best(INCEPTION_EXPERIMENT, paths)
    summary = {
        "comparison_rows": len(comparison),
        "registered_model": registered,
    }
    LOGGER.info("Classification best model: %s", registered)
    return json_safe(summary)


def evaluate_detection() -> dict[str, Any]:
    """Evaluate YOLO on the test split, log results and register the best run."""
    from diagnostic_des_tumeurs.yolo.functions.evaluation import evaluate
    from diagnostic_des_tumeurs.yolo.functions.tracking import (
        log_evaluation,
        register_best,
        save_comparison,
    )

    paths = _paths()
    config = YoloConfig()

    LOGGER.info("Evaluating YOLOv8n on the test split")
    result = evaluate(paths, config)
    run_id = log_evaluation(result, paths, YOLO_EXPERIMENT)
    comparison = save_comparison(YOLO_EXPERIMENT, paths)
    registered = register_best(YOLO_EXPERIMENT, paths)
    summary = {
        "run_id": run_id,
        "test_metrics": result.metrics,
        "verdict": result.verdict,
        "comparison_rows": len(comparison),
        "registered_model": registered,
    }
    LOGGER.info("YOLO best model: %s", registered)
    return json_safe(summary)


def save_artifacts() -> dict[str, Any]:
    """Check the saved models/artefacts and write a local inventory manifest."""
    paths = _paths()
    required = [
        paths.inception_transfer,
        paths.inception_finetuned,
        paths.inception_frozen_history,
        paths.inception_finetuned_history,
        paths.inception_comparison_json,
        paths.inception_registered_model,
        paths.yolo_saved_model,
        paths.yolo_training_summary,
        paths.yolo_training_args,
        paths.yolo_training_results,
        paths.yolo_comparison_json,
        paths.yolo_registered_model,
        paths.yolo_evaluation_dir / "test_evaluation.json",
    ]
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        raise DatasetValidationError(
            "Cannot save artefacts; missing: " + ", ".join(missing)
        )

    candidates = required + [
        paths.inception_comparison_csv,
        paths.inception_plots_dir,
        paths.yolo_training_plots,
        paths.yolo_evaluation_dir / "plots",
        paths.yolo_evaluation_dir / "test_results.csv",
    ]
    entries: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.is_dir():
            entries.extend(
                _artifact_entry(path, paths.root)
                for path in sorted(
                    entry for entry in candidate.rglob("*") if entry.is_file()
                )
            )
        elif candidate.is_file():
            entries.append(_artifact_entry(candidate, paths.root))

    manifest = {
        "created_by": "diagnostic_des_tumeurs.airflow_tasks",
        "created_at": datetime.now(UTC).isoformat(),
        "root": paths.root.as_posix(),
        "count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "artifacts": entries,
    }
    output = paths.models_dir / ARTIFACTS_MANIFEST
    save_json(output, manifest)
    LOGGER.info("Artefact inventory written to %s", output)
    return json_safe(
        {
            "manifest": output.as_posix(),
            "count": manifest["count"],
            "total_bytes": manifest["total_bytes"],
        }
    )


def _artifact_entry(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
