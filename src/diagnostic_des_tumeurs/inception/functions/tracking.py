"""MLflow logging for the InceptionV3 workflow."""

from __future__ import annotations

from dataclasses import asdict

import mlflow
import mlflow.tensorflow
import pandas as pd

from diagnostic_des_tumeurs.config.hyperparameters import (
    INCEPTION_REGISTERED_MODEL,
    InceptionConfig,
)
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import save_json
from diagnostic_des_tumeurs.helpers.mlflow import (
    compare_runs,
    configure_experiment,
    log_artifacts,
    log_history,
    log_metrics,
    log_params,
    metric_key,
    register_best_run,
)
from diagnostic_des_tumeurs.inception.functions.data import ClassificationData
from diagnostic_des_tumeurs.inception.functions.preprocessing import (
    ClassificationDatasets,
)
from diagnostic_des_tumeurs.inception.functions.training import TrainingResult


def _stage_params(
    result: TrainingResult,
    config: InceptionConfig,
    data: ClassificationData,
    datasets: ClassificationDatasets,
) -> dict[str, object]:
    params = asdict(config)
    params.update(
        {
            "stage": result.stage,
            "epochs": (
                config.epochs if result.stage == "frozen" else config.fine_tune_epochs
            ),
            "learning_rate": (
                config.learning_rate
                if result.stage == "frozen"
                else config.fine_tune_learning_rate
            ),
            "num_classes": len(data.classes),
            "classes": ",".join(data.classes),
            "train_size": len(datasets.train_paths),
            "val_size": len(datasets.val_paths),
            "test_size": len(datasets.test_paths),
        }
    )
    if result.trainable_backbone_layers is not None:
        params["trainable_backbone_layers"] = result.trainable_backbone_layers
    return params


def log_stage(
    result: TrainingResult,
    config: InceptionConfig,
    data: ClassificationData,
    datasets: ClassificationDatasets,
    paths: Paths,
    experiment: str,
) -> str:
    """Log one training stage (frozen head or fine-tuning) and its model."""
    configure_experiment(experiment, paths)

    with mlflow.start_run(run_name=f"inception-{result.stage}") as run:
        mlflow.set_tags(
            {
                "pipeline": "inception",
                "stage": result.stage,
                "model": "inceptionv3",
                "dataset": "data/Data",
            }
        )
        log_params(_stage_params(result, config, data, datasets))
        log_history(result.history)
        log_metrics(result.evaluation.metrics, prefix="test_")
        for row in result.evaluation.per_class:
            class_prefix = f"test_{metric_key(str(row['class']))}_"
            log_metrics(
                {
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1-score"],
                    "support": row["support"],
                },
                prefix=class_prefix,
            )

        log_artifacts(
            [
                result.history_path,
                result.plot_path,
                result.confusion_path,
                paths.splits_path,
                paths.balanced_path,
                paths.image_metadata_path,
            ]
        )
        mlflow.tensorflow.log_model(result.model, name="model")
        return run.info.run_id


def save_comparison(
    experiment: str, paths: Paths, metric: str = "test_accuracy"
) -> pd.DataFrame:
    comparison = compare_runs(experiment, metric)
    if not comparison.empty:
        comparison.to_csv(paths.inception_comparison_csv, index=False)
        save_json(
            paths.inception_comparison_json,
            comparison.to_dict(orient="records"),
        )
    return comparison


def register_best(
    experiment: str, paths: Paths, metric: str = "test_accuracy"
) -> dict[str, object]:
    info = register_best_run(
        experiment=experiment,
        metric=metric,
        model_name=INCEPTION_REGISTERED_MODEL,
        description=(
            "Best InceptionV3 run selected by test accuracy. "
            "Load with mlflow.pyfunc.load_model('models:/inception-brain-tumor@best')."
        ),
    )
    save_json(paths.inception_registered_model, info)
    return info
