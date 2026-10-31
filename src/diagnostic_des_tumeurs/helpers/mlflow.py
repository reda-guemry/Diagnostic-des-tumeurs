"""Shared MLflow tracking, comparison and model-registration helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from diagnostic_des_tumeurs.config.hyperparameters import DEFAULT_TRACKING_URI_ENV
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import float_metrics

_METRIC_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def metric_key(name: str) -> str:
    """Turn ``metrics/mAP50-95(B)`` into a portable MLflow metric name."""
    cleaned = _METRIC_PATTERN.sub("_", str(name)).strip("_")
    return cleaned or "metric"


def configure_experiment(
    experiment: str,
    paths: Paths,
    tracking_uri: str | None = None,
) -> str:
    """Point MLflow at the project SQLite store and select ``experiment``."""
    uri = (
        tracking_uri
        or os.environ.get(DEFAULT_TRACKING_URI_ENV)
        or f"sqlite:///{paths.mlflow_db.resolve()}"
    )
    mlflow.set_tracking_uri(uri)

    client = MlflowClient()
    if client.get_experiment_by_name(experiment) is None:
        paths.mlflow_artifacts.mkdir(parents=True, exist_ok=True)
        client.create_experiment(
            experiment,
            artifact_location=paths.mlflow_artifacts.resolve().as_uri(),
        )
    mlflow.set_experiment(experiment)
    return uri


def _param_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def log_params(params: Mapping[str, Any]) -> None:
    cleaned = {
        str(key): _param_value(value)
        for key, value in params.items()
        if value is not None
    }
    mlflow.log_params(cleaned)


def log_metrics(
    metrics: Mapping[str, Any],
    prefix: str = "",
    step: int | None = None,
) -> None:
    cleaned = {
        metric_key(f"{prefix}{key}"): value
        for key, value in float_metrics(metrics).items()
    }
    if cleaned:
        mlflow.log_metrics(cleaned, step=step)


def log_history(history: Mapping[str, Any], step_offset: int = 1) -> None:
    """Log one metric value per epoch from a Keras-style history dict."""
    series = {
        key: list(values)
        for key, values in history.items()
        if isinstance(values, (list, tuple))
    }
    if not series:
        return
    length = min(len(values) for values in series.values())
    for index in range(length):
        log_metrics(
            {key: values[index] for key, values in series.items()},
            step=index + step_offset,
        )


def log_artifacts(
    artifacts: Iterable[str | Path],
    artifact_path: str | None = None,
) -> None:
    for artifact in artifacts:
        path = Path(artifact)
        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        elif path.is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)


def compare_runs(
    experiment: str,
    metric: str,
    include_failed: bool = False,
    extra_filter: str | None = None,
) -> pd.DataFrame:
    """Return every run of ``experiment``, best value of ``metric`` first."""
    filters = []
    if not include_failed:
        filters.append("attributes.status = 'FINISHED'")
    if extra_filter:
        filters.append(extra_filter)
    filter_string = " and ".join(filters)
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=filter_string,
        order_by=[f"metrics.{metric} DESC"],
    )
    if runs.empty:
        return runs
    keep = [
        "run_id",
        "tags.mlflow.runName",
        "status",
        "start_time",
        f"metrics.{metric}",
        "params.stage",
    ]
    keep = [column for column in keep if column in runs.columns]
    other_metrics = [
        column
        for column in runs.columns
        if column.startswith("metrics.")
        and column not in keep
        and column != f"metrics.{metric}"
    ]
    columns = keep + other_metrics
    return runs.loc[:, columns].sort_values(
        f"metrics.{metric}", ascending=False, na_position="last"
    )


def register_best_run(
    experiment: str,
    metric: str,
    model_name: str,
    description: str | None = None,
    extra_filter: str | None = None,
) -> dict[str, Any]:
    """Register the model logged by the best run and alias it as ``best``."""
    runs = compare_runs(experiment, metric, extra_filter=extra_filter)
    runs = runs.dropna(subset=[f"metrics.{metric}"])
    if runs.empty:
        raise RuntimeError(
            f"No finished MLflow run found for experiment {experiment!r}"
        )

    best = runs.iloc[0]
    run_id = str(best["run_id"])
    model_uri = f"runs:/{run_id}/model"

    model_version = mlflow.register_model(model_uri, model_name)
    client = MlflowClient()
    value = float(best[f"metrics.{metric}"])
    note = description or (
        f"Selected automatically from experiment {experiment!r}: "
        f"{metric}={value:.6f}"
    )
    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=note,
    )
    client.set_model_version_tag(model_name, model_version.version, "metric", metric)
    client.set_model_version_tag(
        model_name, model_version.version, "metric_value", f"{value:.6f}"
    )
    client.set_model_version_tag(model_name, model_version.version, "run_id", run_id)
    client.set_registered_model_alias(model_name, "best", model_version.version)

    return {
        "name": model_name,
        "version": str(model_version.version),
        "run_id": run_id,
        "metric": metric,
        "metric_value": value,
        "source": model_uri,
        "alias": "best",
        "description": note,
    }
