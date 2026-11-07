"""Airflow DAG for the brain-tumour data, training and evaluation pipeline.

The DAG only orchestrates the existing package functions. Training,
preprocessing and evaluation logic remain in ``diagnostic_des_tumeurs``.

Task graph::

    prepare_classification_data ┐
                                ├─> validate_datasets
    prepare_detection_data      ┘
        ├─> train_frozen ─> fine_tune ─> evaluate_and_register (classification)
        └─> train ─────────────────────> evaluate_and_register (detection)
                                             └─> save_artifacts

The DAG is manual (``schedule=None``) and allows one active run at a time so
concurrent runs cannot rebuild the same datasets or Ultralytics run directory.

Set ``BTD_PROJECT_ROOT`` when Airflow is not started from the project root.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG, TaskGroup

from diagnostic_des_tumeurs.airflow_tasks import (
    evaluate_classification,
    evaluate_detection,
    fine_tune_classification,
    prepare_classification_data,
    prepare_detection_data,
    save_artifacts,
    train_classification_frozen,
    train_detection,
    validate_datasets,
)

LOGGER = logging.getLogger(__name__)


def _on_failure(context: dict[str, Any]) -> None:
    task_instance = context.get("task_instance")
    LOGGER.error(
        "Airflow task %s failed for run %s: %s",
        getattr(task_instance, "task_id", "unknown"),
        context.get("run_id", "unknown"),
        context.get("exception", "unknown error"),
    )


def _on_retry(context: dict[str, Any]) -> None:
    task_instance = context.get("task_instance")
    LOGGER.warning(
        "Retrying Airflow task %s (try %s) for run %s",
        getattr(task_instance, "task_id", "unknown"),
        getattr(task_instance, "try_number", "unknown"),
        context.get("run_id", "unknown"),
    )


default_args = {
    "owner": "diagnostic-des-tumeurs",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": _on_failure,
    "on_retry_callback": _on_retry,
}


with DAG(
    dag_id="brain_tumour_pipeline",
    description=(
        "Prepare data, train InceptionV3 and YOLOv8, evaluate, log to MLflow "
        "and save the produced artefacts."
    ),
    default_args=default_args,
    start_date=datetime(2024, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["brain-tumour", "inceptionv3", "yolov8", "mlflow"],
    doc_md=__doc__,
) as dag:
    with TaskGroup("data_preparation") as data_preparation:
        prepare_classification = PythonOperator(
            task_id="prepare_classification_data",
            python_callable=prepare_classification_data,
        )
        prepare_detection = PythonOperator(
            task_id="prepare_detection_data",
            python_callable=prepare_detection_data,
        )
        validate = PythonOperator(
            task_id="validate_datasets",
            python_callable=validate_datasets,
        )
        [prepare_classification, prepare_detection] >> validate

    with TaskGroup("classification") as classification:
        train_frozen = PythonOperator(
            task_id="train_frozen",
            python_callable=train_classification_frozen,
            retries=0,
        )
        fine_tune = PythonOperator(
            task_id="fine_tune",
            python_callable=fine_tune_classification,
            retries=0,
        )
        evaluate_classification_task = PythonOperator(
            task_id="evaluate_and_register",
            python_callable=evaluate_classification,
        )
        train_frozen >> fine_tune >> evaluate_classification_task

    with TaskGroup("detection") as detection:
        train_yolo = PythonOperator(
            task_id="train",
            python_callable=train_detection,
            retries=0,
        )
        evaluate_detection_task = PythonOperator(
            task_id="evaluate_and_register",
            python_callable=evaluate_detection,
        )
        train_yolo >> evaluate_detection_task

    with TaskGroup("artifacts") as artifacts:
        save = PythonOperator(
            task_id="save_artifacts",
            python_callable=save_artifacts,
        )

    validate >> train_frozen
    validate >> train_yolo
    evaluate_classification_task >> save
    evaluate_detection_task >> save
