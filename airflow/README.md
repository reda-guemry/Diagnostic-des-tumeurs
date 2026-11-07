# Airflow orchestration

The DAG in `airflow/dags/brain_tumour_pipeline.py` orchestrates the existing
InceptionV3 and YOLOv8 pipelines. It does not duplicate or change the training,
preprocessing or evaluation code; each task calls a function from
`diagnostic_des_tumeurs.airflow_tasks`, which in turn calls the package code.

## Task graph

```
data_preparation.prepare_classification_data ┐
                                             ├─> data_preparation.validate_datasets
data_preparation.prepare_detection_data      ┘
        │
        ├─> classification.train_frozen ─> classification.fine_tune
        │       └─> classification.evaluate_and_register
        │
        └─> detection.train ─> detection.evaluate_and_register
                                      │
                                      └─> artifacts.save_artifacts
```

- **Data preparation** validates the classification folders and the YOLO
  archive, writes the split/balanced manifests, and rebuilds `data/yolo/`.
- **Validation** checks missing files, label ranges, split overlap, class
  balance, unreadable images, YOLO annotation errors and image/label pairing.
- **Classification** trains the frozen head then fine-tunes InceptionV3. Each
  stage evaluates and logs its MLflow run.
- **Detection** trains YOLOv8n, then evaluates on the test split and logs the
  test run.
- **Evaluation/registration** compares MLflow runs and registers the best model
  with the alias `best`.
- **Artifact saving** checks the model/report files and writes
  `models/airflow_artifacts.json`, an inventory with sizes and SHA-256 hashes.
  The large model files stay in their canonical locations instead of being
  duplicated.

The DAG is manual (`schedule=None`), catch-up is disabled and
`max_active_runs=1` prevents concurrent runs from rebuilding the same datasets or
Ultralytics run directory.

## Local setup

Airflow 3 is kept out of the project's runtime dependencies. Install it in the
same virtual environment after syncing the project:

```bash
uv sync
uv pip install -r airflow/requirements.txt
```

Airflow needs an `AIRFLOW_HOME` and a DAG folder. The following uses a local
`.airflow/` directory and links the project DAG folder into it:

```bash
export AIRFLOW_HOME="$PWD/.airflow"
export BTD_PROJECT_ROOT="$PWD"
mkdir -p "$AIRFLOW_HOME"
ln -sfn "$PWD/airflow/dags" "$AIRFLOW_HOME/dags"

# --no-sync keeps uv from removing the optional Airflow packages
uv run --no-sync airflow standalone
```

Open the Airflow UI at <http://localhost:8080>. The standalone command prints the
generated admin credentials in the terminal.

`BTD_PROJECT_ROOT` is only required when Airflow is not started from the project
root. It must point to the directory containing `pyproject.toml` and
`src/diagnostic_des_tumeurs/`.

## Docker Compose profile

The root `Dockerfile` has an optional `airflow` target, and
`docker-compose.yml` exposes it as the `orchestration` profile. The runtime,
API, UI and MLflow services are not affected when the profile is disabled.

```bash
# Start the tracking server first
docker compose up -d mlflow

# Build and start Airflow
docker compose --profile orchestration up --build airflow
```

The container:

- sets `AIRFLOW_HOME=/app/.airflow` and exposes `airflow/dags` as the DAG
  folder (bind mount plus entrypoint fallback);
- mounts `./data` and `./models` read-write so training tasks can write outputs;
- reaches MLflow at `http://mlflow:5000`;
- uses `airflow standalone`, with the same manual DAG and retry policy as the
  local deployment.

Open <http://localhost:8080> and trigger `brain_tumour_pipeline` manually. The
first Airflow build is heavy because it installs Airflow on top of the runtime
image. For production, use a managed Airflow deployment and a shared metadata
database instead of `airflow standalone`.

## Running the pipeline

1. Open the `brain_tumour_pipeline` DAG.
2. Trigger it manually.
3. Follow task status in the Grid/Graph view and open a task log for the
   captured Python/TensorFlow/Ultralytics output.

The DAG uses the normal project defaults:

- MLflow tracking: `mlflow.db` and `mlartifacts/` under the project root.
- Classification: `data/processed/` and `models/inception_reports/`.
- Detection: `data/yolo/` and `models/yolo/`.

## Error handling and logs

- Data preparation and validation retry once after five minutes.
- Training tasks have `retries=0`: restarting a failed long training run is a
  deliberate choice, not an automatic side effect.
- Evaluation and artifact tasks retry once.
- `on_failure_callback` and `on_retry_callback` write the task id, run id and
  exception to the Airflow task log.
- Airflow records every task state (`queued`, `running`, `success`, `failed`,
  `upstream_failed`) in its metadata database and UI.

## Using a different Airflow deployment

The DAG only needs:

1. The `diagnostic-des-tumeurs` package importable by the Airflow workers.
2. `BTD_PROJECT_ROOT` pointing at the project.
3. Access to the project data, models and SQLite MLflow store.
4. Enough CPU/RAM/GPU resources for TensorFlow and Ultralytics.

For Docker or a multi-worker deployment, mount the project at the same absolute
path on every worker, or set `BTD_PROJECT_ROOT` to that mounted path. The
training tasks expect the data and model directories to be shared between
workers.
