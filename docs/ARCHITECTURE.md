# Architecture and code reference

This document describes the runtime architecture, the main modules and the
principal functions of `diagnostic-des-tumeurs`. It is intended for developers
who need to change the pipelines without reading every notebook or source file
first.

- [1. System overview](#1-system-overview)
- [2. Runtime services](#2-runtime-services)
- [3. Repository layout](#3-repository-layout)
- [4. End-to-end data flow](#4-end-to-end-data-flow)
- [5. Configuration and paths](#5-configuration-and-paths)
- [6. Classification pipeline](#6-classification-pipeline)
- [7. Detection pipeline](#7-detection-pipeline)
- [8. MLflow tracking and registry](#8-mlflow-tracking-and-registry)
- [9. FastAPI inference service](#9-fastapi-inference-service)
- [10. Streamlit interface](#10-streamlit-interface)
- [11. Airflow orchestration](#11-airflow-orchestration)
- [12. Docker and Compose](#12-docker-and-compose)
- [13. Environment variables](#13-environment-variables)
- [14. Invariants and extension points](#14-invariants-and-extension-points)

---

## 1. System overview

The project contains two independent model pipelines that share configuration,
I/O helpers, a React-free Streamlit front end and an HTTP inference layer.

| Pipeline | Model | Input | Output |
| --- | --- | --- | --- |
| Classification | InceptionV3 + dense head | MRI image | one of `glioma`, `meningioma`, `notumor`, `pituitary` plus softmax scores |
| Detection | YOLOv8n | MRI image | bounding boxes, class names and confidence scores |

Both pipelines can be run in three ways:

1. **CLI** — `diagnostic-des-tumeurs`, `btd-inception`, `btd-yolo`.
2. **Airflow** — the `brain_tumour_pipeline` DAG calls the same Python functions.
3. **Web** — Streamlit → FastAPI → local Keras/YOLO weights.

The API and UI never train a model. They load the checkpoints already present
under `models/`.

```text
Browser
  │
  ▼
Streamlit UI (port 8501)
  │ HTTP multipart image
  ▼
FastAPI API (port 8000)
  ├── InceptionV3 classifier (Keras)
  └── YOLOv8n detector (Ultralytics)
```

The optional MLflow tracking server (port 5000) is used by the training and
evaluation workflows. The optional Airflow service (port 8080) orchestrates
those workflows.

---

## 2. Runtime services

The Compose stack separates the application into four services:

| Service | Image target | Responsibility | Port |
| --- | --- | --- | --- |
| `mlflow` | `runtime` | Tracking server, SQLite metadata store, artifact store | 5000 |
| `api` | `runtime` | FastAPI inference endpoints | 8000 |
| `ui` | `runtime` | Streamlit front end | 8501 |
| `airflow` | `airflow` (profile) | Optional DAG scheduler/webserver | 8080 |

The `api` service only mounts `data/` and `models/` read-only. The `ui` service
does not need the datasets or checkpoints at all. The `airflow` service mounts
them read-write because its tasks create splits, reports and model files.

The default Compose command starts `mlflow`, `api` and `ui`:

```bash
docker compose up --build
```

The Airflow service is opt-in:

```bash
docker compose --profile orchestration up --build airflow
```

---

## 3. Repository layout

```text
.
├── Dockerfile                  # runtime + optional Airflow targets
├── docker-compose.yml          # service topology
├── docker/                     # container entrypoints
├── app/streamlit_app.py        # Streamlit front end
├── airflow/
│   ├── dags/brain_tumour_pipeline.py
│   ├── requirements.txt
│   └── README.md
├── src/diagnostic_des_tumeurs/
│   ├── api.py                  # FastAPI application
│   ├── airflow_tasks.py        # DAG task callables
│   ├── main.py                 # top-level CLI dispatcher
│   ├── config/
│   │   ├── paths.py            # every input/output path
│   │   └── hyperparameters.py  # user-facing model defaults
│   ├── helpers/
│   │   ├── io.py               # JSON/YAML and JSON-safe conversion
│   │   └── mlflow.py           # tracking, comparison and registration
│   ├── inception/
│   │   ├── main.py
│   │   ├── functions/          # data, preprocessing, model, training, evaluation, tracking
│   │   └── helpers/
│   └── yolo/
│       ├── main.py
│       ├── functions/          # inspection, dataset, training, evaluation, tracking
│       └── helpers/            # annotations, plots, MLflow pyfunc model
├── data/
│   ├── Data/                   # classification images by class
│   ├── archive/                # source YOLO images + labels
│   └── processed/              # manifests, splits and generated figures
├── models/                     # checkpoints and generated reports
├── docs/                       # architecture and test-result documentation
└── notebooks/                  # step-by-step exploration notebooks
```

Generated data under `data/yolo/`, reports under `models/inception_reports/`
and `models/yolo/`, local MLflow files and Docker-local state are ignored by Git.

---

## 4. End-to-end data flow

### Classification

```text
data/Data/<class>/*.jpg
        │
        ▼
build_splits() ──► data/processed/splits.json
        │          data/processed/image_metadata.csv
        ▼
build_balanced_manifest() ──► data/processed/train_balanced.json
        │
        ▼
build_datasets() ──► tf.data.Dataset (train/val/test)
        │
        ├── train_frozen() ──► models/inceptionv3_transfer.keras
        │                       history, plots, confusion matrix
        └── fine_tune() ─────► models/inceptionv3_finetuned.keras
                                history, plots, confusion matrix
        │
        ▼
evaluate_model() ──► test metrics
        │
        ▼
log_stage() ──► MLflow run + registered model alias `best`
```

### Detection

```text
data/archive/{Train,Val}/<class>/{images,labels}
        │
        ▼
build_manifest() ──► data/processed/yolo_manifest.json
        │
        ▼
prepare_dataset() ──► data/yolo/{images,labels}/{train,valid,test}
        │             data/yolo/data.yaml
        ▼
train() ──► models/yolo/brain_tumor_yolov8n.pt
        │    training_summary.json, results.csv, plots
        ▼
evaluate() ──► test metrics, error analysis, prediction figures
        │
        ▼
log_training() / log_evaluation() ──► MLflow runs
        │
        ▼
register_best() ──► model alias `best`
```

The YOLO archive `Train` folder becomes `train`. The archive `Val` folder is
split 50/50, stratified by class, into `valid` and `test`. The classification
test split is stratified and never used during training.

---

## 5. Configuration and paths

### `Paths` — `src/diagnostic_des_tumeurs/config/paths.py`

`Paths.from_root(root=None)` builds every input/output path from one project
root. The root is `Path.cwd()`, or the explicit `root` argument. If the current
directory is `notebooks/`, the parent directory is used.

The most important attributes are:

| Attribute | Meaning |
| --- | --- |
| `data_dir` | `data/Data`, classification source images |
| `archive_dir` | `data/archive`, YOLO source images/labels |
| `processed_dir` | `data/processed`, manifests and splits |
| `yolo_dataset_dir` | `data/yolo`, generated Ultralytics dataset |
| `inception_transfer` | `models/inceptionv3_transfer.keras` |
| `inception_finetuned` | `models/inceptionv3_finetuned.keras` |
| `yolo_saved_model` | `models/yolo/brain_tumor_yolov8n.pt` |
| `mlflow_db` | `mlflow.db` at the project root |
| `mlflow_artifacts` | `mlartifacts/` at the project root |

`Paths.ensure_output_dirs()` creates the generated report/model directories.
It is idempotent and is called at the start of every pipeline and Airflow task.

### `InceptionConfig` and `YoloConfig` — `config/hyperparameters.py`

The dataclasses hold the user-facing defaults:

- `InceptionConfig`: image size, batch size, frozen/fine-tuning epochs, learning
  rates, number of unfrozen layers, dropout, dense units, patience and seed.
- `YoloConfig`: optional Ultralytics overrides and a seed. `None` values mean
  “use the device defaults from `resolve_training_args`”. The project keeps the
  standard Ultralytics defaults for CPU/GPU, all available CPU workers and no
  custom augmentation.

### Environment overrides

| Variable | Used by | Effect |
| --- | --- | --- |
| `BTD_PROJECT_ROOT` | API, Streamlit, Airflow | Project root when the current directory is not the repository |
| `BTD_API_URL` | Streamlit | FastAPI base URL, default `http://127.0.0.1:8000` |
| `MLFLOW_TRACKING_URI` | training/evaluation | Overrides the SQLite tracking URI |

### I/O helpers — `helpers/io.py`

| Function | Responsibility |
| --- | --- |
| `load_json(path)` | Read a UTF-8 JSON object. |
| `save_json(path, payload)` | Create parent directories and write JSON-safe output. |
| `save_yaml(path, payload)` | Create parent directories and write YAML. |
| `relative_path(path, root)` | Return a POSIX path relative to the project root. |
| `json_safe(value)` | Recursively convert numpy/pandas/path values into JSON-friendly objects. |
| `float_metrics(values)` | Keep only finite numeric metrics for MLflow/JSON. |

---

## 6. Classification pipeline

### Top-level CLI — `main.py`

```python
run_all(root=None, inception_config=None, yolo_config=None,
        register=True, tracking=True) -> dict[str, Any]
```

Runs the Inception workflow and then the YOLO workflow in one process.
`main(argv)` dispatches `diagnostic-des-tumeurs [all|inception|yolo]` and keeps
the two workflows independent.

### `inception/main.py`

```python
run(root=None, config=None, experiment=INCEPTION_EXPERIMENT,
    register=True, tracking=True) -> InceptionSummary
```

Runs the complete classification workflow:

1. `build_splits`
2. `build_balanced_manifest`
3. `build_datasets(..., balanced=...)`
4. `train_frozen`
5. `build_datasets(...)` on the original split
6. `fine_tune`
7. `save_comparison` and `register_best` when tracking is enabled

`InceptionSummary.to_dict()` returns the class balance, split sizes, per-stage
metrics and registered-model record as a JSON-safe dictionary.

### Data functions — `inception/functions/data.py`

| Function | Responsibility |
| --- | --- |
| `build_splits(paths, seed=42, image_size=(224, 224))` | Discover class folders, validate images, create stratified train/val/test splits, write `splits.json` and `image_metadata.csv`, and draw overview figures. |
| `class_balance(metadata, classes)` | Return the number of images per class. |
| `ClassificationData` | Immutable container for classes, label maps, metadata, splits and figure paths. |

### Preprocessing — `inception/functions/preprocessing.py`

| Function | Responsibility |
| --- | --- |
| `load_image(path, image_size, label=None)` | Read with OpenCV, convert BGR→RGB, resize and return `float32` pixels in `[0, 255]`. |
| `augment_data(image, label)` | Apply one random blur, horizontal flip or Gaussian-noise transform. |
| `make_dataset(paths, labels, config, training=False, augmentation="frozen")` | Build a batched `tf.data.Dataset`; training datasets are shuffled and augmented. |
| `build_balanced_manifest(paths, splits, seed=42)` | Oversample minority classes by repeating paths, without writing augmented images. |
| `build_datasets(paths, config, splits=None, balanced=None)` | Build train/val/test datasets and carry their paths/labels/classes in `ClassificationDatasets`. |

The frozen stage uses the balanced manifest and the `frozen` augmentation. The
fine-tuning stage uses the original training split and flip-plus-noise
augmentation, matching the notebooks.

### Model — `inception/functions/model.py`

```python
build_model(num_classes, input_shape, dropout=0.5, dense_units=256)
```

Builds one Keras model: frozen InceptionV3 backbone → global average pooling →
dense head → dropout → softmax. Image rescaling to `[-1, 1]` is part of the
model, so the data pipeline can keep raw `[0, 255]` pixels.

### Training — `inception/functions/training.py`

| Function | Responsibility |
| --- | --- |
| `train_frozen(paths, config, datasets, data)` | Trains only the head, evaluates the test split, saves `inceptionv3_transfer.keras`, history, curves and confusion matrix. |
| `fine_tune(paths, config, datasets, data)` | Loads the frozen model, unfreezes the last `unfreeze_last_n` InceptionV3 layers (BatchNorm stays frozen), trains with balanced class weights, evaluates and saves `inceptionv3_finetuned.keras`. |

Both return `TrainingResult` (`stage`, model, history, evaluation, output paths,
class weights and trainable-layer count).

### Evaluation — `inception/functions/evaluation.py`

| Function | Responsibility |
| --- | --- |
| `evaluate_model(model, dataset, labels, classes)` | Run test predictions and build accuracy, loss, classification report, per-class rows and confusion matrix. |
| `save_confusion_matrix(evaluation, classes, path, title)` | Save the confusion-matrix figure. |
| `save_history_plot(history, path, title)` | Save training/validation curves. |

`Evaluation.metrics` exposes `loss`, `accuracy`, `macro_precision`,
`macro_recall`, `macro_f1` and `weighted_f1`. `Evaluation.per_class` contains the
per-class precision/recall/F1/support rows.

### Tracking — `inception/functions/tracking.py`

| Function | Responsibility |
| --- | --- |
| `log_stage(result, config, data, datasets, paths, experiment)` | Log params, epoch history, test metrics, per-class metrics, figures, manifests and the TensorFlow model to MLflow. Returns the run id. |
| `save_comparison(experiment, paths, metric="test_accuracy")` | Export the MLflow run comparison to `comparison.csv` / `comparison.json`. |
| `register_best(experiment, paths, metric="test_accuracy")` | Register the best run and set the `best` alias. |

---

## 7. Detection pipeline

### `yolo/main.py`

```python
run(root=None, config=None, experiment=YOLO_EXPERIMENT,
    register=True, tracking=True) -> YoloSummary
```

Runs inspection → dataset preparation → training → test evaluation →
comparison/registration. `YoloSummary.to_dict()` returns data checks, split
counts, training metadata, test metrics, verdict and the registered model.

### Inspection — `yolo/functions/inspection.py`

```python
build_manifest(paths, seed=42) -> ManifestReport
```

Scans `data/archive/{Train,Val}/<class>/{images,labels}`, pairs images and
labels, validates YOLO lines with `parse_yolo_label`, records unreadable images
and mixed-class annotations, and writes `data/processed/yolo_manifest.json` plus
overview figures.

### Dataset preparation — `yolo/functions/dataset.py`

```python
prepare_dataset(paths, seed=42, test_size=0.5) -> DatasetReport
```

Copies the validated pairs into `data/yolo/`, splits archive `Val` into
`valid`/`test`, writes `data/yolo/data.yaml`, verifies image/label pairing and
raises if the copies are inconsistent.

### Training — `yolo/functions/training.py`

| Function | Responsibility |
| --- | --- |
| `resolve_training_args(paths, config)` | Apply GPU/CPU defaults (30/10 epochs, 640/320 px, 32/8 batch), all CPU workers, standard Ultralytics augmentation and return the argument dictionary plus device name. |
| `train(paths, config=None)` | Train YOLOv8n, validate, copy `best.pt` to `models/yolo/brain_tumor_yolov8n.pt`, save summaries/args/results and plots. |
| `per_class_table(metrics)` | Convert Ultralytics validation metrics into a per-class DataFrame. |

### Evaluation — `yolo/functions/evaluation.py`

| Function | Responsibility |
| --- | --- |
| `evaluate(paths, config=None)` | Run `model.val` on the `test` split, compute per-class metrics, error analysis and validation-vs-test comparison, and write the evaluation artifacts. |
| `_error_analysis(...)` | Match predictions and ground-truth boxes at IoU 0.5 to count TP/FP/FN per class. |
| `_sample_predictions(...)` | Render one prediction sample per class for the evaluation report. |

### Tracking — `yolo/functions/tracking.py`

| Function | Responsibility |
| --- | --- |
| `log_training(result, config, paths, experiment)` | Log training params, epoch metrics, validation metrics, plots and the YOLO pyfunc model. |
| `log_evaluation(result, paths, experiment)` | Log test metrics, per-class metrics and evaluation artifacts. |
| `save_comparison(experiment, paths)` | Export the training-run comparison, filtered by `params.stage = 'train'`. |
| `register_best(experiment, paths)` | Register the best training run by validation `mAP50-95` and set the `best` alias. |

---

## 8. MLflow tracking and registry

`helpers/mlflow.py` centralises the behaviour shared by both pipelines:

| Function | Responsibility |
| --- | --- |
| `configure_experiment(experiment, paths, tracking_uri=None)` | Use `MLFLOW_TRACKING_URI` if set, otherwise `sqlite:///<project>/mlflow.db`; create the experiment if needed. |
| `metric_key(name)` | Convert names such as `metrics/mAP50-95(B)` into portable MLflow metric keys. |
| `log_params`, `log_metrics`, `log_history`, `log_artifacts` | Thin, null-safe wrappers around MLflow logging. |
| `compare_runs(experiment, metric, ...)` | Return finished runs ordered by a metric. |
| `register_best_run(experiment, metric, model_name, ...)` | Register the model artifact of the best run and assign the `best` alias. |

The SQLite backend is required because the MLflow model registry is not
available with a plain file store. In Compose, the tracking URI points to the
`mlflow` service over HTTP and its data lives in `./mlflow-data/`.

---

## 9. FastAPI inference service

`src/diagnostic_des_tumeurs/api.py` creates a small FastAPI application. Models
are loaded once and cached with `lru_cache`.

| Endpoint | Input | Output |
| --- | --- | --- |
| `GET /` | — | Service name and docs link |
| `GET /health` | — | `status` (`ok`/`degraded`) and the detected model filenames |
| `POST /classify` | multipart `file` | Predicted class, confidence and all class scores |
| `POST /detect` | multipart `file`, form `conf`, `iou`, `imgsz` | Image size, box count and detections with `xyxy`, class id/name and confidence |

Internal helpers:

| Function | Responsibility |
| --- | --- |
| `_paths()` | Build `Paths` from `BTD_PROJECT_ROOT`. |
| `_find_classifier(paths)` | Select the fine-tuned checkpoint, or the transfer fallback. |
| `_find_detector(paths)` | Select the saved YOLO checkpoint, or the latest run's `best.pt`. |
| `_classifier()` / `_detector()` | Load and cache the Keras/YOLO model. |
| `_class_names()` | Read class names from `data/processed/splits.json`. |
| `_warm_up_detector()` | Initialise YOLO/PyTorch before TensorFlow is used. |
| `_read_image(file)` | Validate the upload, apply EXIF orientation and convert to RGB. |

Important implementation details:

- `_find_classifier` prefers `models/inceptionv3_finetuned.keras` and falls back
  to `models/inceptionv3_transfer.keras`.
- `_find_detector` prefers `models/yolo/brain_tumor_yolov8n.pt` and then the
  latest run's `best.pt`.
- `_warm_up_detector` runs one dummy YOLO prediction during startup. On this
  TensorFlow/PyTorch stack, running TensorFlow first can abort the process with
  `free(): invalid pointer`; initialising YOLO first avoids that crash.
- Images are EXIF-corrected, converted to RGB and resized to the classifier
  input size. Classification pixels stay in `[0, 255]` because rescaling is
  inside the Keras model.
- Missing models return HTTP 503 with an actionable message; invalid images
  return HTTP 400.

---

## 10. Streamlit interface

`app/streamlit_app.py` is a single-page front end with two tabs:

1. **Classification** — calls `/classify`, shows the predicted class, global
   confidence and one score bar per class.
2. **Detection** — calls `/detect` with a confidence slider, draws bounding
   boxes with PIL and displays a table of classes, confidences and coordinates.

UI helpers:

| Function | Responsibility |
| --- | --- |
| `_post(endpoint, api_url, filename, image_bytes, content_type, data=None)` | POST the multipart image and surface API errors in the UI. |
| `_class_label(name)` / `_class_color(name)` | Map model class names to French labels and theme colours. |
| `_score_bars(scores)` | Render one confidence bar per class. |
| `_draw_detections(image, detections)` | Draw YOLO boxes and labels over the uploaded image. |
| `_load_font(size)` | Load DejaVu Sans when available, otherwise Pillow's default font. |

The sidebar contains the API URL and a health check. Results are cached in
`st.session_state` and reset when a new image is uploaded. The page uses the
dark theme from `.streamlit/config.toml`.

The API URL is read from `BTD_API_URL`; Compose sets it to `http://api:8000`.

---

## 11. Airflow orchestration

`airflow/dags/brain_tumour_pipeline.py` is a thin Airflow 3 DAG. It imports task
callables from `diagnostic_des_tumeurs.airflow_tasks` and never duplicates
training, preprocessing or evaluation logic.

```text
prepare_classification_data ┐
                            ├─> validate_datasets ─> train_frozen ─> fine_tune
prepare_detection_data      ┘                    └─> train

train_frozen ─> fine_tune ─> evaluate_classification ┐
train ─────────────────────> evaluate_detection     ├─> save_artifacts
```

Key task functions:

| Function | Responsibility |
| --- | --- |
| `project_root()` | Resolve `BTD_PROJECT_ROOT` and verify it is a project root. |
| `prepare_classification_data()` | Rebuild splits and the balanced manifest. |
| `prepare_detection_data()` | Rebuild the YOLO manifest and `data/yolo/`. |
| `validate_datasets()` | Validate files, labels, split overlap, class balance, unreadable images, annotation errors and image/label pairing. |
| `train_classification_frozen()` | Call `train_frozen` and log the stage. |
| `fine_tune_classification()` | Call `fine_tune` and log the stage. |
| `train_detection()` | Call `train` and log the training run. |
| `evaluate_classification()` | Compare runs and register the best classifier. |
| `evaluate_detection()` | Evaluate on `test`, log metrics and register the best detector. |
| `save_artifacts()` | Verify required outputs and write `models/airflow_artifacts.json` with sizes and SHA-256 hashes. |

The DAG is manual (`schedule=None`), has `catchup=False` and
`max_active_runs=1`. Training tasks use `retries=0`; preparation, validation,
evaluation and artifact tasks retry once. See `airflow/README.md` for the full
operating guide.

---

## 12. Docker and Compose

### Image targets

`Dockerfile` has two targets:

- `runtime` — Python 3.12, system libraries for OpenCV/TensorFlow, the locked
  project environment and the API/UI/Airflow source. This is the default image
  for `api`, `ui` and `mlflow`.
- `airflow` — extends `runtime`, installs `airflow/requirements.txt` and adds
  the entrypoint that links `airflow/dags` into `AIRFLOW_HOME`.

Datasets and weights are deliberately **not** copied into the image. They are
mounted from the host:

| Host path | Container path | Access |
| --- | --- | --- |
| `./data` | `/app/data` | read-only for `api`, read-write for `airflow` |
| `./models` | `/app/models` | read-only for `api`, read-write for `airflow` |
| `./mlflow-data` | `/mlflow` | read-write for `mlflow` |
| `./.airflow` | `/app/.airflow` | read-write for `airflow` |
| `./airflow/dags` | `/app/.airflow/dags` and `/app/airflow/dags` | read-only for `airflow` |

### Health checks

- `api`: `GET /health`
- `ui`: `GET /_stcore/health`
- `mlflow`: `GET /health`

`ui` waits for `api` to become healthy. `airflow` waits for `mlflow` to become
healthy. The API health check returns 200 even when a model is missing; the
payload reports `degraded` and `/classify` or `/detect` returns 503.

### Volumes and persistence

`docker compose down` keeps the host directories. `docker compose down -v`
does not delete bind mounts, so the local checkpoints, datasets and MLflow data
remain safe. Delete `mlflow-data/` or `.airflow/` explicitly if you want a
clean reset.

---

## 13. Environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `BTD_PROJECT_ROOT` | detected from the package/cwd | Set to `/app` in Compose |
| `BTD_API_URL` | `http://127.0.0.1:8000` | Set to `http://api:8000` for the Compose UI |
| `MLFLOW_TRACKING_URI` | `sqlite:///<root>/mlflow.db` | Set to `http://mlflow:5000` in Compose |
| `BTD_API_PORT` | `8000` | Host port, Compose interpolation |
| `BTD_UI_PORT` | `8501` | Host port, Compose interpolation |
| `BTD_MLFLOW_PORT` | `5000` | Host port, Compose interpolation |
| `BTD_AIRFLOW_PORT` | `8080` | Host port, Compose interpolation |

---

## 14. Invariants and extension points

Keep these invariants when changing the project:

1. **No training logic in the DAG, API or UI.** They call the existing package
   functions.
2. **The test split stays untouched.** Classification test images and the YOLO
   `test` split are never used for fitting or model selection.
3. **The classifier keeps its preprocessing inside the model.** The data
   pipeline supplies `[0, 255]` pixels; the Keras model rescales to `[-1, 1]`.
4. **YOLO stays on Ultralytics defaults.** The only explicit project overrides
   are the seed, device defaults, worker count and standard training arguments.
5. **Use `btd-yolo`**, not `yolo`, to avoid the Ultralytics console script.
6. **MLflow uses SQLite.** The registry and the `best` alias need a database
   backend.
7. **Model and data directories are runtime state.** Do not bake checkpoints or
   datasets into the Docker image.
8. **The API loads local checkpoints directly.** It does not depend on the
   MLflow registry being online.

Useful extension points:

- Add a new model stage in `inception/functions/training.py` and expose it
  through `inception/main.py` and `airflow_tasks.py`.
- Add a new YOLO evaluation metric in `yolo/functions/evaluation.py`; tracking
  will log it through `log_evaluation`.
- Add a new API endpoint in `api.py` and a matching tab in
  `app/streamlit_app.py`.
- Add a new Compose service by following the `api` service pattern: build the
  `runtime` target, mount only the directories it needs and add a health check.
