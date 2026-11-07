# Brain-tumour diagnosis pipelines

Two reproducible workflows for brain-tumour MRI analysis:

- **Classification** with an InceptionV3 backbone and a small dense head,
  followed by fine-tuning of the last InceptionV3 layers.
- **Detection and localisation** with YOLOv8n trained on the YOLO-format
  archive.

The project provides a CLI, a FastAPI inference API, a Streamlit interface, an
Airflow DAG and a Docker Compose stack that keeps the web, tracking and
orchestration services separated.

| Layer | Technology | Entry point |
| --- | --- | --- |
| Classification | TensorFlow / Keras / InceptionV3 | `btd-inception` |
| Detection | Ultralytics / YOLOv8n | `btd-yolo` |
| Inference API | FastAPI + Uvicorn | `diagnostic_des_tumeurs.api:app` |
| Web UI | Streamlit | `app/streamlit_app.py` |
| Orchestration | Airflow 3 | `airflow/dags/brain_tumour_pipeline.py` |
| Tracking | MLflow with SQLite | `mlflow.db` / Compose `mlflow` service |

## Test-set results

Measured on the project's held-out test splits. The full protocol, per-class
tables, confusion matrix and raw JSON are in
[`docs/TEST_RESULTS.md`](docs/TEST_RESULTS.md).

| Model | Test data | Main result |
| --- | --- | --- |
| InceptionV3 transfer checkpoint | 1,054 images | **92.41 % accuracy**, **92.12 % macro F1** |
| YOLOv8n final checkpoint | 255 images / 272 boxes | **98.09 % mAP50**, **79.74 % mAP50-95** |

InceptionV3 per class: glioma F1 92.54 %, meningioma 84.96 %, notumor 97.00 %,
pituitary 93.97 %. YOLO per class mAP50-95: meningioma 85.44 %, no tumor
79.25 %, glioma 77.82 %, pituitary 76.46 %.

> The repository contains the transfer-stage classifier
> `models/inceptionv3_transfer.keras` but not a completed
> `models/inceptionv3_finetuned.keras`. The classification figures describe the
> checkpoint currently deployed by the API, not a fine-tuned model. See
> [`docs/TEST_RESULTS.md`](docs/TEST_RESULTS.md) for details.

## Quick start with Docker Compose

### Prerequisites

- Docker Engine with the Compose plugin.
- The datasets under `data/` and the model artifacts under `models/` (they are
  mounted into the containers, not baked into the image).
- Enough RAM for TensorFlow and PyTorch. The inference API and the UI do not
  need a GPU.

### Start the stack

```bash
cp .env.example .env          # optional: change host ports
docker compose up --build
```

The first build installs TensorFlow, PyTorch/Ultralytics and the rest of the
locked Python environment, so it can take several minutes.

| Service | URL | Purpose |
| --- | --- | --- |
| Streamlit UI | <http://localhost:8501> | Upload an MRI and run classification or detection |
| FastAPI docs | <http://localhost:8000/docs> | Interactive API documentation |
| MLflow | <http://localhost:5000> | Tracking server used by training/evaluation |
| Airflow | <http://localhost:8080> | Optional, only with the `orchestration` profile |

The UI receives `BTD_API_URL=http://api:8000` from Compose and waits for the API
health check before starting.

### Stop the stack

```bash
docker compose down
```

`data/`, `models/`, `mlflow-data/` and `.airflow/` are bind mounts and survive a
`down`. Delete `mlflow-data/` or `.airflow/` explicitly for a clean reset.

### Optional Airflow orchestration

Airflow is built from the dedicated `airflow` image target and is not started by
default:

```bash
docker compose --profile orchestration up --build airflow
```

The service uses `airflow standalone`, runs the existing DAG and writes its
metadata to `.airflow/`. The DAG is manual (`schedule=None`); open the Airflow UI
and trigger `brain_tumour_pipeline` when you are ready. Training tasks are not
started automatically.

### Service separation

| Service | Mounts | Network |
| --- | --- | --- |
| `api` | `./data` and `./models` read-only | receives UI requests |
| `ui` | none | calls `http://api:8000` |
| `mlflow` | `./mlflow-data` read-write | tracking server |
| `airflow` (profile) | `./data`, `./models`, `./.airflow`, DAG folder | calls MLflow and the package code |

The shared `runtime` image contains the application and its Python dependencies.
The datasets and checkpoints remain outside the image.

## Local Python workflow

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Run the API and UI locally

```bash
# Terminal 1 — FastAPI (Swagger at http://127.0.0.1:8000/docs)
uv run uvicorn diagnostic_des_tumeurs.api:app --port 8000

# Terminal 2 — Streamlit (http://localhost:8501)
uv run streamlit run app/streamlit_app.py
```

Set `BTD_API_URL` to point the UI at another API, and `BTD_PROJECT_ROOT` when
the commands are not run from the project root.

### Run the pipelines from the CLI

```bash
uv run diagnostic-des-tumeurs all
uv run diagnostic-des-tumeurs inception
uv run diagnostic-des-tumeurs yolo

# Installed aliases (btd-yolo avoids Ultralytics' own `yolo` command)
uv run btd-inception
uv run btd-yolo
```

A full run rebuilds derived data, trains and evaluates both models, logs to
MLflow and registers the best runs. Hyperparameter overrides are intentionally
small:

```bash
uv run btd-inception --epochs 10 --fine-tune-epochs 5
uv run btd-yolo --epochs 30 --imgsz 512 --batch 16
```

Use `--help` for the complete flag list. Training is never triggered by the
Docker API/UI services.

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Report `ok`/`degraded` and the classification/detection model files |
| `POST` | `/classify` | Multipart `file`; returns the predicted class, confidence and every class score |
| `POST` | `/detect` | Multipart `file` plus optional `conf`, `iou`, `imgsz`; returns boxes (`xyxy`), class names and confidences |

Example:

```bash
curl -F "file=@data/Data/glioma/Tr-gl_0714.jpg" \
  http://127.0.0.1:8000/classify
```

The classifier prefers `models/inceptionv3_finetuned.keras` and falls back to
`models/inceptionv3_transfer.keras`. The detector prefers
`models/yolo/brain_tumor_yolov8n.pt` and then the latest run's `best.pt`. Models
are loaded once and cached. If a model is missing, `/health` reports `degraded`
and the related inference endpoint returns HTTP 503.

## Airflow

The DAG prepares and validates data, trains both models, evaluates them, logs to
MLflow, registers the best runs and writes an artifact inventory. It calls the
existing package functions; training and preprocessing logic are not duplicated.

```bash
uv sync
uv pip install -r airflow/requirements.txt

export AIRFLOW_HOME="$PWD/.airflow"
export BTD_PROJECT_ROOT="$PWD"
mkdir -p "$AIRFLOW_HOME"
ln -sfn "$PWD/airflow/dags" "$AIRFLOW_HOME/dags"

uv run --no-sync airflow standalone
```

See [`airflow/README.md`](airflow/README.md) for the task graph, validation
rules, retry policy, Docker profile and deployment notes.

## Data layout

```text
data/
├── Data/
│   └── {glioma,meningioma,notumor,pituitary}/   # classification images
├── archive/
│   └── {Train,Val}/<class>/{images,labels}/     # YOLO images + .txt labels
└── processed/                                   # generated manifests/splits
```

The classification pipeline writes `data/processed/splits.json`,
`data/processed/train_balanced.json` and `data/processed/image_metadata.csv`.
The detection pipeline validates the archive into
`data/processed/yolo_manifest.json` and builds `data/yolo/` with a stratified
`train` / `valid` / `test` split (4,737 / 255 / 255 images). The test split is
carved from the archive `Val` folder; `Train` is never used for validation or
test.

## Models, MLflow and reports

MLflow uses a SQLite backend because the model registry is not available with a
plain file store. In Compose, the database and artifacts live in
`./mlflow-data/`; a local `uv` run uses `mlflow.db` and `mlartifacts/` at the
project root.

- `inception-brain-tumor` — frozen and fine-tuning runs; best run selected by
  test accuracy.
- `yolo-brain-tumor` — training and test-evaluation runs; best training run
  selected by validation `mAP50-95`.

Both pipelines register the selected model with the alias `best`:

```python
import mlflow

classifier = mlflow.pyfunc.load_model("models:/inception-brain-tumor@best")
detector = mlflow.pyfunc.load_model("models:/brain-tumor-yolov8@best")
```

The YOLO pyfunc model accepts a DataFrame with an `image` column of local image
paths, plus optional `conf`, `iou` and `imgsz` params. Each pipeline also writes
a `registered_model.json` record.

Generated outputs:

- `models/inception_reports/` — histories, plots, comparisons and registered
  model record.
- `models/yolo/` — weights, training summary/args/results, curves, evaluation
  tables/plots and registered model record.
- `models/airflow_artifacts.json` — Airflow artifact inventory with sizes and
  SHA-256 hashes.
- `docs/results/` — raw test-set metric JSON files.

Generated YOLO data, reports, new weights and MLflow files are ignored by Git.
The existing `models/inceptionv3_transfer.keras` and
`models/yolo/brain_tumor_yolov8n.pt` checkpoints are tracked.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — services, data flow, module
  responsibilities, principal functions, Docker topology and extension points.
- [`docs/TEST_RESULTS.md`](docs/TEST_RESULTS.md) — test protocol, metrics,
  per-class results and confusion matrix.
- [`airflow/README.md`](airflow/README.md) — DAG operation and deployment.
- [`notebooks/README.md`](notebooks/README.md) — notebook-by-notebook walkthrough.

## Notes and limitations

- This is a research prototype, not a medical device.
- The API loads local checkpoints; it does not require MLflow to be online.
- The Airflow Docker profile installs optional orchestration dependencies on top
  of the runtime image. Use a managed Airflow deployment for production.
- The first Compose build downloads large TensorFlow/PyTorch wheels; subsequent
  builds reuse the dependency layer.
- Containers run as root by default, so files created in `models/` or
  `mlflow-data/` by the Airflow profile may be owned by root. Adjust ownership
  with `sudo chown -R "$USER:$USER" models mlflow-data .airflow` if needed.
- The classifier checkpoint currently deployed is the transfer-stage model
  because the fine-tuned artifact is not present. Re-evaluate after fine-tuning.

## Project layout

```text
.
├── Dockerfile
├── docker-compose.yml
├── app/streamlit_app.py
├── airflow/
│   ├── dags/brain_tumour_pipeline.py
│   ├── requirements.txt
│   └── README.md
├── src/diagnostic_des_tumeurs/
│   ├── api.py
│   ├── airflow_tasks.py
│   ├── config/
│   ├── helpers/
│   ├── inception/
│   └── yolo/
├── data/
├── models/
├── docs/
└── notebooks/
```
