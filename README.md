# Brain-tumour diagnosis pipelines

Two reproducible pipelines for the brain-tumour dataset:

- **Classification** with a frozen InceptionV3 backbone plus a small head,
  followed by fine-tuning of the last InceptionV3 layers.
- **Detection** with YOLOv8n trained on the YOLO-format archive.

Both pipelines can be run from the command line or explored notebook by
notebook. The implementation lives in `src/diagnostic_des_tumeurs/`; the
notebooks in `notebooks/` document the steps and display the results.

## Setup

```bash
uv sync
```

## Run a pipeline

```bash
# Both workflows, one after the other
uv run diagnostic-des-tumeurs all

# One workflow
uv run diagnostic-des-tumeurs inception
uv run diagnostic-des-tumeurs yolo

# Installed aliases (the YOLO one avoids Ultralytics' own `yolo` command)
uv run btd-inception
uv run btd-yolo
```

Every workflow rebuilds its derived data, trains/evaluates, logs to MLflow and
registers the best model. Use `--help` for the small set of overridable
hyperparameters:

```bash
uv run diagnostic-des-tumeurs inception --help
uv run diagnostic-des-tumeurs yolo --help
```

Examples:

```bash
uv run btd-inception --epochs 10 --fine-tune-epochs 5
uv run btd-yolo --epochs 30 --imgsz 512 --batch 16
```

## Data layout

```
data/
├── Data/
│   └── {glioma,meningioma,notumor,pituitary}/   # classification images
└── archive/
    └── {Train,Val}/<class>/{images,labels}/     # YOLO images + .txt labels
```

The classification pipeline writes `data/processed/splits.json`,
`data/processed/train_balanced.json` and `data/processed/image_metadata.csv`.
The detection pipeline validates the archive into
`data/processed/yolo_manifest.json` and builds `data/yolo/` with a stratified
`train` / `valid` / `test` split (4737 / 255 / 255 images). The test split is
carved from the archive `Val` folder; `Train` is never used for validation or
test.

## MLflow

Tracking uses a SQLite store (`mlflow.db`) because the model registry is not
available with a plain file store. Artifacts live in `mlartifacts/`.

- `inception-brain-tumor`: frozen stage and fine-tuning runs, best run selected
  by test accuracy.
- `yolo-brain-tumor`: training and test-evaluation runs, best training run
  selected by validation `mAP50-95`.

Best models are registered with the alias `best`:

```python
import mlflow

classifier = mlflow.pyfunc.load_model("models:/inception-brain-tumor@best")
detector = mlflow.pyfunc.load_model("models:/brain-tumor-yolov8@best")
```

The YOLO model takes a DataFrame with an `image` column of local image paths and
the optional params `conf`, `iou` and `imgsz`. Each pipeline also writes a
`registered_model.json` with the version, run id, metric and source URI.

## Reports

- `notebooks/README.md` — notebook-by-notebook walkthrough and output paths.
- `models/inception_reports/` — histories, plots, comparisons and the
  registered-model record.
- `models/yolo/` — saved weights, training summary/args/results, curves,
  evaluation tables and plots, comparison and registered-model record.

Generated YOLO datasets, reports, new weights and MLflow files are ignored by
Git. The existing `models/inceptionv3_transfer.keras` checkpoint remains
tracked.
