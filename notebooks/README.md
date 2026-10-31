# Notebook and package pipelines

Two independent pipelines live in this project: InceptionV3 classification and
YOLOv8 detection. Both are available as notebooks and as the installable
`diagnostic_des_tumeurs` package.

## Package layout

```
src/diagnostic_des_tumeurs/
├── main.py                 # run both workflows
├── config/
│   ├── paths.py            # every input/output path, derived from the root
│   └── hyperparameters.py  # the few user-facing knobs
├── helpers/
│   ├── io.py               # JSON/YAML and JSON-safe conversion
│   └── mlflow.py           # tracking, comparison, best-model registration
├── inception/
│   ├── main.py             # full classification workflow
│   ├── functions/          # data, preprocessing, model, training, evaluation, tracking
│   └── helpers/            # figures
└── yolo/
    ├── main.py             # full detection workflow
    ├── functions/          # inspection, dataset, training, evaluation, tracking
    └── helpers/            # annotations, plots, MLflow pyfunc model
```

Run everything from the project root:

```bash
uv sync

uv run diagnostic-des-tumeurs all          # both pipelines, in order
uv run diagnostic-des-tumeurs inception    # classification only
uv run diagnostic-des-tumeurs yolo         # detection only

uv run diagnostic-des-tumeurs inception --help
uv run diagnostic-des-tumeurs yolo --help
```

The installed console scripts are `btd-inception` and `btd-yolo`. They are
short aliases for the two workflows; the detection entry point is deliberately
not called `yolo` because Ultralytics already installs a command with that name.

Only a handful of hyperparameters are exposed. Paths, dataset layout and
`data.yaml` are derived automatically, and YOLO uses its normal Ultralytics
defaults for everything else. On CPU the YOLO defaults are 10 epochs at 320 px
with batch 8; on GPU they are 100 epochs at 640 px with batch 32. The data loader
uses all available CPU workers. Override only what you need, for example:

```bash
uv run btd-yolo --epochs 30 --imgsz 512 --batch 16
```

## MLflow

Both workflows log to MLflow with a SQLite backend (`mlflow.db` plus
`mlartifacts/`). A plain file store cannot host the model registry, so do not
replace the tracking URI with a `file:` URI if you want registration.

- Inception experiment: `inception-brain-tumor`
  - one run for the frozen backbone + head (`inception-frozen`),
  - one run for fine-tuning (`inception-fine_tune`),
  - best run selected by test accuracy and registered as
    `inception-brain-tumor` with alias `best`.
- YOLO experiment: `yolo-brain-tumor`
  - one training run (`yolo-train`),
  - one evaluation run (`yolo-test`, excluded from registration),
  - best training run selected by validation `mAP50-95` and registered as
    `brain-tumor-yolov8` with alias `best`.

Each pipeline writes a `registered_model.json` next to its artefacts. Reuse a
registered model directly:

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/inception-brain-tumor@best")
model = mlflow.pyfunc.load_model("models:/brain-tumor-yolov8@best")
```

The YOLO wrapper accepts a DataFrame with an `image` column (local image paths)
and the optional params `conf`, `iou` and `imgsz`.

## 1. Classification (TensorFlow / Keras)

Uses the classification dataset in `data/Data/` (one folder per class, no
annotations).

1. **`01_data_overview.ipynb`** — class discovery, file checks (readable images,
   dimensions, channels), class distribution, sample images and a stratified
   70/15/15 train/validation/test split.
   - Outputs: `data/processed/splits.json`, `data/processed/image_metadata.csv`.
2. **`02_preprocessing_augmentation.ipynb`** — image loading, blur / flip /
   Gaussian-noise augmentation, class balancing by oversampling and a `tf.data`
   pipeline with on-the-fly augmentation.
   - Output: `data/processed/train_balanced.json`.
   - Augmentation is applied in memory only; no augmented image is written.
3. **`03_feature_extraction.ipynb`** — builds a **single** model made of a
   frozen InceptionV3 backbone plus a small classifier head, trains it on the
   balanced training set (only the head learns), evaluates it on the test split
   and saves the model.
   - Output: `models/inceptionv3_transfer.keras`.
   - The backbone and the head are never split into separate models and no
     `.npy` feature arrays are produced.
4. **`04_fine_tuning.ipynb`** — loads the saved model, unfreezes the last 30
   InceptionV3 layers (BatchNorm stays frozen), fine-tunes with a small learning
   rate on the original training split with class weights, evaluates on the test
   split and saves the final model.
   - Output: `models/inceptionv3_finetuned.keras`.

Notes:

- Notebook 03 uses InceptionV3 from Keras Applications instead of the old
  InceptionV1/TF-Hub model because `tensorflow-hub` 0.16.1 provides a Keras 2
  layer, which is incompatible with Keras 3.
- Notebook 03 must run before notebook 04: notebook 04 starts from
  `models/inceptionv3_transfer.keras`.
- Training in notebook 03 runs the frozen backbone for every batch, so on CPU
  expect a few minutes per epoch.

## 2. Object detection (YOLOv8 / Ultralytics)

Uses the YOLO-format dataset in `data/archive/`: images **and** one `.txt`
annotation per image, stored as `{Train,Val}/<class>/images` and
`.../labels`.

1. **`05_yolo_data_overview.ipynb`** — scans the archive, pairs images and
   annotations by base name, reports images without annotation and annotations
   without image, validates the YOLO format and class consistency, plots the
   class distribution and shows sample images with their bounding boxes.
   - Output: `data/processed/yolo_manifest.json` (validated pairs).
   - Current archive: 4737 paired `Train` images, 510 paired `Val` images,
     272 images whose annotation contains another class (reported but kept),
     and 4 discarded files (2 missing labels, 2 missing images).
   - Read-only: nothing in `data/archive/` is modified.
2. **`06_yolo_preprocessing.ipynb`** — recreates `data/yolo/` with
   `images/{train,valid,test}` and `labels/{train,valid,test}`, copies each
   image together with its annotation, splits the archive `Val` set 50/50
   (stratified) into validation and test, and generates `data.yaml`.
   - Output: `data/yolo/` (4737 train / 255 valid / 255 test, ignored by Git).
   - There is no separate `train_config.json`; the YOLO hyperparameters live in
     `config/hyperparameters.py`.
3. **`07_yolo_training.ipynb`** — picks GPU or CPU defaults, trains YOLOv8n
   with the standard Ultralytics settings, plots the losses and metrics,
   validates on the validation split, saves the artefacts and logs the run to
   MLflow.
   - Outputs: `models/yolo/brain_tumor_yolov8n.pt`,
     `models/yolo/training_summary.json`, `training_args.yaml`,
     `training_results.csv`, `training_plots/`.
4. **`08_yolo_evaluation.ipynb`** — loads the trained model, evaluates it on
   the **test** split (precision, recall, mAP@0.5, mAP@0.5:0.95), shows the
   confusion matrix / PR / F1 curves, compares ground truth and predictions,
   runs a TP/FP/FN analysis at IoU 0.5, compares validation vs test to judge
   generalization, logs the evaluation run and registers the best training run.
   - Outputs: `models/yolo/evaluation/test_evaluation.json`,
     `models/yolo/evaluation/plots/` and `models/yolo/registered_model.json`.

Notes:

- Notebooks 05–08 are thin wrappers around
  `diagnostic_des_tumeurs.yolo.functions`; the package is the source of truth
  and the notebooks mainly display the returned tables and figures.
- `data/yolo/`, `models/yolo/` and the MLflow store are generated and ignored
  by Git; notebooks 05 and 06 can be rerun to rebuild the dataset.
- The CPU defaults train at 320 px for 10 epochs (about 5 min/epoch on a
  laptop). On GPU the notebook automatically switches to 640 px and 100 epochs.
- The test split is carved from the archive `Val` folder because the archive
  has no separate test set; `Train` is never used for validation or test.

## Environment

Run `uv sync` once before executing the notebooks. The project uses Python
3.12, TensorFlow 2.21 / Keras 3.15 for classification, Ultralytics 8.4 / PyTorch
for YOLOv8, and MLflow 3 for tracking and the model registry.
