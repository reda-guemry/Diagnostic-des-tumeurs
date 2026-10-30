# Notebook pipelines

Two independent pipelines live in this folder. Run each notebook in order and
from the project root or from the `notebooks/` directory.

Every code cell is preceded by a short summary that explains what the step does
and how to read its output.

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
4. **`04_fine_tuning.ipynb`** — loads the saved model, unfreezes the last 20
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
   - Read-only: nothing in `data/archive/` is modified.
2. **`06_yolo_preprocessing.ipynb`** — recreates `data/yolo/` with
   `images/{train,valid,test}` and `labels/{train,valid,test}`, copies each
   image together with its annotation, splits the archive `Val` set 50/50 into
   validation and test, and writes the dataset and augmentation configuration.
   - Outputs: `data/yolo/data.yaml`, `data/yolo/train_config.json` and the
     copied dataset (ignored by Git).
3. **`07_yolo_training.ipynb`** — chooses a GPU or CPU training profile,
   trains YOLOv8n with the augmentation policy, plots the losses and metrics,
   validates on the validation split and saves the artefacts.
   - Outputs: `models/yolo/brain_tumor_yolov8n.pt`,
     `models/yolo/training_summary.json`, `training_args.yaml`,
     `training_results.csv`, `training_plots/`.
4. **`08_yolo_evaluation.ipynb`** — loads the trained model, evaluates it on
   the **test** split (precision, recall, mAP@0.5, mAP@0.5:0.95), shows the
   confusion matrix / PR / F1 curves, compares ground truth and predictions,
   runs a TP/FP/FN analysis at IoU 0.5, compares validation vs test to judge
   generalization and optionally fine-tunes the model.
   - Outputs: `models/yolo/evaluation/test_evaluation.json` and
     `models/yolo/evaluation/plots/`.

Notes:

- `data/yolo/` and `models/yolo/runs/` are generated and ignored by Git;
  `05` and `06` can be rerun to rebuild them.
- The CPU profile trains at 320 px for 10 epochs (about 5 min/epoch on a
  laptop). On GPU the notebook automatically switches to 640 px and 100 epochs.
- The test split is carved from the archive `Val` folder because the archive
  has no separate test set; `Train` is never used for validation or test.

## Environment

Run `uv sync` once before executing the notebooks. The project uses Python
3.12, TensorFlow 2.21 / Keras 3.15 for classification and Ultralytics 8.4 /
PyTorch for YOLOv8.
