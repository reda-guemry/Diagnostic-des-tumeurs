# Held-out test-set results

These are the measured results of the model artifacts present in the repository
at the time of the last evaluation (**2026-11-05**). The test splits are the ones
produced by the project itself and are never used for training or model
selection.

> **Scope.** This is a research/engineering benchmark, not a clinical
> validation. The figures below must not be used as medical evidence.

## Evaluation setup

| Item | Classification | Detection |
| --- | --- | --- |
| Artifact | `models/inceptionv3_transfer.keras` | `models/yolo/brain_tumor_yolov8n.pt` |
| Test manifest | `data/processed/splits.json` (`test`) | `data/yolo/data.yaml` (`test`) |
| Test size | 1,054 images | 255 images / 272 annotated boxes |
| Input size | 224 × 224 RGB | 640 × 640 |
| Classes | glioma, meningioma, notumor, pituitary | Glioma, Meningioma, No Tumor, Pituitary |
| Device | CPU | CPU |
| Main metrics | accuracy, macro/weighted precision-recall-F1, confusion matrix | P, R, mAP50, mAP50-95 |

Classification preprocessing matches the training/API path: OpenCV BGR→RGB,
resize to 224 × 224, pixels kept in `[0, 255]` because the Keras model rescales
internally. YOLO uses the standard Ultralytics validation protocol
(`conf=0.001`, `iou=0.7`, `imgsz=640`).

### Important note on the classifier

At evaluation time the repository contained the **transfer-stage** checkpoint
`inceptionv3_transfer.keras`, but not a completed
`inceptionv3_finetuned.keras`. The API therefore used the transfer checkpoint
as its fallback. The classification numbers below describe that deployed
checkpoint, not a fine-tuned model. Retrain and re-run this evaluation once the
fine-tuned artifact exists.

## Classification — InceptionV3 transfer checkpoint

| Metric | Value |
| --- | ---: |
| Test accuracy | **92.41 %** |
| Test loss | 0.1900 |
| Macro precision | 92.28 % |
| Macro recall | 92.07 % |
| Macro F1 | **92.12 %** |
| Weighted precision | 92.48 % |
| Weighted recall | 92.41 % |
| Weighted F1 | 92.39 % |

### Per-class results

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| glioma | 243 | 96.02 % | 89.30 % | 92.54 % |
| meningioma | 247 | 85.31 % | 84.62 % | 84.96 % |
| notumor | 300 | 97.00 % | 97.00 % | 97.00 % |
| pituitary | 264 | 90.81 % | 97.35 % | 93.97 % |

### Confusion matrix

Rows are the true class, columns are the predicted class.

| True \ Predicted | glioma | meningioma | notumor | pituitary |
| --- | ---: | ---: | ---: | ---: |
| **glioma** | 217 | 23 | 1 | 2 |
| **meningioma** | 8 | 209 | 8 | 22 |
| **notumor** | 1 | 6 | 291 | 2 |
| **pituitary** | 0 | 7 | 0 | 257 |

The main confusion is between glioma and meningioma, and between meningioma and
pituitary. `notumor` is the strongest class.

## Detection — YOLOv8n

| Metric | Value |
| --- | ---: |
| Precision | 94.67 % |
| Recall | 96.50 % |
| mAP50 | **98.09 %** |
| mAP50-95 | **79.74 %** |

### Per-class results

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Meningioma | 97.64 % | 100.00 % | 99.49 % | 85.44 % |
| No Tumor | 100.00 % | 99.35 % | 99.50 % | 79.25 % |
| Glioma | 87.48 % | 93.24 % | 95.46 % | 77.82 % |
| Pituitary | 93.58 % | 93.40 % | 97.92 % | 76.46 % |

The weakest localization class is Pituitary on the stricter mAP50-95 metric,
while Glioma has the lowest precision. Meningioma is the strongest class at
IoU 0.5 and above.

## Raw results

The exact JSON outputs are kept for traceability:

- [`docs/results/inceptionv3_transfer_test_metrics.json`](results/inceptionv3_transfer_test_metrics.json)
- [`docs/results/yolov8n_test_metrics.json`](results/yolov8n_test_metrics.json)

## Reproducing the evaluation

The canonical pipeline evaluation functions are:

```python
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.yolo.functions.evaluation import evaluate

paths = Paths.from_root()
result = evaluate(paths)
print(result.metrics)
```

For the classifier, load the checkpoint listed in the setup table, resize the
test images to 224 × 224, keep raw `[0, 255]` pixels and compare `argmax` with the
labels in `splits.json["test"]["labels"]`.

The classification evaluation used 1,054 images. The detection evaluation used
255 images and 272 ground-truth boxes from `data/yolo/data.yaml`. Both were run
on CPU; the YOLO checkpoint itself records a 30-epoch, 640 px, batch-32 training
run on GPU 0.
