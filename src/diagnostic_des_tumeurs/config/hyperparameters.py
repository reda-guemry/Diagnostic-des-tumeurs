"""Only the knobs that are useful to change.

Paths are derived automatically in :mod:`diagnostic_des_tumeurs.config.paths`;
everything else has a sensible default so a fresh checkout can run without
editing any configuration file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InceptionConfig:
    """Hyperparameters of the frozen-backbone + fine-tuning workflow."""

    image_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    fine_tune_epochs: int = 20
    fine_tune_learning_rate: float = 1e-5
    unfreeze_last_n: int = 30
    dropout: float = 0.5
    dense_units: int = 256
    patience: int = 5
    seed: int = 42


@dataclass(frozen=True)
class YoloConfig:
    """Hyperparameters of the YOLOv8 workflow.

    ``None`` values use the defaults for the current device
    (GPU: 30 epochs / 640 px / batch 32, CPU: 10 epochs / 320 px / batch 8).
    """

    epochs: int | None = None
    imgsz: int | None = None
    batch: int | None = None
    patience: int | None = None
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    seed: int = 42
    device: str | int | None = None
    fraction: float = 1.0


YOLO_CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
YOLO_CLASS_TO_ID = {name: index for index, name in enumerate(YOLO_CLASS_NAMES)}
YOLO_ID_TO_CLASS = {index: name for name, index in YOLO_CLASS_TO_ID.items()}
YOLO_SPLITS = ("Train", "Val")
YOLO_TARGET_SPLITS = ("train", "valid", "test")
YOLO_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

INCEPTION_EXPERIMENT = "inception-brain-tumor"
YOLO_EXPERIMENT = "yolo-brain-tumor"
INCEPTION_REGISTERED_MODEL = "inception-brain-tumor"
YOLO_REGISTERED_MODEL = "brain-tumor-yolov8"
DEFAULT_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
