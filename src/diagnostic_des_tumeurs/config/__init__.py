"""Configuration shared by the two workflows.

Only the hyperparameters that are useful to change live here.  Paths are
derived from the project root, and the YOLO data files are generated from the
archive, so nothing has to be configured by hand.
"""

from diagnostic_des_tumeurs.config.hyperparameters import (
    INCEPTION_EXPERIMENT,
    INCEPTION_REGISTERED_MODEL,
    YOLO_CLASS_NAMES,
    YOLO_CLASS_TO_ID,
    YOLO_EXPERIMENT,
    YOLO_ID_TO_CLASS,
    YOLO_IMAGE_EXTENSIONS,
    YOLO_REGISTERED_MODEL,
    YOLO_SPLITS,
    YOLO_TARGET_SPLITS,
    InceptionConfig,
    YoloConfig,
)
from diagnostic_des_tumeurs.config.paths import Paths

__all__ = [
    "INCEPTION_EXPERIMENT",
    "INCEPTION_REGISTERED_MODEL",
    "InceptionConfig",
    "Paths",
    "YOLO_CLASS_NAMES",
    "YOLO_CLASS_TO_ID",
    "YOLO_EXPERIMENT",
    "YOLO_ID_TO_CLASS",
    "YOLO_IMAGE_EXTENSIONS",
    "YOLO_REGISTERED_MODEL",
    "YOLO_SPLITS",
    "YOLO_TARGET_SPLITS",
    "YoloConfig",
]
