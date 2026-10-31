"""Image loading, augmentation, balancing and ``tf.data`` pipelines."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

from diagnostic_des_tumeurs.config.hyperparameters import InceptionConfig
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import load_json, relative_path, save_json


@dataclass(frozen=True)
class ClassificationDatasets:
    """The three batched datasets plus the paths/labels they were built from."""

    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    train_paths: list[str]
    train_labels: np.ndarray
    val_paths: list[str]
    val_labels: np.ndarray
    test_paths: list[str]
    test_labels: np.ndarray
    classes: list[str]
    id_to_class: dict[int, str]
    image_size: tuple[int, int]
    batch_size: int


def absolute_path(root: Path, relative: str) -> str:
    return str((root / relative).resolve())


def load_splits(paths: Paths) -> dict[str, Any]:
    if not paths.splits_path.exists():
        raise FileNotFoundError(
            f"{paths.splits_path} not found. Run the data overview step first."
        )
    return load_json(paths.splits_path)


def build_balanced_manifest(
    paths: Paths, splits: dict[str, Any], seed: int = 42
) -> dict[str, Any]:
    """Oversample minority classes so every class has the same train count.

    No augmented image is written: the duplicates are paths and the
    augmentation happens on the fly in the ``tf.data`` pipeline.
    """
    train_paths = [absolute_path(paths.root, path) for path in splits["train"]["paths"]]
    train_labels = list(splits["train"]["labels"])
    counts = Counter(train_labels)
    target = max(counts.values())
    rng = random.Random(seed)

    balanced_paths: list[str] = []
    balanced_labels: list[int] = []
    for class_id in range(len(splits["classes"])):
        class_paths = [
            path for path, label in zip(train_paths, train_labels) if label == class_id
        ]
        balanced_paths.extend(class_paths)
        balanced_labels.extend([class_id] * len(class_paths))

        missing = target - len(class_paths)
        if missing > 0:
            balanced_paths.extend(rng.choices(class_paths, k=missing))
            balanced_labels.extend([class_id] * missing)

    balanced = {
        "classes": splits["classes"],
        "class_to_id": splits["class_to_id"],
        "id_to_class": splits["id_to_class"],
        "img_size": splits.get("img_size", [224, 224]),
        "seed": seed,
        "paths": [
            relative_path(path, paths.root) for path in balanced_paths
        ],
        "labels": [int(label) for label in balanced_labels],
    }
    save_json(paths.balanced_path, balanced)
    return balanced


def load_image(
    path: str | Path,
    image_size: tuple[int, int],
    label: int | None = None,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Load one image as an RGB ``float32`` array in the ``[0, 255]`` range."""
    if isinstance(path, tf.Tensor):
        path = path.numpy().decode("utf-8")

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image.astype(np.float32)

    if label is None:
        return image
    return image, label


def apply_blur(image: np.ndarray, label: int) -> tuple[np.ndarray, int]:
    return cv2.GaussianBlur(image, (5, 5), 0), label


def apply_flip(image: np.ndarray, label: int) -> tuple[np.ndarray, int]:
    return cv2.flip(image, 1), label


def apply_noise(image: np.ndarray, label: int) -> tuple[np.ndarray, int]:
    noise = np.random.normal(loc=0, scale=10, size=image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.float32)
    return noisy, label


def augment_data(image: np.ndarray, label: int) -> tuple[np.ndarray, int]:
    """Apply one random augmentation (blur, flip or Gaussian noise)."""
    transformations = [apply_blur, apply_flip, apply_noise]
    return random.choice(transformations)(image, label)


def _decode_path(value: Any) -> str:
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def _make_loader(image_size: tuple[int, int]):
    def load_image_tf(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        def _load(path_tensor: Any) -> np.ndarray:
            return load_image(_decode_path(path_tensor), image_size)

        image = tf.numpy_function(_load, [path], tf.float32)
        image.set_shape((image_size[1], image_size[0], 3))
        return image, label

    return load_image_tf


def _augment_frozen(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """One random blur/flip/noise transformation (notebook 02/03)."""
    def _augment(image_np: Any, label_np: Any) -> np.ndarray:
        augmented, _ = augment_data(image_np, label_np)
        return augmented.astype(np.float32)

    augmented = tf.numpy_function(_augment, [image, label], tf.float32)
    augmented.set_shape(image.shape)
    return augmented, label


def _augment_fine_tune(
    image: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Random horizontal flip plus Gaussian noise (notebook 04)."""
    image = tf.image.random_flip_left_right(image)
    noise = tf.random.normal(
        shape=tf.shape(image), mean=0.0, stddev=10.0, dtype=tf.float32
    )
    image = tf.clip_by_value(image + noise, 0.0, 255.0)
    return image, label


def make_dataset(
    paths: list[str],
    labels: list[int] | np.ndarray,
    config: InceptionConfig,
    training: bool = False,
    augmentation: str = "frozen",
) -> tf.data.Dataset:
    image_size = (config.image_size, config.image_size)
    dataset = tf.data.Dataset.from_tensor_slices(
        (list(paths), np.asarray(labels, dtype=np.int32))
    )
    if training:
        dataset = dataset.shuffle(
            len(paths), seed=config.seed, reshuffle_each_iteration=True
        )
    dataset = dataset.map(
        _make_loader(image_size), num_parallel_calls=tf.data.AUTOTUNE
    )
    if training:
        if augmentation == "frozen":
            dataset = dataset.map(
                _augment_frozen, num_parallel_calls=tf.data.AUTOTUNE
            )
        elif augmentation == "fine_tune":
            dataset = dataset.map(
                _augment_fine_tune, num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            raise ValueError(f"Unknown augmentation mode: {augmentation!r}")
    return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


def build_datasets(
    paths: Paths,
    config: InceptionConfig,
    splits: dict[str, Any] | None = None,
    balanced: dict[str, Any] | None = None,
) -> ClassificationDatasets:
    """Build train/val/test datasets.

    Pass ``balanced`` for the frozen-backbone stage and omit it to use the
    original training split for fine-tuning, matching the notebooks.  The
    frozen stage uses the one-of-three blur/flip/noise augmentation from
    notebook 02; fine-tuning uses the flip-plus-noise augmentation from
    notebook 04.
    """
    splits = splits or load_splits(paths)
    classes = splits["classes"]
    id_to_class = {int(key): value for key, value in splits["id_to_class"].items()}
    image_size = (config.image_size, config.image_size)

    if balanced is not None:
        train_paths = [
            absolute_path(paths.root, path) for path in balanced["paths"]
        ]
        train_labels = np.asarray(balanced["labels"], dtype=np.int32)
    else:
        train_paths = [
            absolute_path(paths.root, path) for path in splits["train"]["paths"]
        ]
        train_labels = np.asarray(splits["train"]["labels"], dtype=np.int32)

    val_paths = [absolute_path(paths.root, path) for path in splits["val"]["paths"]]
    val_labels = np.asarray(splits["val"]["labels"], dtype=np.int32)
    test_paths = [absolute_path(paths.root, path) for path in splits["test"]["paths"]]
    test_labels = np.asarray(splits["test"]["labels"], dtype=np.int32)

    return ClassificationDatasets(
        train=make_dataset(
            train_paths,
            train_labels,
            config,
            training=True,
            augmentation="frozen" if balanced is not None else "fine_tune",
        ),
        val=make_dataset(val_paths, val_labels, config, training=False),
        test=make_dataset(test_paths, test_labels, config, training=False),
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        classes=classes,
        id_to_class=id_to_class,
        image_size=image_size,
        batch_size=config.batch_size,
    )
