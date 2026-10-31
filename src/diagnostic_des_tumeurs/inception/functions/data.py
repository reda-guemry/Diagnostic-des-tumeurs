"""Dataset discovery, validation and the stratified train/val/test split."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import relative_path, save_json

VALID_EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp")


@dataclass(frozen=True)
class ClassificationData:
    """Everything the other classification steps need to know about the data."""

    classes: list[str]
    class_to_id: dict[str, int]
    id_to_class: dict[int, str]
    metadata: pd.DataFrame
    splits: dict[str, Any]
    figure_paths: dict[str, Path] = field(default_factory=dict)


def _discover_classes(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Expected folders such as data/Data/glioma/*.jpg"
        )
    classes = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError(f"At least two class folders are required in {data_dir}")
    return classes


def _collect_paths(
    data_dir: Path, classes: list[str], class_to_id: dict[str, int]
) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for class_name in classes:
        class_dir = data_dir / class_name
        for file in sorted(class_dir.iterdir()):
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
                paths.append(file)
                labels.append(class_to_id[class_name])
    return paths, labels


def _inspect_image(path: str) -> dict[str, Any]:
    image = cv2.imread(path)
    if image is None:
        return {
            "path": path,
            "readable": False,
            "width": None,
            "height": None,
            "channels": None,
        }
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    return {
        "path": path,
        "readable": True,
        "width": width,
        "height": height,
        "channels": channels,
    }


def _plot_class_distribution(
    class_counts: pd.Series, classes: list[str], path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        hue=class_counts.index,
        palette="viridis",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Images per class")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Number of images")
    for index, value in enumerate(class_counts.values):
        axes[0].text(index, value, str(int(value)), ha="center", va="bottom")

    axes[1].pie(
        class_counts.values,
        labels=class_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("viridis", len(classes)),
    )
    axes[1].set_title("Class distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_resolutions(metadata: pd.DataFrame, path: Path) -> None:
    counts = (
        metadata.groupby(["width", "height"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    fig, axis = plt.subplots(figsize=(10, 4))
    labels = [f"{int(width)}x{int(height)}" for width, height in counts.index]
    sns.barplot(
        x=labels,
        y=counts.values,
        hue=labels,
        palette="mako",
        legend=False,
        ax=axis,
    )
    axis.set_title("Most common image resolutions")
    axis.set_xlabel("Resolution")
    axis.set_ylabel("Number of images")
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_samples(
    metadata: pd.DataFrame,
    classes: list[str],
    image_size: tuple[int, int],
    seed: int,
    path: Path,
    root: Path,
    samples_per_class: int = 4,
) -> None:
    rng = random.Random(seed)
    fig, axes = plt.subplots(
        len(classes), samples_per_class, figsize=(15, 3 * len(classes))
    )
    axes = np.atleast_2d(axes)

    for row, class_name in enumerate(classes):
        candidates = metadata.loc[
            (metadata["class_name"] == class_name) & metadata["readable"], "path"
        ].tolist()
        chosen = rng.sample(candidates, min(samples_per_class, len(candidates)))
        for column in range(samples_per_class):
            axis = axes[row, column]
            if column < len(chosen):
                image = cv2.imread(str((root / chosen[column]).resolve()))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    axis.imshow(cv2.resize(image, image_size))
                    axis.set_title(class_name)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_splits(
    paths: Paths,
    seed: int = 42,
    image_size: tuple[int, int] = (224, 224),
) -> ClassificationData:
    """Scan ``data/Data``, validate images and write the split manifests."""
    paths.ensure_output_dirs()
    classes = _discover_classes(paths.data_dir)
    class_to_id = {name: index for index, name in enumerate(classes)}
    id_to_class = {index: name for name, index in class_to_id.items()}

    data_paths, data_labels = _collect_paths(paths.data_dir, classes, class_to_id)
    if not data_paths:
        raise FileNotFoundError(f"No images found in {paths.data_dir}")

    absolute_df = pd.DataFrame(
        {
            "absolute_path": [str(path) for path in data_paths],
            "label": data_labels,
        }
    )
    absolute_df["class_name"] = absolute_df["label"].map(id_to_class)
    absolute_df["file_size_kb"] = absolute_df["absolute_path"].map(
        lambda value: Path(value).stat().st_size / 1024
    )

    inspected = pd.DataFrame(
        [_inspect_image(value) for value in absolute_df["absolute_path"]]
    ).rename(columns={"path": "absolute_path"})
    absolute_df = absolute_df.merge(inspected, on="absolute_path", how="left")
    metadata = absolute_df[
        [
            "absolute_path",
            "label",
            "class_name",
            "file_size_kb",
            "readable",
            "width",
            "height",
            "channels",
        ]
    ].rename(columns={"absolute_path": "path"})
    metadata.attrs["root"] = str(paths.root)

    class_counts = metadata["class_name"].value_counts().reindex(classes)
    figures = {
        "class_distribution": paths.inception_plots_dir / "class_distribution.png",
        "image_resolutions": paths.inception_plots_dir / "image_resolutions.png",
        "class_samples": paths.inception_plots_dir / "class_samples.png",
    }
    _plot_class_distribution(class_counts, classes, figures["class_distribution"])
    _plot_resolutions(metadata, figures["image_resolutions"])
    _plot_samples(
        metadata,
        classes,
        image_size,
        seed,
        figures["class_samples"],
        paths.root,
    )

    train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
        data_paths,
        data_labels,
        test_size=0.30,
        random_state=seed,
        stratify=data_labels,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        remaining_paths,
        remaining_labels,
        test_size=0.50,
        random_state=seed,
        stratify=remaining_labels,
    )

    splits = {
        "classes": classes,
        "class_to_id": class_to_id,
        "id_to_class": {str(key): value for key, value in id_to_class.items()},
        "img_size": list(image_size),
        "seed": seed,
        "train": {
            "paths": [relative_path(path, paths.root) for path in train_paths],
            "labels": [int(label) for label in train_labels],
        },
        "val": {
            "paths": [relative_path(path, paths.root) for path in val_paths],
            "labels": [int(label) for label in val_labels],
        },
        "test": {
            "paths": [relative_path(path, paths.root) for path in test_paths],
            "labels": [int(label) for label in test_labels],
        },
    }

    save_json(paths.splits_path, splits)
    metadata.to_csv(paths.image_metadata_path, index=False)

    return ClassificationData(
        classes=classes,
        class_to_id=class_to_id,
        id_to_class=id_to_class,
        metadata=metadata,
        splits=splits,
        figure_paths=figures,
    )


def class_balance(metadata: pd.DataFrame, classes: list[str]) -> dict[str, int]:
    counts = Counter(metadata["class_name"])
    return {name: int(counts.get(name, 0)) for name in classes}
