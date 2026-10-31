"""Scan and validate the YOLO archive, then write a manifest."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from diagnostic_des_tumeurs.config.hyperparameters import (
    YOLO_CLASS_NAMES,
    YOLO_CLASS_TO_ID,
    YOLO_ID_TO_CLASS,
    YOLO_IMAGE_EXTENSIONS,
    YOLO_SPLITS,
)
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import relative_path, save_json
from diagnostic_des_tumeurs.yolo.helpers.annotations import (
    label_classes,
    parse_yolo_label,
    relative,
)
from diagnostic_des_tumeurs.yolo.helpers.visualization import (
    save_class_distribution,
    save_resolution_plot,
    save_sample_grid,
)


@dataclass
class ManifestReport:
    manifest: dict[str, Any]
    records: dict[str, list[dict[str, Any]]]
    discarded: dict[str, list[dict[str, Any]]]
    paired_df: pd.DataFrame
    invalid_df: pd.DataFrame
    consistency_df: pd.DataFrame
    image_pivot: pd.DataFrame
    instance_pivot: pd.DataFrame
    checks_df: pd.DataFrame
    unreadable_df: pd.DataFrame
    size_counts: pd.Series
    figure_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def totals(self) -> dict[str, int]:
        return {split: len(self.records[split]) for split in YOLO_SPLITS}

    def summary(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest.get("_path", "")),
            "paired_per_split": self.totals,
            "invalid_annotations": len(self.invalid_df),
            "mixed_class_images": int((~self.consistency_df["consistent"]).sum()),
            "missing_labels": len(self.discarded["missing_labels"]),
            "missing_images": len(self.discarded["missing_images"]),
            "unreadable_images": len(self.unreadable_df),
            "figures": {
                key: value.as_posix() for key, value in self.figure_paths.items()
            },
        }


def _collect_archive(
    archive_dir: Path, root: Path
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    if not archive_dir.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive_dir}\n"
            "Expected data/archive/Train/<class>/images and .../labels"
        )

    records: dict[str, list[dict[str, Any]]] = {split: [] for split in YOLO_SPLITS}
    discarded: dict[str, list[dict[str, Any]]] = {
        "missing_labels": [],
        "missing_images": [],
    }
    folder_counts: list[dict[str, Any]] = []

    for split in YOLO_SPLITS:
        split_dir = archive_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"{split_dir} not found")

        for folder in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if folder.name not in YOLO_CLASS_TO_ID:
                raise ValueError(
                    f"Unknown class folder {folder.name!r} in {split}. "
                    f"Expected one of {YOLO_CLASS_NAMES}."
                )
            images_dir = folder / "images"
            labels_dir = folder / "labels"
            if not images_dir.exists() or not labels_dir.exists():
                continue

            images = {
                path.stem: path
                for path in sorted(images_dir.iterdir())
                if path.suffix.lower() in YOLO_IMAGE_EXTENSIONS
            }
            labels = {
                path.stem: path
                for path in sorted(labels_dir.iterdir())
                if path.suffix.lower() == ".txt"
            }

            for stem, image_path in images.items():
                label_path = labels.get(stem)
                if label_path is None:
                    discarded["missing_labels"].append(
                        {
                            "path": relative_path(image_path, root),
                            "split": split,
                            "folder": folder.name,
                        }
                    )
                    continue
                records[split].append(
                    {
                        "image": relative_path(image_path, root),
                        "label": relative_path(label_path, root),
                        "split": split,
                        "folder": folder.name,
                        "folder_class": YOLO_CLASS_TO_ID[folder.name],
                        "stem": stem,
                    }
                )

            for stem, label_path in labels.items():
                if stem not in images:
                    discarded["missing_images"].append(
                        {
                            "path": relative_path(label_path, root),
                            "split": split,
                            "folder": folder.name,
                        }
                    )

            folder_counts.append(
                {
                    "split": split,
                    "folder": folder.name,
                    "images": len(images),
                    "labels": len(labels),
                    "paired": sum(1 for stem in images if stem in labels),
                }
            )
    return records, discarded, folder_counts


def build_manifest(paths: Paths, seed: int = 42) -> ManifestReport:
    """Replicate the overview notebook and persist the validated manifest."""
    paths.ensure_output_dirs()
    records, discarded, folder_counts = _collect_archive(paths.archive_dir, paths.root)
    paired_df = pd.DataFrame(folder_counts)

    invalid_rows: list[dict[str, Any]] = []
    for split in YOLO_SPLITS:
        for record in records[split]:
            boxes, errors = parse_yolo_label(
                paths.root / record["label"], num_classes=len(YOLO_CLASS_NAMES)
            )
            record["boxes"] = boxes
            record["label_classes"] = label_classes(boxes)
            record["consistent"] = set(record["label_classes"]) <= {
                record["folder_class"]
            }
            record["valid"] = not errors
            record["errors"] = errors
            if errors:
                invalid_rows.append(
                    {
                        "label": record["label"],
                        "split": split,
                        "errors": "; ".join(errors),
                    }
                )
    invalid_df = pd.DataFrame(invalid_rows)

    consistency_df = pd.DataFrame(
        [
            {
                "split": split,
                "folder": record["folder"],
                "folder_class": record["folder_class"],
                "label_classes": ",".join(
                    str(value) for value in record["label_classes"]
                )
                or "(empty)",
                "consistent": record["consistent"],
            }
            for split in YOLO_SPLITS
            for record in records[split]
        ]
    )

    if not paired_df.empty:
        image_pivot = (
            paired_df.pivot(index="folder", columns="split", values="paired")
            .reindex(YOLO_CLASS_NAMES)
            .fillna(0)
            .astype(int)
        )
    else:
        image_pivot = pd.DataFrame(0, index=YOLO_CLASS_NAMES, columns=list(YOLO_SPLITS))

    instance_counts = Counter()
    for split in YOLO_SPLITS:
        for record in records[split]:
            for class_id in record["label_classes"]:
                instance_counts[(split, class_id)] += 1
    instance_pivot = pd.DataFrame(
        {
            "class": [
                YOLO_ID_TO_CLASS[index]
                for index in range(len(YOLO_CLASS_NAMES))
            ],
            **{
                split: [
                    instance_counts[(split, class_id)]
                    for class_id in range(len(YOLO_CLASS_NAMES))
                ]
                for split in YOLO_SPLITS
            },
        }
    ).set_index("class")

    image_checks: list[dict[str, Any]] = []
    for split in YOLO_SPLITS:
        for record in records[split]:
            image = cv2.imread(str(paths.root / record["image"]))
            if image is None:
                image_checks.append(
                    {
                        "image": record["image"],
                        "readable": False,
                        "width": None,
                        "height": None,
                        "channels": None,
                    }
                )
                continue
            height, width = image.shape[:2]
            channels = image.shape[2] if image.ndim == 3 else 1
            image_checks.append(
                {
                    "image": record["image"],
                    "readable": True,
                    "width": width,
                    "height": height,
                    "channels": channels,
                }
            )
    checks_df = pd.DataFrame(image_checks)
    unreadable_df = checks_df.loc[~checks_df["readable"]]
    size_counts = (
        checks_df.dropna(subset=["width", "height"])
        .groupby(["width", "height"])
        .size()
        .sort_values(ascending=False)
    )

    figures = {
        "class_distribution": paths.yolo_overview_dir / "class_distribution.png",
        "image_resolutions": paths.yolo_overview_dir / "image_resolutions.png",
        "samples": paths.yolo_overview_dir / "samples.png",
    }
    save_class_distribution(image_pivot, instance_pivot, figures["class_distribution"])
    save_resolution_plot(size_counts, figures["image_resolutions"])
    train_records = records["Train"]
    save_sample_grid(
        train_records,
        paths.root,
        YOLO_CLASS_NAMES,
        figures["samples"],
        seed,
    )

    manifest = {
        "created_by": "diagnostic_des_tumeurs.yolo.functions.inspection",
        "source_dir": relative(paths.archive_dir, paths.root),
        "format": "YOLO detection: class x_center y_center width height (normalized)",
        "classes": YOLO_CLASS_NAMES,
        "class_to_id": YOLO_CLASS_TO_ID,
        "id_to_class": {str(key): value for key, value in YOLO_ID_TO_CLASS.items()},
        "seed": seed,
        "splits": {
            split: [
                {
                    "image": record["image"],
                    "label": record["label"],
                    "split": record["split"],
                    "folder": record["folder"],
                    "folder_class": int(record["folder_class"]),
                    "stem": record["stem"],
                    "boxes": [
                        [int(class_id), float(x), float(y), float(width), float(height)]
                        for class_id, x, y, width, height in record["boxes"]
                    ],
                    "label_classes": [int(value) for value in record["label_classes"]],
                    "consistent": bool(record["consistent"]),
                    "valid": bool(record["valid"]),
                    "errors": record["errors"],
                }
                for record in records[split]
            ]
            for split in YOLO_SPLITS
        },
        "discarded": discarded,
        "counts": {
            "paired_per_split": {split: len(records[split]) for split in YOLO_SPLITS},
            "invalid_annotations": len(invalid_df),
            "missing_labels": len(discarded["missing_labels"]),
            "missing_images": len(discarded["missing_images"]),
            "images_per_folder": image_pivot.to_dict(),
            "boxes_per_class": instance_pivot.to_dict(),
        },
        "image_checks": {
            "unreadable": unreadable_df["image"].tolist(),
            "most_common_sizes": {
                f"{int(width)}x{int(height)}": int(count)
                for (width, height), count in size_counts.head(10).items()
            },
        },
    }
    save_json(paths.yolo_manifest_path, manifest)
    manifest["_path"] = paths.yolo_manifest_path.as_posix()

    return ManifestReport(
        manifest=manifest,
        records=records,
        discarded=discarded,
        paired_df=paired_df,
        invalid_df=invalid_df,
        consistency_df=consistency_df,
        image_pivot=image_pivot,
        instance_pivot=instance_pivot,
        checks_df=checks_df,
        unreadable_df=unreadable_df,
        size_counts=size_counts,
        figure_paths=figures,
    )
