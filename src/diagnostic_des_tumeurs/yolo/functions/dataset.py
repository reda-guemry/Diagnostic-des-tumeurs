"""Prepare the Ultralytics dataset from the validated manifest."""

from __future__ import annotations

import collections
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from diagnostic_des_tumeurs.config.hyperparameters import YOLO_TARGET_SPLITS
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import load_json, save_yaml
from diagnostic_des_tumeurs.yolo.helpers.visualization import save_dataset_samples


@dataclass
class DatasetReport:
    assignments: dict[str, list[dict[str, Any]]]
    verification: pd.DataFrame
    data_config: dict[str, Any]
    copied: dict[str, int]
    skipped: list[dict[str, str]]
    figure_paths: dict[str, Path] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "images_per_split": {
                split: int(self.verification.loc[
                    self.verification["split"] == split, "images"
                ].iloc[0])
                for split in YOLO_TARGET_SPLITS
            },
            "unpaired": int(self.verification["unpaired"].sum()),
            "skipped_files": len(self.skipped),
            "data_yaml": self.data_config.get("_path", ""),
            "figures": {
                key: value.as_posix() for key, value in self.figure_paths.items()
            },
        }


def prepare_dataset(
    paths: Paths,
    seed: int = 42,
    test_size: float = 0.5,
) -> DatasetReport:
    """Copy the archive into ``data/yolo`` and generate ``data.yaml``.

    The archive ``Train`` folder becomes ``train``.  The archive ``Val`` folder
    is split 50/50, stratified by class, into ``valid`` and ``test``.
    """
    if not paths.yolo_manifest_path.exists():
        raise FileNotFoundError(
            f"{paths.yolo_manifest_path} not found. Run the data overview step first."
        )
    manifest = load_json(paths.yolo_manifest_path)
    classes = manifest["classes"]
    class_to_id = {name: int(index) for name, index in manifest["class_to_id"].items()}
    id_to_class = {int(key): value for key, value in manifest["id_to_class"].items()}
    assert id_to_class == {index: name for index, name in enumerate(classes)}

    def usable(split: str) -> list[dict[str, Any]]:
        return [record for record in manifest["splits"][split] if record["valid"]]

    train_records = usable("Train")
    val_source = usable("Val")

    if val_source:
        valid_records, test_records = train_test_split(
            val_source,
            test_size=test_size,
            random_state=seed,
            stratify=[record["folder_class"] for record in val_source],
        )
    else:
        valid_records, test_records = [], []

    assignments = {
        "train": train_records,
        "valid": valid_records,
        "test": test_records,
    }

    if paths.yolo_dataset_dir.exists():
        shutil.rmtree(paths.yolo_dataset_dir)
    for split in YOLO_TARGET_SPLITS:
        (paths.yolo_dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (paths.yolo_dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    copied = collections.Counter()
    skipped: list[dict[str, str]] = []
    seen_stems = {split: set() for split in YOLO_TARGET_SPLITS}

    for split, records in assignments.items():
        for record in records:
            source_image = paths.root / record["image"]
            source_label = paths.root / record["label"]
            if not source_image.exists() or not source_label.exists():
                skipped.append(
                    {
                        "split": split,
                        "image": record["image"],
                        "label": record["label"],
                    }
                )
                continue
            if record["stem"] in seen_stems[split]:
                raise FileExistsError(
                    f"Duplicate file stem in {split}: {record['stem']}"
                )
            seen_stems[split].add(record["stem"])
            shutil.copy2(
                source_image,
                paths.yolo_dataset_dir / "images" / split / source_image.name,
            )
            shutil.copy2(
                source_label,
                paths.yolo_dataset_dir / "labels" / split / source_label.name,
            )
            copied[split] += 1

    verification_rows = []
    for split in YOLO_TARGET_SPLITS:
        images = {
            path.stem
            for path in (paths.yolo_dataset_dir / "images" / split).iterdir()
            if path.is_file()
        }
        labels = {
            path.stem
            for path in (paths.yolo_dataset_dir / "labels" / split).iterdir()
            if path.is_file()
        }
        verification_rows.append(
            {
                "split": split,
                "images": len(images),
                "labels": len(labels),
                "unpaired": len(images.symmetric_difference(labels)),
            }
        )
    verification = pd.DataFrame(verification_rows)
    if not (verification["unpaired"] == 0).all():
        raise RuntimeError("Some images and labels are not paired")
    if verification["images"].sum() != sum(copied.values()):
        raise RuntimeError("Copied image count does not match the verification")

    data_config: dict[str, Any] = {
        "path": str(paths.yolo_dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": len(classes),
        "names": classes,
    }
    save_yaml(paths.yolo_data_yaml, data_config)

    max_class_id = max(
        (
            class_id
            for split in manifest["splits"]
            for record in manifest["splits"][split]
            for class_id in record["label_classes"]
        ),
        default=-1,
    )
    if max_class_id >= data_config["nc"]:
        raise ValueError("Manifest contains a class id outside the configured names")

    figure = paths.yolo_overview_dir / "dataset_samples.png"
    save_dataset_samples(
        paths.yolo_dataset_dir, YOLO_TARGET_SPLITS, classes, figure
    )

    return DatasetReport(
        assignments=assignments,
        verification=verification,
        data_config={**data_config, "_path": paths.yolo_data_yaml.as_posix()},
        copied=dict(copied),
        skipped=skipped,
        figure_paths={"dataset_samples": figure},
    )
