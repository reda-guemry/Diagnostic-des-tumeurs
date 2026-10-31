"""Run the full InceptionV3 classification workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from diagnostic_des_tumeurs.config.hyperparameters import (
    INCEPTION_EXPERIMENT,
    InceptionConfig,
)
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import json_safe
from diagnostic_des_tumeurs.inception.functions.data import (
    ClassificationData,
    build_splits,
    class_balance,
)
from diagnostic_des_tumeurs.inception.functions.preprocessing import (
    build_balanced_manifest,
    build_datasets,
)
from diagnostic_des_tumeurs.inception.functions.tracking import (
    log_stage,
    register_best,
    save_comparison,
)
from diagnostic_des_tumeurs.inception.functions.training import (
    TrainingResult,
    fine_tune,
    train_frozen,
)


@dataclass
class InceptionSummary:
    data: ClassificationData
    frozen: TrainingResult
    fine_tuned: TrainingResult
    frozen_run_id: str | None = None
    fine_tune_run_id: str | None = None
    comparison: pd.DataFrame | None = None
    registered_model: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": self.data.classes,
            "class_balance": class_balance(self.data.metadata, self.data.classes),
            "splits": {
                split: len(self.data.splits[split]["paths"])
                for split in ("train", "val", "test")
            },
            "frozen": {
                "model": self.frozen.model_path.as_posix(),
                "epochs": len(self.frozen.history.get("loss", [])),
                "metrics": self.frozen.evaluation.metrics,
                "run_id": self.frozen_run_id,
            },
            "fine_tuned": {
                "model": self.fine_tuned.model_path.as_posix(),
                "epochs": len(self.fine_tuned.history.get("loss", [])),
                "metrics": self.fine_tuned.evaluation.metrics,
                "run_id": self.fine_tune_run_id,
                "class_weight": self.fine_tuned.class_weight,
                "trainable_backbone_layers": (
                    self.fine_tuned.trainable_backbone_layers
                ),
            },
            "comparison": (
                self.comparison.to_dict(orient="records")
                if self.comparison is not None
                else None
            ),
            "registered_model": self.registered_model,
        }


def run(
    root: str | None = None,
    config: InceptionConfig | None = None,
    experiment: str = INCEPTION_EXPERIMENT,
    register: bool = True,
    tracking: bool = True,
) -> InceptionSummary:
    """Run data preparation, frozen training, fine-tuning and evaluation."""
    config = config or InceptionConfig()
    paths = Paths.from_root(root)
    paths.ensure_output_dirs()

    print("[inception] 1/4 — scanning data and writing the stratified split")
    data = build_splits(
        paths,
        seed=config.seed,
        image_size=(config.image_size, config.image_size),
    )
    print(
        "[inception] train/val/test:"
        f" {len(data.splits['train']['paths'])}"
        f" / {len(data.splits['val']['paths'])}"
        f" / {len(data.splits['test']['paths'])}"
    )

    print("[inception] 2/4 — balancing the training split")
    balanced = build_balanced_manifest(paths, data.splits, seed=config.seed)
    frozen_datasets = build_datasets(
        paths, config, data.splits, balanced=balanced
    )

    print("[inception] 3/4 — training the frozen backbone + head")
    frozen = train_frozen(paths, config, frozen_datasets, data)
    frozen_run_id = (
        log_stage(
            frozen, config, data, frozen_datasets, paths, experiment
        )
        if tracking
        else None
    )
    print(
        f"[inception] frozen test accuracy: {frozen.evaluation.accuracy:.4f}"
    )

    print("[inception] 4/4 — fine-tuning the last InceptionV3 layers")
    fine_datasets = build_datasets(paths, config, data.splits)
    fine_tuned = fine_tune(paths, config, fine_datasets, data)
    fine_tune_run_id = (
        log_stage(
            fine_tuned, config, data, fine_datasets, paths, experiment
        )
        if tracking
        else None
    )
    print(
        f"[inception] fine-tuned test accuracy: {fine_tuned.evaluation.accuracy:.4f}"
    )

    comparison = None
    registered_model = None
    if tracking:
        comparison = save_comparison(experiment, paths)
        if register:
            registered_model = register_best(experiment, paths)
            print(
                "[inception] registered model:"
                f" {registered_model['name']} v{registered_model['version']}"
            )

    return InceptionSummary(
        data=data,
        frozen=frozen,
        fine_tuned=fine_tuned,
        frozen_run_id=frozen_run_id,
        fine_tune_run_id=fine_tune_run_id,
        comparison=comparison,
        registered_model=registered_model,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the InceptionV3 brain-tumour classification workflow."
    )
    parser.add_argument("--root", default=None, help="Project root (default: cwd).")
    parser.add_argument("--experiment", default=INCEPTION_EXPERIMENT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--fine-tune-epochs", type=int, default=None)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=None)
    parser.add_argument("--unfreeze-last-n", type=int, default=None)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--no-tracking", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    base = InceptionConfig()
    overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "fine_tune_epochs": args.fine_tune_epochs,
            "fine_tune_learning_rate": args.fine_tune_learning_rate,
            "unfreeze_last_n": args.unfreeze_last_n,
        }.items()
        if value is not None
    }
    config = replace(base, **overrides)

    summary = run(
        root=args.root,
        config=config,
        experiment=args.experiment,
        register=not args.no_register,
        tracking=not args.no_tracking,
    )
    print(json.dumps(json_safe(summary.to_dict()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
