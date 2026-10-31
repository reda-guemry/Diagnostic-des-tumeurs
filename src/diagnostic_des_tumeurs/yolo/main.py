"""Run the full YOLOv8 detection workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from diagnostic_des_tumeurs.config.hyperparameters import (
    YOLO_EXPERIMENT,
    YoloConfig,
)
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import json_safe
from diagnostic_des_tumeurs.yolo.functions.dataset import DatasetReport, prepare_dataset
from diagnostic_des_tumeurs.yolo.functions.evaluation import (
    EvaluationResult,
    evaluate,
)
from diagnostic_des_tumeurs.yolo.functions.inspection import (
    ManifestReport,
    build_manifest,
)
from diagnostic_des_tumeurs.yolo.functions.tracking import (
    log_evaluation,
    log_training,
    register_best,
    save_comparison,
)
from diagnostic_des_tumeurs.yolo.functions.training import TrainingResult, train


@dataclass
class YoloSummary:
    inspection: ManifestReport
    dataset: DatasetReport
    training: TrainingResult
    evaluation: EvaluationResult
    training_run_id: str | None = None
    evaluation_run_id: str | None = None
    comparison: pd.DataFrame | None = None
    registered_model: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.inspection.summary(),
            "dataset": self.dataset.summary(),
            "training": {
                "weights": self.training.weights.as_posix(),
                "run_dir": self.training.run_dir.as_posix(),
                "device": self.training.device_name,
                "epochs": self.training.summary["epochs_completed"],
                "imgsz": self.training.train_args["imgsz"],
                "batch": self.training.train_args["batch"],
                "validation_metrics": self.training.validation_metrics,
                "run_id": self.training_run_id,
            },
            "evaluation": {
                "test_metrics": self.evaluation.metrics,
                "verdict": self.evaluation.verdict,
                "mAP50_gap": self.evaluation.summary.get("mAP50_gap"),
                "run_id": self.evaluation_run_id,
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
    config: YoloConfig | None = None,
    experiment: str = YOLO_EXPERIMENT,
    register: bool = True,
    tracking: bool = True,
) -> YoloSummary:
    """Run inspection, dataset preparation, training, evaluation and tracking."""
    config = config or YoloConfig()
    paths = Paths.from_root(root)
    paths.ensure_output_dirs()

    print("[yolo] 1/4 — scanning the archive and validating annotations")
    inspection = build_manifest(paths, seed=config.seed)
    totals = inspection.totals
    print(
        f"[yolo] paired images: Train={totals['Train']} Val={totals['Val']}"
        f" | invalid annotations={len(inspection.invalid_df)}"
        f" | mixed-class images={int((~inspection.consistency_df['consistent']).sum())}"
    )

    print("[yolo] 2/4 — building data/yolo and data.yaml")
    dataset = prepare_dataset(paths, seed=config.seed)
    print(
        "[yolo] images per split:"
        f" {dataset.summary()['images_per_split']}"
        f" | unpaired={dataset.summary()['unpaired']}"
    )

    print("[yolo] 3/4 — training YOLOv8n")
    training = train(paths, config)
    training_run_id = (
        log_training(training, config, paths, experiment) if tracking else None
    )
    print(
        "[yolo] validation mAP50-95:"
        f" {training.validation_metrics.get('metrics/mAP50-95(B)', float('nan')):.4f}"
    )

    print("[yolo] 4/4 — evaluating on the test split")
    evaluation = evaluate(paths, config)
    evaluation_run_id = (
        log_evaluation(evaluation, paths, experiment) if tracking else None
    )
    print(
        "[yolo] test mAP50-95:"
        f" {evaluation.metrics.get('metrics/mAP50-95(B)', float('nan')):.4f}"
        f" | {evaluation.verdict}"
    )

    comparison = None
    registered_model = None
    if tracking:
        comparison = save_comparison(experiment, paths)
        if register:
            registered_model = register_best(experiment, paths)
            print(
                "[yolo] registered model:"
                f" {registered_model['name']} v{registered_model['version']}"
            )

    return YoloSummary(
        inspection=inspection,
        dataset=dataset,
        training=training,
        evaluation=evaluation,
        training_run_id=training_run_id,
        evaluation_run_id=evaluation_run_id,
        comparison=comparison,
        registered_model=registered_model,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the YOLOv8 brain-tumour detection workflow."
    )
    parser.add_argument("--root", default=None, help="Project root (default: cwd).")
    parser.add_argument("--experiment", default=YOLO_EXPERIMENT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--no-tracking", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "fraction": args.fraction,
            "device": args.device,
            "seed": args.seed,
        }.items()
        if value is not None
    }
    config = replace(YoloConfig(), **overrides)

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
