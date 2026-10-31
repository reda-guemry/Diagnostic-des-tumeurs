"""Filesystem layout shared by the classification and detection pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """All input and output locations, derived from the project root."""

    root: Path

    # Classification (InceptionV3)
    data_dir: Path
    processed_dir: Path
    splits_path: Path
    balanced_path: Path
    image_metadata_path: Path
    models_dir: Path
    inception_transfer: Path
    inception_finetuned: Path
    inception_reports_dir: Path
    inception_plots_dir: Path
    inception_frozen_history: Path
    inception_finetuned_history: Path
    inception_comparison_csv: Path
    inception_comparison_json: Path
    inception_registered_model: Path

    # Detection (YOLOv8)
    archive_dir: Path
    yolo_manifest_path: Path
    yolo_overview_dir: Path
    yolo_dataset_dir: Path
    yolo_data_yaml: Path
    yolo_models_dir: Path
    yolo_runs_dir: Path
    yolo_pretrained: Path
    yolo_saved_model: Path
    yolo_training_summary: Path
    yolo_training_args: Path
    yolo_training_results: Path
    yolo_training_plots: Path
    yolo_evaluation_dir: Path
    yolo_comparison_csv: Path
    yolo_comparison_json: Path
    yolo_registered_model: Path

    # MLflow
    mlflow_db: Path
    mlflow_artifacts: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "Paths":
        root_path = Path(root).expanduser() if root is not None else Path.cwd()
        root_path = root_path.resolve()
        if root_path.name == "notebooks":
            root_path = root_path.parent

        data_dir = root_path / "data"
        processed_dir = data_dir / "processed"
        models_dir = root_path / "models"
        inception_reports_dir = models_dir / "inception_reports"
        yolo_models_dir = models_dir / "yolo"
        yolo_dataset_dir = data_dir / "yolo"

        return cls(
            root=root_path,
            data_dir=data_dir / "Data",
            processed_dir=processed_dir,
            splits_path=processed_dir / "splits.json",
            balanced_path=processed_dir / "train_balanced.json",
            image_metadata_path=processed_dir / "image_metadata.csv",
            models_dir=models_dir,
            inception_transfer=models_dir / "inceptionv3_transfer.keras",
            inception_finetuned=models_dir / "inceptionv3_finetuned.keras",
            inception_reports_dir=inception_reports_dir,
            inception_plots_dir=inception_reports_dir / "plots",
            inception_frozen_history=(
                inception_reports_dir / "history_frozen.json"
            ),
            inception_finetuned_history=(
                inception_reports_dir / "history_finetuned.json"
            ),
            inception_comparison_csv=inception_reports_dir / "comparison.csv",
            inception_comparison_json=inception_reports_dir / "comparison.json",
            inception_registered_model=inception_reports_dir / "registered_model.json",
            archive_dir=data_dir / "archive",
            yolo_manifest_path=processed_dir / "yolo_manifest.json",
            yolo_overview_dir=processed_dir / "yolo_overview",
            yolo_dataset_dir=yolo_dataset_dir,
            yolo_data_yaml=yolo_dataset_dir / "data.yaml",
            yolo_models_dir=yolo_models_dir,
            yolo_runs_dir=yolo_models_dir / "runs",
            yolo_pretrained=yolo_models_dir / "yolov8n.pt",
            yolo_saved_model=yolo_models_dir / "brain_tumor_yolov8n.pt",
            yolo_training_summary=yolo_models_dir / "training_summary.json",
            yolo_training_args=yolo_models_dir / "training_args.yaml",
            yolo_training_results=yolo_models_dir / "training_results.csv",
            yolo_training_plots=yolo_models_dir / "training_plots",
            yolo_evaluation_dir=yolo_models_dir / "evaluation",
            yolo_comparison_csv=yolo_models_dir / "comparison.csv",
            yolo_comparison_json=yolo_models_dir / "comparison.json",
            yolo_registered_model=yolo_models_dir / "registered_model.json",
            mlflow_db=root_path / "mlflow.db",
            mlflow_artifacts=root_path / "mlartifacts",
        )

    def ensure_output_dirs(self) -> None:
        for directory in (
            self.processed_dir,
            self.models_dir,
            self.inception_reports_dir,
            self.inception_plots_dir,
            self.yolo_overview_dir,
            self.yolo_models_dir,
            self.yolo_runs_dir,
            self.yolo_evaluation_dir,
            self.yolo_training_plots,
        ):
            directory.mkdir(parents=True, exist_ok=True)
