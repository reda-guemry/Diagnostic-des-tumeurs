"""A small MLflow pyfunc wrapper so YOLO weights can be reused from the registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from mlflow.types.schema import ParamSchema, ParamSpec


class YoloPyfuncModel(mlflow.pyfunc.PythonModel):
    """Run Ultralytics predictions on image paths from a DataFrame/list."""

    def __init__(
        self,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> None:
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)

    def load_context(self, context) -> None:
        from ultralytics import YOLO

        self._model = YOLO(context.artifacts["weights"])

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        params = params or {}
        conf = float(params.get("conf", self.conf))
        iou = float(params.get("iou", self.iou))
        imgsz = int(params.get("imgsz", self.imgsz))

        images = _image_paths(model_input)
        results = self._model.predict(
            source=images,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )

        rows = []
        for image, result in zip(images, results):
            detections = []
            if result.boxes is not None:
                for xyxy, confidence, class_id in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                ):
                    detections.append(
                        {
                            "class_id": int(class_id),
                            "class_name": result.names[int(class_id)],
                            "confidence": float(confidence),
                            "xyxy": [float(value) for value in xyxy],
                        }
                    )
            rows.append(
                {
                    "image": image,
                    "count": len(detections),
                    "detections": json.dumps(detections),
                }
            )
        return pd.DataFrame(rows)


def _image_paths(model_input: Any) -> list[str]:
    if isinstance(model_input, pd.DataFrame):
        if "image" in model_input.columns:
            values = model_input["image"].tolist()
        else:
            values = model_input.iloc[:, 0].tolist()
    elif isinstance(model_input, (list, tuple)):
        values = list(model_input)
    elif isinstance(model_input, str):
        values = [model_input]
    else:
        values = list(model_input)
    return [str(value) for value in values]


def yolo_signature() -> ModelSignature:
    return ModelSignature(
        inputs=Schema([ColSpec("string", "image")]),
        outputs=Schema(
            [
                ColSpec("string", "image"),
                ColSpec("long", "count"),
                ColSpec("string", "detections"),
            ]
        ),
        params=ParamSchema(
            [
                ParamSpec("conf", "double", 0.25),
                ParamSpec("iou", "double", 0.5),
                ParamSpec("imgsz", "long", 640),
            ]
        ),
    )


def log_yolo_model(
    weights_path: str | Path,
    imgsz: int,
    conf: float = 0.25,
    iou: float = 0.5,
) -> None:
    """Log the weights plus the pyfunc wrapper under the ``model`` artifact."""
    mlflow.pyfunc.log_model(
        name="model",
        python_model=YoloPyfuncModel(imgsz=imgsz, conf=conf, iou=iou),
        artifacts={"weights": str(Path(weights_path).resolve())},
        signature=yolo_signature(),
        pip_requirements=[
            "ultralytics",
            "torch",
            "pandas",
            "numpy",
            "opencv-python",
            "pyyaml",
        ],
    )
