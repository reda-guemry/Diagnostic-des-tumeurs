"""Small FastAPI inference service for the Streamlit interface."""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from diagnostic_des_tumeurs.config.hyperparameters import InceptionConfig
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import load_json

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _warm_up_detector()
    yield


app = FastAPI(
    title="Brain Tumour Inference API",
    description="Classification InceptionV3 et détection YOLOv8 pour l'interface Streamlit.",
    version="0.1.0",
    lifespan=_lifespan,
)


def _paths() -> Paths:
    return Paths.from_root(os.environ.get("BTD_PROJECT_ROOT"))


def _find_classifier(paths: Paths) -> Path | None:
    for candidate in (paths.inception_finetuned, paths.inception_transfer):
        if candidate.is_file():
            return candidate
    return None


def _find_detector(paths: Paths) -> Path | None:
    candidates = (
        paths.yolo_saved_model,
        paths.yolo_runs_dir / "brain_tumor_yolov8n" / "weights" / "best.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=1)
def _classifier() -> tuple[Any, Path]:
    import keras

    path = _find_classifier(_paths())
    if path is None:
        raise FileNotFoundError(
            "Aucun modèle de classification trouvé dans models/. "
            "Lancez d'abord le pipeline Inception."
        )
    return keras.models.load_model(path, compile=False), path


@lru_cache(maxsize=1)
def _detector() -> tuple[Any, Path]:
    from ultralytics import YOLO

    path = _find_detector(_paths())
    if path is None:
        raise FileNotFoundError(
            "Aucun modèle YOLO entraîné trouvé dans models/yolo/. "
            "Lancez d'abord le pipeline de détection."
        )
    return YOLO(str(path)), path


@lru_cache(maxsize=1)
def _class_names() -> list[str]:
    splits = load_json(_paths().splits_path)
    return list(splits["classes"])


def _warm_up_detector() -> None:
    """Exercise YOLO before TensorFlow is loaded.

    On this stack, running a TensorFlow session before the first PyTorch/YOLO
    inference can abort the process with ``free(): invalid pointer``. A single
    dummy YOLO prediction initializes PyTorch first and avoids that crash.
    """
    if _find_detector(_paths()) is None:
        return
    try:
        model, _ = _detector()
        model.predict(
            source=np.zeros((64, 64, 3), dtype=np.uint8),
            conf=0.99,
            verbose=False,
        )
    except Exception:
        logger.exception("YOLO warm-up failed; /detect will report the error")


def _read_image(file: UploadFile) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file.file.read()))
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Fichier image invalide ou illisible.",
        ) from exc


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "brain-tumour-inference", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    paths = _paths()
    classifier = _find_classifier(paths)
    detector = _find_detector(paths)
    return {
        "status": "ok" if classifier and detector else "degraded",
        "models": {
            "classification": classifier.name if classifier else None,
            "detection": detector.name if detector else None,
        },
    }


@app.post("/classify")
def classify(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
    """Return the predicted class and the softmax score for every class."""
    try:
        model, model_path = _classifier()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    image = _read_image(file)
    size = InceptionConfig().image_size
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)

    probabilities = np.asarray(model.predict(batch, verbose=0))[0]
    try:
        names = _class_names()
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "data/processed/splits.json est introuvable ou invalide. "
                "Lancez d'abord le pipeline Inception."
            ),
        ) from exc
    if len(probabilities) != len(names):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Le modèle retourne {len(probabilities)} classes mais "
                f"{len(names)} noms sont configurés."
            ),
        )

    scores = [
        {"class_name": name, "score": float(score)}
        for name, score in zip(names, probabilities)
    ]
    best_index = int(np.argmax(probabilities))
    return {
        "filename": file.filename,
        "model": model_path.name,
        "predicted_class": names[best_index],
        "confidence": float(probabilities[best_index]),
        "scores": sorted(scores, key=lambda item: item["score"], reverse=True),
    }


@app.post("/detect")
def detect(
    file: Annotated[UploadFile, File()],
    conf: Annotated[float, Form()] = 0.25,
    iou: Annotated[float, Form()] = 0.5,
    imgsz: Annotated[int, Form()] = 640,
) -> dict[str, Any]:
    """Return YOLO boxes, class names and confidence scores."""
    try:
        model, model_path = _detector()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    image = _read_image(file)
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    result = results[0]

    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        for xyxy, confidence, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        ):
            detections.append(
                {
                    "class_id": int(class_id),
                    "class_name": str(result.names[int(class_id)]),
                    "confidence": float(confidence),
                    "xyxy": [float(value) for value in xyxy],
                }
            )
    detections.sort(key=lambda item: item["confidence"], reverse=True)

    return {
        "filename": file.filename,
        "model": model_path.name,
        "width": image.width,
        "height": image.height,
        "confidence_threshold": float(conf),
        "iou_threshold": float(iou),
        "count": len(detections),
        "detections": detections,
    }
