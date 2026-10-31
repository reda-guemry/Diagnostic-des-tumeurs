"""Test-set evaluation, reports, curves and confusion matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


@dataclass(frozen=True)
class Evaluation:
    loss: float
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray
    report: str
    report_dict: dict[str, Any]
    confusion: np.ndarray

    @property
    def metrics(self) -> dict[str, float]:
        macro = self.report_dict.get("macro avg", {})
        weighted = self.report_dict.get("weighted avg", {})
        return {
            "loss": float(self.loss),
            "accuracy": float(self.accuracy),
            "macro_precision": float(macro.get("precision", 0.0)),
            "macro_recall": float(macro.get("recall", 0.0)),
            "macro_f1": float(macro.get("f1-score", 0.0)),
            "weighted_f1": float(weighted.get("f1-score", 0.0)),
        }

    @property
    def per_class(self) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for class_name, values in self.report_dict.items():
            if not isinstance(values, dict) or class_name in {
                "accuracy",
                "macro avg",
                "weighted avg",
            }:
                continue
            rows.append(
                {
                    "class": class_name,
                    "precision": float(values["precision"]),
                    "recall": float(values["recall"]),
                    "f1-score": float(values["f1-score"]),
                    "support": int(values["support"]),
                }
            )
        return rows


def evaluate_model(
    model: Any,
    dataset: Any,
    labels: np.ndarray,
    classes: list[str],
    verbose: int = 0,
) -> Evaluation:
    loss, accuracy = model.evaluate(dataset, verbose=verbose)

    y_true: list[int] = []
    y_pred: list[int] = []
    for images, batch_labels in dataset:
        probabilities = model(images, training=False)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities.numpy(), axis=1).tolist())

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    label_ids = np.arange(len(classes))
    report_dict = classification_report(
        y_true_array,
        y_pred_array,
        labels=label_ids,
        target_names=classes,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report = classification_report(
        y_true_array,
        y_pred_array,
        labels=label_ids,
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    return Evaluation(
        loss=float(loss),
        accuracy=float(accuracy_score(labels, y_pred_array)),
        y_true=y_true_array,
        y_pred=y_pred_array,
        report=report,
        report_dict=report_dict,
        confusion=confusion_matrix(y_true_array, y_pred_array, labels=label_ids),
    )


def save_confusion_matrix(
    evaluation: Evaluation,
    classes: list[str],
    path: str | Path,
    title: str = "Confusion matrix",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(
        confusion_matrix=evaluation.confusion,
        display_labels=classes,
    )
    display.plot(xticks_rotation=45, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_history_plot(
    history: dict[str, list[float]],
    path: str | Path,
    title: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.get("accuracy", []), label="Train accuracy")
    axes[0].plot(history.get("val_accuracy", []), label="Validation accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{title} — accuracy")
    axes[0].legend()

    axes[1].plot(history.get("loss", []), label="Train loss")
    axes[1].plot(history.get("val_loss", []), label="Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{title} — loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
