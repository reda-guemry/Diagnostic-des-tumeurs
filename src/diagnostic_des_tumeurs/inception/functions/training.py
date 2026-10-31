"""Frozen-backbone training and fine-tuning of the InceptionV3 model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from diagnostic_des_tumeurs.config.hyperparameters import InceptionConfig
from diagnostic_des_tumeurs.config.paths import Paths
from diagnostic_des_tumeurs.helpers.io import save_json
from diagnostic_des_tumeurs.inception.functions.data import ClassificationData
from diagnostic_des_tumeurs.inception.functions.evaluation import (
    Evaluation,
    evaluate_model,
    save_confusion_matrix,
    save_history_plot,
)
from diagnostic_des_tumeurs.inception.functions.model import build_model
from diagnostic_des_tumeurs.inception.functions.preprocessing import (
    ClassificationDatasets,
)


@dataclass
class TrainingResult:
    stage: str
    model: keras.Model
    history: dict[str, list[float]]
    evaluation: Evaluation
    model_path: Path
    history_path: Path
    plot_path: Path
    confusion_path: Path
    class_weight: dict[int, float] | None = None
    trainable_backbone_layers: int | None = None


def _history_dict(history: keras.callbacks.History) -> dict[str, list[float]]:
    return {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }


def train_frozen(
    paths: Paths,
    config: InceptionConfig,
    datasets: ClassificationDatasets,
    data: ClassificationData,
) -> TrainingResult:
    """Train only the classifier head while InceptionV3 stays frozen."""
    keras.utils.set_random_seed(config.seed)

    num_classes = len(data.classes)
    model = build_model(
        num_classes=num_classes,
        input_shape=datasets.image_size + (3,),
        dropout=config.dropout,
        dense_units=config.dense_units,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        datasets.train,
        validation_data=datasets.val,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    evaluation = evaluate_model(
        model, datasets.test, datasets.test_labels, data.classes
    )

    model.save(paths.inception_transfer)
    result = TrainingResult(
        stage="frozen",
        model=model,
        history=_history_dict(history),
        evaluation=evaluation,
        model_path=paths.inception_transfer,
        history_path=paths.inception_frozen_history,
        plot_path=paths.inception_plots_dir / "frozen_history.png",
        confusion_path=paths.inception_plots_dir / "frozen_confusion_matrix.png",
    )
    save_json(result.history_path, result.history)
    save_history_plot(
        result.history, result.plot_path, "Frozen InceptionV3 + head"
    )
    save_confusion_matrix(
        evaluation,
        data.classes,
        result.confusion_path,
        title="Confusion matrix — frozen InceptionV3 + head",
    )
    return result


def fine_tune(
    paths: Paths,
    config: InceptionConfig,
    datasets: ClassificationDatasets,
    data: ClassificationData,
) -> TrainingResult:
    """Unfreeze the last InceptionV3 layers and fine-tune on the original split."""
    if not paths.inception_transfer.exists():
        raise FileNotFoundError(
            f"{paths.inception_transfer} not found. Run the frozen-training step first."
        )
    keras.utils.set_random_seed(config.seed)

    model = keras.models.load_model(paths.inception_transfer, compile=False)
    base_model = model.get_layer("inception_v3")
    base_model.trainable = True

    if config.unfreeze_last_n > 0:
        for layer in base_model.layers[: -config.unfreeze_last_n]:
            layer.trainable = False
        for layer in base_model.layers[-config.unfreeze_last_n :]:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = False
    trainable_layers = sum(1 for layer in base_model.layers if layer.trainable)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.fine_tune_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=paths.inception_finetuned,
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=3,
            min_lr=1e-7,
        ),
    ]

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(data.classes)),
        y=datasets.train_labels,
    )
    class_weight = {
        index: float(weight) for index, weight in enumerate(class_weights_array)
    }

    history = model.fit(
        datasets.train,
        validation_data=datasets.val,
        epochs=config.fine_tune_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )
    evaluation = evaluate_model(
        model, datasets.test, datasets.test_labels, data.classes
    )
    model.save(paths.inception_finetuned)

    result = TrainingResult(
        stage="fine_tune",
        model=model,
        history=_history_dict(history),
        evaluation=evaluation,
        model_path=paths.inception_finetuned,
        history_path=paths.inception_finetuned_history,
        plot_path=paths.inception_plots_dir / "finetuned_history.png",
        confusion_path=paths.inception_plots_dir / "finetuned_confusion_matrix.png",
        class_weight=class_weight,
        trainable_backbone_layers=trainable_layers,
    )
    save_json(result.history_path, result.history)
    save_history_plot(result.history, result.plot_path, "Fine-tuning")
    save_confusion_matrix(
        evaluation,
        data.classes,
        result.confusion_path,
        title="Confusion matrix — fine-tuned InceptionV3",
    )
    return result
