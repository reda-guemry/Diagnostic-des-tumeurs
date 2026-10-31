"""The single frozen-InceptionV3 + classifier-head model."""

from __future__ import annotations

import keras
from keras.applications import InceptionV3


def build_model(
    num_classes: int,
    input_shape: tuple[int, int, int],
    dropout: float = 0.5,
    dense_units: int = 256,
) -> keras.Model:
    """InceptionV3 backbone (frozen) followed by a small classification head.

    The backbone and the head belong to one model; no separate feature
    extractor or feature arrays are produced.
    """
    base_model = InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        name="inception_v3",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="image")
    # InceptionV3 expects pixels in [-1, 1]; rescaling lives inside the model
    # so the data pipeline can keep raw [0, 255] values.
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="dense_1")(x)

    return keras.Model(inputs, outputs, name="inceptionv3_classifier")
