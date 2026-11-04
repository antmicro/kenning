# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with CNN based class used for anomaly detection.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


def get_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    A function that returns a CNN based model for anomaly detection.

    Parameters
    ----------
    input_shape : Tuple[int, int]
        Window length and number of features.

    Returns
    -------
    tf.keras.Model
        Two-class classifier with softmax probabilities.

    Raises
    ------
    ValueError
        When provided input_shape is invalid.
    """
    if len(input_shape) != 2 or input_shape[0] is None or input_shape[0] < 2:
        raise ValueError(
            "input_shape must be (window_size >= 2, num_features)"
        )

    def res_block(x, filters, kernel_size=5, dilation=1, dropout=0.1):
        shortcut = x
        for i in range(2):
            x = layers.Conv1D(
                filters,
                kernel_size=kernel_size,
                padding="causal",
                dilation_rate=dilation,
                use_bias=False,
            )(x)
            x = layers.BatchNormalization()(x)
            if i == 0:
                x = layers.ReLU(max_value=6)(x)

        x = layers.Add()([x, shortcut])
        x = layers.ReLU(max_value=6)(x)
        return layers.SpatialDropout1D(dropout)(x)

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(24, 1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Receptive field: 1 + 2 * (5 - 1) * (1 + 2 + 4 + 8) = 121 samples.
    for dilation in (1, 2, 4, 8):
        x = res_block(x, 24, dilation=dilation)

    crop = (input_shape[0] - 1, 0)
    last = layers.Flatten()(layers.Cropping1D(cropping=crop)(x))
    current = layers.Flatten()(layers.Cropping1D(cropping=crop)(inputs))
    x = layers.Concatenate()([last, current])

    x = layers.Dense(24)(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)
