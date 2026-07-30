# Copyright (c) 2025-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Functions and common depth estimation utils.
"""
from typing import NamedTuple

import matplotlib
import numpy as np

from kenning.core.measurements import Measurements
from kenning.core.metrics import (
    mean_absolute_error,
    mean_absolute_relative_error,
    mean_squared_error,
    regression_threshold_accuracy,
)


def render_depth(
    values: np.ndarray, colormap_name: str = "magma_r"
) -> np.ndarray:
    """
    Method used to render depth to an image.

    Parameters
    ----------
    values : np.ndarray
        Depth data
    colormap_name: str
        Colormap type used for visualization

    Returns
    -------
    np.ndarray
        Rendered Image.
    """
    values = values.squeeze()
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)
    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True)
    colors = colors[:, :, :3]
    colors = colors[:, :, ::-1]  # (RGB -> BGR)
    return np.array(colors)


def calculate_partial_metrics(
    predictions: np.ndarray, truth: np.ndarray
) -> Measurements:
    """
    Method used to calculate partial depth estimation evaluation metrics
    for a single batch. Because holding entire depth maps for all images
    in memory is often impossible, it is necessary to calculate the metrics
    for each batch separately and later compute their mean.

    Parameters
    ----------
    predictions : np.ndarray
        Depth map returned by the model.
    truth : np.ndarray
        Depth map from dataset target.

    Returns
    -------
    Measurements
        Measurements object with calculated partial metrics.

    """
    measurements = Measurements()

    measurements.add_measurement(
        "depth_part_MAE",
        [mean_absolute_error(predictions, truth)],
        lambda: [],
    )

    measurements.add_measurement(
        "depth_part_MSE",
        [mean_squared_error(predictions, truth)],
        lambda: [],
    )

    measurements.add_measurement(
        "depth_part_MARE",
        [mean_absolute_relative_error(predictions, truth)],
        lambda: [],
    )

    for power_coef in (0.125, 1, 2, 3):
        measurements.add_measurement(
            f"depth_part_reg_thresh_acc_{power_coef}",
            [regression_threshold_accuracy(predictions, truth, power_coef)],
            lambda: [],
        )

    return measurements


class PredictionSample(NamedTuple):
    """
    A tuple holding information necessary for saving a single prediction
    received in the evaluate method.

    Attributes
    ----------
    internal_idx : int
        The idx of the image and depth map in the dataset internal fields.
    score : float
        The value of the metric selected for evaluation.
    prediction : np.ndarray
        The full predicted depth map.
    """

    internal_idx: int
    score: float
    prediction: np.ndarray
