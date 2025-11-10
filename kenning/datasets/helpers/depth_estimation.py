# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Functions and common depth estimation utils.
"""
import matplotlib
import numpy as np


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
