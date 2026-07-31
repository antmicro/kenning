# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for basic model statistics report generation.
"""

from collections import Counter
from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from kenning.converters import converter_registry
from kenning.core.exceptions import ConversionError
from kenning.core.model import ModelWrapper
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def model_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    draw_titles: bool = True,
    colors: Optional[List] = None,
    color_offset: int = 0,
    model_wrapper: Optional[ModelWrapper] = None,
    **kwargs: Any,
) -> Tuple[str, Dict]:
    """
    Creates model section of the report.

    Parameters
    ----------
    measurementsdata : Dict[str, Any]
        Statistics from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    imgprefix : str
        Prefix to the image file name.
    root_dir : Path
        Path to the root of the documentation project
        involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    colors : Optional[List]
        Colors to be used in the plots.
    color_offset : int
        How many colors from default color list should be skipped.
    model_wrapper: Optional[ModelWrapper]
        ModelWrapper of the reported model
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Tuple[str, Dict]
        Content of the report in MyST format.
    """
    KLogger.info(f'Running model_report for {measurementsdata["model_name"]}')

    if model_wrapper is None:
        KLogger.warn(
            "Cannot generate model specification report"
            " - model_wrapper is not provided"
        )
        return "", {}

    io_spec = model_wrapper.get_io_specification()
    model_cls = None

    try:
        model_wrapper.prepare_model()
        model_cls = model_wrapper.model
    except AttributeError:
        KLogger.warn(
            "Could not prepare model from model_wrapper and retrieve model"
        )
        pass

    conversion_kwargs = {
        "io_spec": io_spec,
        "model_cls": model_cls,
    }

    input_type = model_wrapper.get_framework()

    measurementsdata["base model type"] = input_type

    measurementsdata[
        "framework version"
    ] = model_wrapper.get_framework_version()

    ms = model_wrapper.get_model_size()

    measurementsdata["base model size"] = int(ms * 1024)

    input_model_path = model_wrapper.get_path()
    try:
        onnx_model = converter_registry.convert(
            input_model_path, input_type, "onnx", **conversion_kwargs, **kwargs
        )
    except ConversionError:
        KLogger.warn("Cannot convert model to onnx")
        return "", {}

    initializer_map = {
        init.name: init for init in onnx_model.graph.initializer
    }

    total_params = 0
    total_bytes = 0
    layer_count = 0

    layers = list()

    from onnx import numpy_helper

    for node in onnx_model.graph.node:
        layer_params = 0
        layer_bytes = 0
        dtypes = set()

        for input_name in node.input:
            if input_name not in initializer_map:
                continue
            tensor = numpy_helper.to_array(initializer_map[input_name])
            layer_params += tensor.size
            layer_bytes += tensor.nbytes
            dtypes.add(str(tensor.dtype))

        total_params += layer_params
        total_bytes += layer_bytes
        dtype_str = ", ".join(sorted(dtypes)) if dtypes else "-"
        layer_count += 1

        layers.append(
            (
                layer_count,
                node.name or node.output[0],
                layer_params,
                layer_bytes,
                dtype_str,
            )
        )

    from kenning.core.drawing import Barplot

    KLogger.info("Using layer type count")
    layer_type_count_plot_path = imgdir / f"{imgprefix}layer_type_count"

    ops = Counter(node.op_type for node in onnx_model.graph.node)
    model_ops_name = sorted(ops, key=str.lower)
    model_ops_count = list(ops[name] for name in model_ops_name)

    Barplot(
        title="Layer operation type counts" if draw_titles else None,
        x_label="Operation type",
        y_label="Layers count",
        x_data=model_ops_name,
        y_data={"count": model_ops_count},
        colors=colors,
        color_offset=color_offset,
        max_bars_matplotlib=32,
    ).plot(layer_type_count_plot_path, image_formats)
    measurementsdata["layer_type_count_bar_path"] = get_plot_wildcard_path(
        layer_type_count_plot_path, root_dir
    )

    measurementsdata["layer count"] = layer_count
    measurementsdata["total parameters"] = total_params
    measurementsdata["total bytes"] = total_bytes

    measurementsdata["layers"] = layers
    measurementsdata["layer statistics"] = [
        "Number",
        "Name",
        "Parameters",
        "Bytes",
        "Data type",
    ]

    with path(reports, "model.md") as report_template:
        report = create_report_from_measurements(
            report_template, measurementsdata
        )
        return report, {}
