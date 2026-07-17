# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for basic model statistics report generation.
"""

from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, Set

from kenning.converters import converter_registry
from kenning.core.exceptions import ConversionError
from kenning.core.model import ModelWrapper
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def model_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    draw_titles: bool = True,
    model_wrapper: ModelWrapper = None,
    **kwargs: Any,

) -> str:
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
    model_wrapper: ModelWrapper = None,
        ModelWrapper of the reported model
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    KLogger.info(
        f'Running model_report for {measurementsdata["model_name"]}'
    )
    if model_wrapper is None:
        KLogger.warn(
            "Cannot generate model specification report"
            " - model_wrapper is not provided"
        )
        return "", {}

    io_spec = model_wrapper.get_io_specification()    
    model_cls = None

    try:
        model_wrapper.create_model_structure()
        model_cls = model_wrapper.model
    except AttributeError:
        pass

    conversion_kwargs = {
        "io_spec": io_spec,
        "model_cls": model_cls,
    }

    input_type = model_wrapper.get_framework()
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

    name_list = list()
    params_list = list()
    bytes_list = list()

    dtypes_list = list()

    from onnx import numpy_helper
    # TODO: weight type detection
    for node in onnx_model.graph.node:
        layer_params = 0
        layer_bytes = 0
        dtypes = set()

        for input_name in node.input:
            if input_name in initializer_map:
                tensor = numpy_helper.to_array(initializer_map[input_name])
                layer_params += tensor.size
                layer_bytes += tensor.nbytes
                dtypes.add(str(tensor.dtype))

        total_params += layer_params
        total_bytes += layer_bytes
        dtype_str = ", ".join(sorted(dtypes)) if dtypes else "-"
        layer_count += 1

        name_list.append(node.name or node.output[0])
        params_list.append(layer_params)
        bytes_list.append(layer_bytes)
        dtypes_list.append(dtype_str)

    measurementsdata["layer count"] = layer_count
    measurementsdata["total parameters"] = total_params
    measurementsdata["parameters"] = params_list
    measurementsdata["total bytes"] = total_bytes
    measurementsdata["bytes"] = bytes_list
    measurementsdata["data types"] = dtypes_list
    measurementsdata["name list"] = name_list

    with path(reports, "model.md") as report_template:
        report = create_report_from_measurements(
            report_template, measurementsdata
        )        
        return report, {}
