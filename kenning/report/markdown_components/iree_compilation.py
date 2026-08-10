# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module generating report section with data gathered from IREE compilation.
"""

from collections import Counter
from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from kenning.report.markdown_components.general import (
    create_report_from_measurements,
)
from kenning.resources import reports


def iree_compilation_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    color_offset: int = 0,
    cmap: Optional[Any] = None,
    colors: Optional[List] = None,
    draw_titles: bool = True,
    **kwargs: Any,
) -> Tuple[str, Dict]:
    """
    Creates IREE compilation statistics section of the report.

    Parameters
    ----------
    measurementsdata : Dict[str, Any]
        Statistics from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    imgprefix : str
        Prefix to the image file name.
    root_dir : Path
        Path to the root of the documentation project involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    color_offset : int
        How many colors from default color list should be skipped.
    cmap : Optional[Any]
        Color map to be used in the plots.
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    **kwargs: Any
        Additional arguments (not used here).

    Returns
    -------
    Tuple[str, Dict]
        Content of the report in MyST format, a dict of measurements.
    """
    iree_metadata = measurementsdata["compilation_metadata"]["iree"]
    if (
        "affinities" not in iree_metadata
        and "register_allocation" not in iree_metadata
    ):
        return "", {}
    device_names = {"device_names": {"coralnpu": "CoralNPU"}}

    affinities_summary = {}
    if "affinities" in iree_metadata:
        for key in (
            "dispatch-count",
            "static-dispatch-count",
            "dynamic-dispatch-count",
        ):
            affinities_summary[key] = sum(
                d[key] for d in iree_metadata["affinities"].values()
            )

    register_allocation_summary = {}
    if "register_allocation" in iree_metadata:
        total_spills = 0
        total_reloads = 0
        has_scalar_spills = False
        vec_registers_usage = Counter()
        vec_registers_count = 0
        for name, data in iree_metadata["register_allocation"].items():
            for dispatch in data["dispatches"]:
                total_spills += dispatch["vec_spills"]
                total_reloads += dispatch["vec_reloads"]
                has_scalar_spills |= dispatch["has_scalar_spills"]
                vec_registers_usage += Counter(
                    dispatch["global_vector_registers"]
                )
                vec_registers_count += dispatch[
                    "global_vector_registers_count"
                ]
        register_allocation_summary["total_spills"] = total_spills
        register_allocation_summary["total_reloads"] = total_reloads
        register_allocation_summary["has_scalar_spills"] = has_scalar_spills
        register_allocation_summary["used_vector_registers"] = list(
            vec_registers_usage.keys()
        )
        register_allocation_summary[
            "used_vector_registers_count"
        ] = vec_registers_count

    with path(reports, "coralnpu_compilation.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            iree_metadata
            | device_names
            | {
                "affinities_summary": affinities_summary,
                "register_allocation_summary": register_allocation_summary,
            },
        ), measurementsdata["compilation_metadata"]["iree"]
