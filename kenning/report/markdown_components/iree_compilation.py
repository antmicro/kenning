# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module generating report section with data gathered from IREE compilation.
"""

from collections import Counter, defaultdict
from importlib.resources import path
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports

# Mapping of IREE device names to human-readable version
# displayed in the report
DEVICE_NAMES = {"device_names": {"coralnpu": "CoralNPU"}}


def iree_compilation_report(
    measurementsdata: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[str, Dict]:
    """
    Creates IREE compilation statistics section of the report.

    Parameters
    ----------
    measurementsdata : Dict[str, Any]
        Statistics from the Measurements class.
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

    register_allocation_summary = defaultdict(dict)
    if "register_allocation" in iree_metadata:
        for name, data in iree_metadata["register_allocation"].items():
            total_spills = 0
            total_reloads = 0
            has_scalar_spills = False
            vec_registers_usage = Counter()
            vec_registers_count = 0
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
            register_allocation_summary[name]["total_spills"] = total_spills
            register_allocation_summary[name]["total_reloads"] = total_reloads
            register_allocation_summary[name][
                "has_scalar_spills"
            ] = has_scalar_spills
            register_allocation_summary[name]["used_vector_registers"] = list(
                vec_registers_usage.keys()
            )
            register_allocation_summary[name][
                "used_vector_registers_count"
            ] = vec_registers_count

    calculated_metrics = {
        "affinities_summary": affinities_summary,
        "register_allocation_summary": register_allocation_summary,
    }
    with path(reports, "coralnpu_compilation.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            iree_metadata
            | DEVICE_NAMES
            | {"model_name": measurementsdata["model_name"]}
            | calculated_metrics,
        ), calculated_metrics


def comparison_iree_compilation_report(
    measurementsdata: List[Dict[str, Any]],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    color_offset: int = 0,
    cmap: Optional[Any] = None,
    colors: Optional[List] = None,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates IREE compilation comparison section of the report.

    Parameters
    ----------
    measurementsdata : List[Dict[str, Any]]
        Statistics from the Measurements class.
    imgdir : Path
        Path to the directory for images.
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
    str
        Content of the report in MyST format.
    """
    from kenning.core.drawing import Barplot

    iree_metadata = [
        data.get("compilation_metadata", {}).get("iree", {})
        for data in measurementsdata
    ]
    if any(
        "affinities" not in data and "register_allocation" not in data
        for data in iree_metadata
    ):
        return ""

    names = [data["model_name"] for data in measurementsdata]
    report_variables = {
        "model_names": names,
    }
    if not any("affinities" not in data for data in iree_metadata):
        dispatch_distribution_path = imgdir / "dispatch_distribution"

        affinities = [data["affinities"] for data in iree_metadata]
        devices = set()
        [devices.update(aff.keys()) for aff in affinities]
        devices = sorted(list(devices))

        data_keys = (
            ("Static", "static-dispatch-count"),
            ("Dynamic", "dynamic-dispatch-count"),
        )
        y = {
            f"{DEVICE_NAMES.get(dev, dev.capitalize())} - {name}": [
                aff[dev][k] if dev in aff else 0 for aff in affinities
            ]
            for dev, (name, k) in product(devices, data_keys)
        }

        Barplot(
            title="Dispatch distribution" if draw_titles else None,
            x_label="Model",
            y_label="Number of dispatches",
            x_data=names,
            y_data=y,
            colors=colors[: len(y.keys())],
            max_bars_matplotlib=32,
            stacked=list(y.keys()),
            tooltip=[
                ("Model", "@xdata"),
                ("Number of $name dispatches", "@$name"),
            ],
        ).plot(dispatch_distribution_path, image_formats)

        report_variables[
            "dispatch_distribution_path"
        ] = get_plot_wildcard_path(dispatch_distribution_path, root_dir)
        report_variables["dispatch_distribution_data"] = {
            "header": list(y.keys()) + ["Total"],
            "Total": list(map(sum, zip(*y.values()))),
        } | y
        report_variables["devices"] = devices
        report_variables["affinities"] = [
            data["affinities"] for data in iree_metadata
        ]

    if not any("register_allocation" not in data for data in iree_metadata):
        report_variables["register_allocations_path"] = {}
        report_variables["register_allocations_data"] = defaultdict(dict)
        allocations = [data["register_allocation"] for data in iree_metadata]
        types = set()
        [types.update(alloc.keys()) for alloc in allocations]
        for regalloc_type in types:
            register_allocations_path = (
                imgdir / f"register_allocations_{regalloc_type}"
            )

            dispatch_names = []
            [
                [
                    dispatch_names.append(dispatch["name"])
                    for dispatch in alloc[regalloc_type]["dispatches"]
                    if dispatch["name"] not in dispatch_names
                ]
                for alloc in allocations
            ]
            dispatch_names = list(dispatch_names)
            model_names = [
                name
                for name, alloc in zip(enumerate(names), allocations)
                if regalloc_type in alloc
            ]
            if len(model_names) < 2:
                # Do not generate plot for single set of data
                continue

            def _find_dispatch(dispatches, dispatch_name):
                return next(
                    filter(
                        lambda x: x["name"] == dispatch_name,
                        dispatches,
                    ),
                    {},
                )

            x = list(product(dispatch_names, [n for _, n in model_names]))
            xi = list(product(dispatch_names, model_names))
            data_keys = (("Spills", "vec_spills"), ("Reloads", "vec_reloads"))
            y = {
                name: [
                    _find_dispatch(
                        allocations[i][regalloc_type]["dispatches"], dispatch
                    ).get(k, 0)
                    for dispatch, (i, _name) in xi
                ]
                for name, k in data_keys
            }

            Barplot(
                title="Register allocations" if draw_titles else None,
                x_label="Dispatch, model",
                y_label="Quantity",
                x_data=x,
                y_data=y,
                colors=colors[: len(y.keys())],
                max_bars_matplotlib=32,
                vertical_x_labels=True,
                tooltip_additional_label="Type",
            ).plot(register_allocations_path, image_formats)

            report_variables["register_allocations_path"][
                regalloc_type
            ] = get_plot_wildcard_path(register_allocations_path, root_dir)
            report_variables["register_allocations_data"][regalloc_type] = {
                (dispatch, name): _find_dispatch(
                    allocations[i][regalloc_type]["dispatches"], dispatch
                )
                for dispatch, (i, name) in xi
            }
            # Calculate register_allocations summary
            register_allocation_summary = defaultdict(
                lambda: defaultdict(dict)
            )
            for metadata, (_, model) in zip(iree_metadata, model_names):
                total_spills = 0
                total_reloads = 0
                has_scalar_spills = False
                vec_registers_usage = Counter()
                vec_registers_count = 0
                for dispatch in metadata["register_allocation"][regalloc_type][
                    "dispatches"
                ]:
                    total_spills += dispatch["vec_spills"]
                    total_reloads += dispatch["vec_reloads"]
                    has_scalar_spills |= dispatch["has_scalar_spills"]
                    vec_registers_usage += Counter(
                        dispatch["global_vector_registers"]
                    )
                    vec_registers_count += dispatch[
                        "global_vector_registers_count"
                    ]
                register_allocation_summary[model][
                    "total_spills"
                ] = total_spills
                register_allocation_summary[model][
                    "total_reloads"
                ] = total_reloads
                register_allocation_summary[model][
                    "has_scalar_spills"
                ] = has_scalar_spills
                register_allocation_summary[model][
                    "used_vector_registers"
                ] = list(vec_registers_usage.keys())
                register_allocation_summary[model][
                    "used_vector_registers_count"
                ] = vec_registers_count
            if "register_allocation_summary" not in report_variables:
                report_variables["register_allocation_summary"] = {}
            report_variables["register_allocation_summary"][
                regalloc_type
            ] = register_allocation_summary
        report_variables["regalloc_types"] = types

    if sum("compilation_duration" in data for data in iree_metadata) > 1:
        report_variables["compilation_duration"] = {
            name: data["compilation_duration"]
            for name, data in zip(names, iree_metadata)
            if "compilation_duration" in data
        }

    with path(reports, "coralnpu_compilation_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            DEVICE_NAMES | report_variables,
        )
