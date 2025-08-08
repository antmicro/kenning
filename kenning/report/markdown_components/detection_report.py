# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for detection quality report generation.
"""

from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from kenning.core.measurements import Measurements
from kenning.core.metrics import Metric, compute_detection_metrics
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def detection_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    color_offset: int = 0,
    cmap: Optional[Any] = None,
    colors: Optional[List] = None,
    draw_titles: bool = True,
) -> str:
    """
    Creates detection quality section of the report.

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
    color_offset : int
        How many colors from default color list should be skipped.
    cmap : Optional[Any]
        Color map to be used in the plots.
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from kenning.core.drawing import (
        LinePlot,
        RecallPrecisionCurvesPlot,
        RecallPrecisionGradients,
        TruePositiveIoUHistogram,
        TruePositivesPerIoURangeHistogram,
    )
    from kenning.datasets.helpers.detection_and_segmentation import (
        compute_ap,
        compute_map_per_threshold,
        get_recall_precision,
    )

    KLogger.info(
        f'Running detection report for {measurementsdata["model_name"]}'
    )
    metrics = compute_detection_metrics(measurementsdata)
    measurementsdata |= metrics

    lines = get_recall_precision(measurementsdata, 0.5)

    aps = []
    for line in lines:
        aps.append(compute_ap(line[0], line[1]))

    curve_path = imgdir / f"{imgprefix}recall_precision_curves"
    RecallPrecisionCurvesPlot(
        title="Recall-Precision curves" if draw_titles else None,
        lines=lines,
        class_names=measurementsdata["class_names"],
    ).plot(curve_path, image_formats)
    measurementsdata["curvepath"] = get_plot_wildcard_path(
        curve_path, root_dir
    )

    gradient_path = imgdir / f"{imgprefix}recall_precision_gradients"
    RecallPrecisionGradients(
        title="Average precision plots" if draw_titles else None,
        lines=lines,
        class_names=measurementsdata["class_names"],
        avg_precisions=aps,
        mean_avg_precision=measurementsdata[Metric.mAP],
        cmap=cmap,
    ).plot(gradient_path, image_formats)
    measurementsdata["gradientpath"] = get_plot_wildcard_path(
        gradient_path, root_dir
    )

    tp_iou = []
    all_tp_ious = []

    for i in measurementsdata["class_names"]:
        dets = (
            measurementsdata[f"eval_det/{i}"]
            if f"eval_det/{i}" in measurementsdata
            else []
        )
        det_tp_iou = [i[2] for i in dets if i[1]]
        if len(det_tp_iou) > 0:
            tp_iou.append(sum(det_tp_iou) / len(det_tp_iou))
            all_tp_ious.extend(det_tp_iou)
        else:
            tp_iou.append(0)

    tpiou_path = imgdir / f"{imgprefix}true_positive_iou_histogram"
    TruePositiveIoUHistogram(
        title="Average True Positive IoU values" if draw_titles else None,
        iou_data=tp_iou,
        class_names=measurementsdata["class_names"],
        colors=colors,
        color_offset=color_offset,
    ).plot(tpiou_path, image_formats)
    measurementsdata["tpioupath"] = get_plot_wildcard_path(
        tpiou_path, root_dir
    )

    if len(all_tp_ious) > 0:
        iouhist_path = imgdir / f"{imgprefix}histogram_tp_iou_values"
        TruePositivesPerIoURangeHistogram(
            title="Histogram of True Positive IoU values"
            if draw_titles
            else None,
            iou_data=all_tp_ious,
            colors=colors,
            color_offset=color_offset,
        ).plot(iouhist_path, image_formats)
        measurementsdata["iouhistpath"] = get_plot_wildcard_path(
            iouhist_path, root_dir
        )

    thresholds = np.arange(0.2, 1.05, 0.05)
    mapvalues = compute_map_per_threshold(measurementsdata, thresholds)

    map_path = imgdir / f"{imgprefix}map"
    LinePlot(
        title=(
            "mAP value change over objectness threshold values"
            if draw_titles
            else None
        ),
        x_label="threshold",
        y_label="mAP",
        lines=[(thresholds, mapvalues)],
        colors=colors,
        color_offset=color_offset,
    ).plot(map_path, image_formats)
    measurementsdata["mappath"] = get_plot_wildcard_path(map_path, root_dir)
    measurementsdata[Metric.MAX_mAP] = max(mapvalues)
    measurementsdata[Metric.MAX_mAP_ID] = thresholds[
        np.argmax(mapvalues)
    ].round(2)

    # Find all the keys that have eval_video/* structure
    video_measurements_keys = [
        key for key in measurementsdata.keys() if key.startswith("eval_video/")
    ]
    if video_measurements_keys:
        # Calculate mAP for all the recordings
        total_mAPs = {}
        thresholds = np.arange(0.2, 1.05, 0.05)
        for key in video_measurements_keys:
            video_measurements = Measurements()
            for item in measurementsdata[key]:
                video_measurements += item
            mapvalues = compute_map_per_threshold(
                {
                    "class_names": measurementsdata["class_names"],
                    **video_measurements.data,
                },
                thresholds,
            )
            video_name = key.split("/")[-1]
            total_mAPs[video_name] = {
                "total": mapvalues[-11:][:10].mean(),
                "values": mapvalues,
            }

        # Get items of up to 5 best and worst recordings
        num_plots_to_draw = min(5, len(total_mAPs))
        sorted_mAPs = sorted(total_mAPs.items(), key=lambda x: x[1]["total"])
        min_mAPs = sorted_mAPs[:num_plots_to_draw]
        max_mAPs = sorted_mAPs[-num_plots_to_draw:]

        # Line plot for top `n` worst recordings
        map_path = imgdir / f"{imgprefix}map_worst_recordings"
        lines = []
        labels = []
        for recording_name, mAP in min_mAPs:
            lines.append((thresholds, mAP["values"]))
            labels.append(recording_name)
        LinePlot(
            title=f"mAP for {num_plots_to_draw} worst recordings"
            if draw_titles
            else None,
            x_label="threshold",
            y_label="mAP",
            lines=lines,
            lines_labels=labels,
            colors=colors,
        ).plot(map_path, image_formats)
        measurementsdata["map_worst_recordings"] = get_plot_wildcard_path(
            map_path, root_dir
        )

        # Line plot for top `n` best recordings
        map_path = imgdir / f"{imgprefix}map_best_recordings"
        lines = []
        labels = []
        for recording_name, mAP in max_mAPs:
            lines.append((thresholds, mAP["values"]))
            labels.append(recording_name)
        LinePlot(
            title=f"mAP for {num_plots_to_draw} best recordings"
            if draw_titles
            else None,
            x_label="threshold",
            y_label="mAP",
            lines=lines,
            lines_labels=labels,
            colors=colors,
        ).plot(map_path, image_formats)
        measurementsdata["map_best_recordings"] = get_plot_wildcard_path(
            map_path, root_dir
        )

    with path(reports, "detection.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, measurementsdata
        ), metrics
