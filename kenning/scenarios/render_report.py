#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that generates report files based on Measurements JSON output.

It requires providing the report type and JSON file to extract data from.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set
import json
import numpy as np
import re
import copy
from servis import (
    render_time_series_plot_with_histogram,
    render_multiple_time_series_plot
)

if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

from kenning.resources import reports
from kenning.core.drawing import (
    draw_confusion_matrix,
    recall_precision_curves,
    recall_precision_gradients,
    true_positive_iou_histogram,
    true_positives_per_iou_range_histogram,
    draw_plot, draw_radar_chart,
    draw_violin_comparison_plot,
    draw_bubble_plot, choose_theme,
    IMMATERIAL_COLORS, RED_GREEN_CMAP)
from kenning.utils import logger
from kenning.core.report import create_report_from_measurements
from kenning.utils.class_loader import get_command
from kenning.core.metrics import compute_performance_metrics, \
    compute_classification_metrics, compute_detection_metrics

log = logger.get_logger()

SERVIS_PLOT_OPTIONS = {
    'figsize': (900, 500),
    'plottype': 'scatter',
    'backend': 'matplotlib',
}


def get_model_name(filepath: Path) -> str:
    """
    Generates the name of the model. Path to the measurements file is used for
    name generation.

    Parameters
    ----------
    filepath : Path
        Path to the measurements file

    Returns
    -------
    str : name of the model used when generating the report
    """
    return str(filepath).replace("/", ".")


def performance_report(
        measurementsdata: Dict[str, List],
        imgdir: Path,
        imgprefix: str,
        rootdir: Path,
        image_formats: Set[str],
        color_offset: int = 0,
        draw_titles: bool = True,
        **kwargs) -> str:
    """
    Creates performance section of the report.

    Parameters
    ----------
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class
    imgdir : Path
        Path to the directory for images
    imgprefix : str
        Prefix to the image file name
    rootdir : Path
        Path to the root of the documentation project involving this report
    image_formats : Set[str]
        Collection with formats which should be used to generate plots
    color_offset : int
        How many colors from default color list should be skipped
    draw_titles : bool
        Should titles be drawn on the plot

    Returns
    -------
    str :
        content of the report in MyST format
    """
    log.info(f'Running performance_report for {measurementsdata["modelname"]}')
    metrics = compute_performance_metrics(measurementsdata)
    measurementsdata |= metrics

    # Shifting colors to match color_offset
    plot_options = copy.deepcopy(SERVIS_PLOT_OPTIONS)
    plot_options['colormap'] = plot_options['colormap'][color_offset:]

    inference_step = None
    if 'target_inference_step' in measurementsdata:
        log.info('Using target measurements for inference time')
        inference_step = 'target_inference_step'
    elif 'protocol_inference_step' in measurementsdata:
        log.info('Using protocol measurements for inference time')
        inference_step = 'protocol_inference_step'
    else:
        log.warning('No inference time measurements in the report')

    if inference_step:
        usepath = imgdir / f'{imgprefix}inference_time'
        render_time_series_plot_with_histogram(
            ydata=measurementsdata[inference_step],
            xdata=measurementsdata[f'{inference_step}_timestamp'],
            title='Inference time' if draw_titles else None,
            xtitle='Time',
            xunit='s',
            ytitle='Inference time',
            yunit='s',
            outpath=str(usepath),
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )

        usepath_asterisk = Path(f'{usepath}.*')
        measurementsdata['inferencetimepath'] = str(
            usepath_asterisk.relative_to(rootdir)
        )

        measurementsdata['inferencetime'] = \
            measurementsdata[inference_step]

    if 'session_utilization_mem_percent' in measurementsdata:
        log.info('Using target measurements memory usage percentage')
        usepath = imgdir / f'{imgprefix}cpu_memory_usage'
        render_time_series_plot_with_histogram(
            ydata=measurementsdata['session_utilization_mem_percent'],
            xdata=measurementsdata['session_utilization_timestamp'],
            title='Memory usage' if draw_titles else None,
            xtitle='Time',
            xunit='s',
            ytitle='Memory usage',
            yunit='%',
            outpath=str(usepath),
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )

        usepath_asterisk = Path(f'{usepath}.*')
        measurementsdata['memusagepath'] = str(
            usepath_asterisk.relative_to(rootdir)
        )
    else:
        log.warning('No memory usage measurements in the report')

    if 'session_utilization_cpus_percent' in measurementsdata:
        log.info('Using target measurements CPU usage percentage')
        usepath = imgdir / f'{imgprefix}cpu_usage'
        render_time_series_plot_with_histogram(
            ydata=measurementsdata['session_utilization_cpus_percent_avg'],
            xdata=measurementsdata['session_utilization_timestamp'],
            title='Average CPU usage' if draw_titles else None,
            xtitle='Time',
            xunit='s',
            ytitle='Average CPU usage',
            yunit='%',
            outpath=str(usepath),
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )

        usepath_asterisk = Path(f'{usepath}.*')
        measurementsdata['cpuusagepath'] = str(
            usepath_asterisk.relative_to(rootdir)
        )
    else:
        log.warning('No memory usage measurements in the report')

    if 'session_utilization_gpu_mem_utilization' in measurementsdata:
        log.info('Using target measurements GPU memory usage percentage')
        usepath = imgdir / f'{imgprefix}gpu_memory_usage'
        gpumemmetric = 'session_utilization_gpu_mem_utilization'
        if len(measurementsdata[gpumemmetric]) == 0:
            log.warning(
                'Incorrectly collected data for GPU memory utilization'
            )
        else:
            render_time_series_plot_with_histogram(
                ydata=measurementsdata[gpumemmetric],
                xdata=measurementsdata[
                    'session_utilization_gpu_timestamp'],
                title='GPU memory usage' if draw_titles else None,
                xtitle='Time',
                xunit='s',
                ytitle='GPU memory usage',
                yunit='%',
                outpath=str(usepath),
                skipfirst=True,
                outputext=image_formats,
                **plot_options,
            )

            usepath_asterisk = Path(f'{usepath}.*')
            measurementsdata['gpumemusagepath'] = str(
                usepath_asterisk.relative_to(rootdir)
            )
    else:
        log.warning('No GPU memory usage measurements in the report')

    if 'session_utilization_gpu_utilization' in measurementsdata:
        log.info('Using target measurements GPU utilization')
        usepath = imgdir / f'{imgprefix}gpu_usage'
        if len(measurementsdata['session_utilization_gpu_utilization']) == 0:
            log.warning('Incorrectly collected data for GPU utilization')
        else:
            render_time_series_plot_with_histogram(
                ydata=measurementsdata[
                    'session_utilization_gpu_utilization'],
                xdata=measurementsdata[
                    'session_utilization_gpu_timestamp'],
                title='GPU Utilization' if draw_titles else None,
                xtitle='Time',
                xunit='s',
                ytitle='Utilization',
                yunit='%',
                outpath=str(usepath),
                skipfirst=True,
                outputext=image_formats,
                **plot_options,
            )

            usepath_asterisk = Path(f'{usepath}.*')
            measurementsdata['gpuusagepath'] = str(
                usepath_asterisk.relative_to(rootdir)
            )
    else:
        log.warning('No GPU utilization measurements in the report')

    with path(reports, 'performance.md') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            measurementsdata
        )


def comparison_performance_report(
        measurementsdata: List[Dict],
        imgdir: Path,
        rootdir: Path,
        image_formats: Set[str],
        colors=None,
        draw_titles: bool = True,
        **kwargs) -> str:
    """
    Creates performance comparison section of report.

    Parameters
    ----------
    measurementsdata : List[Dict]
        Statistics of every model from the Measurements class
    imgdir : Path
        Path to the directory for images
    rootdir : Path
        Path to the root of the documentation project involving this report
    image_formats : Set[str]
        Collection with formats which should be used to generate plots
    draw_titles : bool
        Should titles be drawn on the plot

    Returns
    -------
    str : content of the report in MyST format
    """
    log.info('Running comparison_performance_report')
    # HTML plots format unsupported, removing html
    _image_formats = image_formats - {'html'}

    metric_names = {
        'inference_step': ('Inference time', 's'),
        'session_utilization_mem_percent': ('Memory usage', '%'),
        'session_utilization_cpus_percent_avg': ('CPU usage', '%'),
        'session_utilization_gpu_mem_utilization': ('GPU memory usage', '%'),
        'session_utilization_gpu_utilization': ('GPU usage', '%')
    }
    common_metrics = set(metric_names.keys())
    hardware_usage_metrics = common_metrics - {'inference_step'}
    report_variables = {
        'reportname': measurementsdata[0]['reportname'],
        'reportname_simple': measurementsdata[0]['reportname_simple']
    }

    for data in measurementsdata:
        if 'target_inference_step' in data:
            data['inference_step'] = data['target_inference_step']
            data['inference_step_timestamp'] = \
                data['target_inference_step_timestamp']
        elif 'protocol_inference_step' in data:
            data['inference_step'] = data['protocol_inference_step']
            data['inference_step_timestamp'] = \
                data['protocol_inference_step_timestamp']

        if 'session_utilization_cpus_percent' in data:
            metrics = compute_performance_metrics(data)
            data['session_utilization_cpus_percent_avg'] = \
                metrics['session_utilization_cpus_percent_avg']

        gpumetrics = [
            'session_utilization_gpu_mem_utilization',
            'session_utilization_gpu_utilization'
        ]
        for gpumetric in gpumetrics:
            if gpumetric in data and len(data[gpumetric]) == 0:
                del data[gpumetric]

        modelmetrics = set(data.keys())
        common_metrics &= modelmetrics

    for metric, (metric_name, unit) in metric_names.items():
        metric_data = {}
        if metric_name == 'Inference time':
            timestamp_key = 'inference_step_timestamp'
        elif metric_name in ('GPU usage', 'GPU memory usage'):
            timestamp_key = 'session_utilization_gpu_timestamp'
        else:
            timestamp_key = 'session_utilization_timestamp'
        if timestamp_key not in data:
            log.warning(
                f'Missing measurement "{timestamp_key}" in the measurements ' +
                f"file. Can't provide benchmarks for {metric_name}"
            )
            continue
        timestamps = {
            data['modelname']: data[timestamp_key]
            for data in measurementsdata
        }

        for data in measurementsdata:
            if metric in data:
                metric_data[data['modelname']] = data[metric]
        if len(metric_data) > 1:
            usepath = imgdir / f"{metric}_comparison"
            render_multiple_time_series_plot(
                ydatas=[list(metric_data.values())],
                xdatas=[list(timestamps.values())],
                title=f'{metric_name} comparison' if draw_titles else None,
                subtitles=None,
                xtitles=['Time'],
                xunits=['s'],
                ytitles=[metric_name],
                yunits=[unit],
                legend_labels=list(metric_data.keys()),
                outpath=usepath,
                outputext=image_formats,
                **SERVIS_PLOT_OPTIONS,
            )
            usepath = imgdir / f"{metric}_comparison.*"
            report_variables[f"{metric}_path"] = str(
                usepath.relative_to(rootdir)
            )

    common_metrics = sorted(list(common_metrics))
    visualizationdata = {}
    for data in measurementsdata:
        visualizationdata[data['modelname']] = [
            data[metric] for metric in common_metrics
        ]

    for format in _image_formats:
        usepath = imgdir / f'mean_performance_comparison.{format}'
        draw_violin_comparison_plot(
            usepath,
            "Performance comparison plot" if draw_titles else None,
            [f'{metric_names[metric][0]} [{metric_names[metric][1]}]'
                for metric in common_metrics],
            visualizationdata,
            colors=colors,
        )
    usepath = imgdir / 'mean_performance_comparison.*'
    report_variables["meanperformancepath"] = str(
        usepath.relative_to(rootdir)
    )

    hardware_usage_metrics = sorted(list(hardware_usage_metrics))
    usage_visualization = {}
    for data in measurementsdata:
        usage_visualization[data['modelname']] = [
            np.mean(data.get(metric, 0))/100
            for metric in hardware_usage_metrics
        ]

    for format in _image_formats:
        usepath = imgdir / f"hardware_usage_comparison.{format}"
        draw_radar_chart(
            usepath,
            "Resource usage comparison" if draw_titles else None,
            usage_visualization,
            [metric_names[metric][0] for metric in hardware_usage_metrics],
            colors=colors
        )
    usepath = imgdir / "hardware_usage_comparison.*"
    report_variables["hardwareusagepath"] = str(
        usepath.relative_to(rootdir)
    )

    with path(reports, 'performance_comparison.md') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            report_variables
        )


def classification_report(
        measurementsdata: Dict[str, List],
        imgdir: Path,
        imgprefix: str,
        rootdir: Path,
        image_formats: Set[str],
        cmap=None,
        draw_titles: bool = True,
        **kwargs) -> str:
    """
    Creates classification quality section of the report.

    Parameters
    ----------
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class
    imgdir : Path
        Path to the directory for images
    imgprefix : str
        Prefix to the image file name
    rootdir : Path
        Path to the root of the documentation project involving this report
    image_formats : Set[str]
        Collection with formats which should be used to generate plots
    draw_titles : bool
        Should titles be drawn on the plot

    Returns
    -------
    str : content of the report in MyST format
    """
    log.info(f'Running classification report for {measurementsdata["modelname"]}')  # noqa: E501
    metrics = compute_classification_metrics(measurementsdata)
    measurementsdata |= metrics

    if 'eval_confusion_matrix' not in measurementsdata:
        log.error('Confusion matrix not present for classification report')
        return ''
    log.info('Using confusion matrix')
    for format in image_formats:
        confusionpath = imgdir / f'{imgprefix}confusion_matrix'
        draw_confusion_matrix(
            measurementsdata['eval_confusion_matrix'],
            str(confusionpath),
            'Confusion matrix' if draw_titles else None,
            measurementsdata['class_names'],
            format=format,
            cmap=cmap,
        )
    confusionpath = imgdir / f'{imgprefix}confusion_matrix.*'
    measurementsdata['confusionpath'] = str(
        confusionpath.relative_to(rootdir)
    )
    with path(reports, 'classification.md') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            measurementsdata
        )


def comparison_classification_report(
        measurementsdata: List[Dict],
        imgdir: Path,
        rootdir: Path,
        image_formats: Set[str],
        colors=None,
        draw_titles: bool = True,
        **kwargs) -> str:
    """
    Creates classification comparison section of report.

    Parameters
    ----------
    measurementsdata : List[Dict]
        Statistics of every model from the Measurements class
    imgdir : Path
        Path to the directory for images
    rootdir : Path
        Path to the root of the documentation project involving this report
    image_formats : Set[str]
        Collection with formats which should be used to generate plots
    draw_titles : bool
        Should titles be drawn on the plot

    Returns
    -------
    str : content of the report in MyST format
    """
    log.info('Running comparison_classification_report')
    # HTML plots format unsupported, removing html
    _image_formats = image_formats - {'html'}

    report_variables = {
        'reportname': measurementsdata[0]['reportname'],
        'reportname_simple': measurementsdata[0]['reportname_simple']
    }
    metric_visualization = {}
    accuracy, mean_inference_time, model_sizes, names = [], [], [], []
    for data in measurementsdata:
        performance_metrics = compute_performance_metrics(data)
        if 'inferencetime_mean' not in performance_metrics:
            log.warning("No inference measurements available, skipping report generation")  # noqa: E501
            return ""

        classification_metrics = compute_classification_metrics(data)
        model_accuracy = classification_metrics['accuracy']
        model_precision = classification_metrics['mean_precision']
        model_sensitivity = classification_metrics['mean_sensitivity']
        accuracy.append(model_accuracy)

        model_inferencetime_mean = performance_metrics['inferencetime_mean']
        mean_inference_time.append(model_inferencetime_mean)

        if 'compiled_model_size' in data:
            model_sizes.append(data['compiled_model_size'])
        else:
            log.warning(
                'Missing information about model size in measurements' +
                ' - computing size based on average RAM usage'
            )
            model_sizes.append(
                performance_metrics['session_utilization_mem_percent_mean']
            )
        names.append(data['modelname'])

        # Accuracy, precision, sensitivity
        metric_visualization[data['modelname']] = [
            model_accuracy,
            model_precision,
            model_sensitivity
        ]

    for format in _image_formats:
        usepath = imgdir / f"accuracy_vs_inference_time.{format}"
        draw_bubble_plot(
            usepath,
            "Accuracy vs Mean inference time" if draw_titles else None,
            mean_inference_time,
            "Mean inference time [s]",
            accuracy,
            "Accuracy",
            model_sizes,
            names,
            colors=colors,
        )
    usepath = imgdir / "accuracy_vs_inference_time.*"
    report_variables['bubbleplotpath'] = str(
        usepath.relative_to(rootdir)
    )

    for format in _image_formats:
        usepath = imgdir / f"classification_metric_comparison.{format}"
        draw_radar_chart(
            usepath,
            "Metric comparison" if draw_titles else None,
            metric_visualization,
            ["Accuracy", "Mean precision", "Mean recall"],
            colors=colors,
        )
    usepath = imgdir / "classification_metric_comparison.*"
    report_variables['radarchartpath'] = str(
        usepath.relative_to(rootdir)
    )
    report_variables['modelnames'] = names
    report_variables = {
        **report_variables,
        **metric_visualization,
    }

    with path(reports, 'classification_comparison.md') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            report_variables
        )


def detection_report(
        measurementsdata: Dict[str, List],
        imgdir: Path,
        imgprefix: str,
        rootdir: Path,
        image_formats: Set[str],
        color_offset: int = 0,
        cmap=None,
        colors=None,
        draw_titles: bool = True) -> str:
    """
    Creates detection quality section of the report.

    Parameters
    ----------
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class
    imgdir : Path
        Path to the directory for images
    imgprefix : str
        Prefix to the image file name
    rootdir : Path
        Path to the root of the documentation project involving this report
    image_formats : Set[str]
        Collection with formats which should be used to generate plots
    color_offset : int
        How many colors from default color list should be skipped
    draw_titles : bool
        Should titles be drawn on the plot

    Returns
    -------
    str : content of the report in MyST format
    """

    from kenning.datasets.helpers.detection_and_segmentation import \
        get_recall_precision, \
        compute_ap, \
        compute_map_per_threshold

    log.info(f'Running detection report for {measurementsdata["modelname"]}')
    metrics = compute_detection_metrics(measurementsdata)
    measurementsdata |= metrics

    lines = get_recall_precision(measurementsdata, 0.5)
    # HTML plots format unsupported, removing html
    _image_formats = image_formats - {'html'}

    aps = []
    for line in lines:
        aps.append(compute_ap(line[0], line[1]))

    for format in _image_formats:
        curvepath = imgdir / f'{imgprefix}recall_precision_curves.{format}'
        recall_precision_curves(
            str(curvepath),
            'Recall-Precision curves' if draw_titles else None,
            lines,
            measurementsdata['class_names']
        )
    curvepath = imgdir / f'{imgprefix}recall_precision_curves.*'
    measurementsdata['curvepath'] = str(
        curvepath.relative_to(rootdir)
    )

    for format in _image_formats:
        gradientpath = imgdir / \
            f'{imgprefix}recall_precision_gradients.{format}'
        recall_precision_gradients(
            str(gradientpath),
            'Average precision plots' if draw_titles else None,
            lines,
            measurementsdata['class_names'],
            aps,
            measurementsdata['mAP'],
            cmap=cmap,
        )
    gradientpath = imgdir / f'{imgprefix}recall_precision_gradients.*'
    measurementsdata['gradientpath'] = str(
        gradientpath.relative_to(rootdir)
    )

    tp_iou = []
    all_tp_ious = []

    for i in measurementsdata['class_names']:
        dets = measurementsdata[f'eval_det/{i}'] if f'eval_det/{i}' in measurementsdata else []  # noqa: E501
        det_tp_iou = [i[2] for i in dets if i[1]]
        if len(det_tp_iou) > 0:
            tp_iou.append(sum(det_tp_iou) / len(det_tp_iou))
            all_tp_ious.extend(det_tp_iou)
        else:
            tp_iou.append(0)

    for format in _image_formats:
        tpioupath = imgdir / f'{imgprefix}true_positive_iou_histogram.{format}'
        true_positive_iou_histogram(
            str(tpioupath),
            'Average True Positive IoU values' if draw_titles else None,
            tp_iou,
            measurementsdata['class_names'],
            colors=colors,
            color_offset=color_offset,
        )
    tpioupath = imgdir / f'{imgprefix}true_positive_iou_histogram.*'
    measurementsdata['tpioupath'] = str(
        tpioupath.relative_to(rootdir)
    )

    if len(all_tp_ious) > 0:
        for format in _image_formats:
            iouhistpath = imgdir / \
                f'{imgprefix}histogram_tp_iou_values.{format}'
            true_positives_per_iou_range_histogram(
                str(iouhistpath),
                "Histogram of True Positive IoU values" if draw_titles
                else None,
                all_tp_ious,
                colors=colors,
                color_offset=color_offset,
            )
        iouhistpath = imgdir / f'{imgprefix}histogram_tp_iou_values.*'
        measurementsdata['iouhistpath'] = str(
            iouhistpath.relative_to(rootdir)
        )

    thresholds = np.arange(0.2, 1.05, 0.05)
    mapvalues = compute_map_per_threshold(measurementsdata, thresholds)

    for format in _image_formats:
        mappath = imgdir / f'{imgprefix}map.{format}'
        draw_plot(
            str(mappath),
            'mAP value change over objectness threshold values' if draw_titles
            else None,
            'threshold',
            None,
            'mAP',
            None,
            [[thresholds, mapvalues]],
            colors=colors,
            color_offset=color_offset,
        )
    mappath = imgdir / f'{imgprefix}map.*'
    measurementsdata['mappath'] = str(
        mappath.relative_to(rootdir)
    )
    measurementsdata['max_mAP'] = max(mapvalues)
    measurementsdata['max_mAP_index'] = thresholds[np.argmax(mapvalues)].round(2)  # noqa: E501

    with path(reports, 'detection.md') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            measurementsdata
        )


def comparison_detection_report(
        measurementsdata: List[Dict],
        imgdir: Path,
        rootdir: Path,
        image_formats: Set[str],
        colors=None,
        draw_titles: bool = True,
        **kwargs) -> str:
    """
    Creates detection comparison section of report.

    Parameters
    ----------
    measurementsdata : List[Dict]
        Statistics of every model from the Measurements class
    imgdir : Path
        Path to the directory for images
    rootdir : Path
        Path to the root of the documentation project involving this report
    image_formats : Set[str]
        Collection with formats which should be used to generate plots
    draw_titles : bool
        Should titles be drawn on the plot

    Returns
    -------
    str : content of the report in MyST format
    """
    log.info('Running comparison_detection_report')

    from kenning.datasets.helpers.detection_and_segmentation import \
        compute_map_per_threshold

    report_variables = {
        'reportname': measurementsdata[0]['reportname'],
        'reportname_simple': measurementsdata[0]['reportname_simple'],
        'modelnames': []
    }
    # HTML plots format unsupported, removing html
    _image_formats = image_formats - {'html'}

    visualization_data = []
    for data in measurementsdata:
        thresholds = np.arange(0.2, 1.05, 0.05)
        mapvalues = compute_map_per_threshold(data, thresholds)
        visualization_data.append((thresholds, mapvalues))
        report_variables['modelnames'].append(data['modelname'])

    for format in _image_formats:
        usepath = imgdir / f"detection_map_thresholds.{format}"
        draw_plot(
            usepath,
            "mAP values comparison over different threshold values"
            if draw_titles else None,
            'threshold',
            None,
            'mAP',
            None,
            visualization_data,
            report_variables['modelnames'],
            colors=colors,
        )
    usepath = imgdir / "detection_map_thresholds.*"
    report_variables['mapcomparisonpath'] = str(
        usepath.relative_to(rootdir)
    )

    with path(reports, 'detection_comparison.md') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            report_variables
        )


def generate_report(
        reportname: str,
        data: List[Dict],
        outputpath: Path,
        imgdir: Path,
        report_types: List[str],
        rootdir: Path,
        image_formats: Set[str],
        command: List[str] = [],
        cmap=None,
        colors=None,
        draw_titles: bool = True) -> str:
    """
    Generates an MyST report based on Measurements data.

    The report is saved to the file in ``outputpath``.

    Parameters
    ----------
    reportname : str
        Name for the report
    data : List[Dict]
        Data for each model coming from the Measurements object,
        loaded i.e. from JSON files
    outputpath : Path
        Path to the MyST file where the report will be saved
    imgdir : Path
        Path to the directory where the report plots should be stored
    report_types : List[str]
        List of report types that define the project, i.e.
        performance, classification
    rootdir : Path
        When the report is a part of a larger MyST document (i.e. Sphinx docs),
        the rootdir parameter defines thte root directory of the document.
        It is used to compute relative paths in the document's references.
    image_formats : Set[str]
        Iterable object with extensions, in which images should be generated.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots
    command : List[str]
        Full command used to render this report, split into separate lines.
    draw_titles : bool
        Should titles be drawn on the plot
    """

    reptypes = {
        'performance': performance_report,
        'classification': classification_report,
        'detection': detection_report
    }
    comparereptypes = {
        'performance': comparison_performance_report,
        'classification': comparison_classification_report,
        'detection': comparison_detection_report
    }

    header_data = {
        'reportname': reportname,
        'modelnames': [],
        'command': []
    }

    for model_data in data:
        header_data['modelnames'].append(model_data['modelname'])
        if 'command' in model_data:
            header_data['command'] += model_data['command'] + ['']
        header_data[model_data['modelname']] = model_data

    header_data['command'] += command

    with path(reports, 'header.md') as reporttemplate:
        content = create_report_from_measurements(
            reporttemplate,
            header_data
        )

    for typ in report_types:
        for i, model_data in enumerate(data):
            if len(data) > 1:
                imgprefix = model_data["modelname"] + "_"
            else:
                imgprefix = ""
            content += reptypes[typ](
                model_data, imgdir, imgprefix, rootdir,
                image_formats, color_offset=i, cmap=cmap, colors=colors,
                draw_titles=draw_titles)
        if len(data) > 1:
            content += comparereptypes[typ](
                data, imgdir, rootdir, image_formats,
                cmap=cmap, colors=colors, draw_titles=draw_titles)

    content = re.sub(r'[ \t]+$', "", content, 0, re.M)

    with open(outputpath, 'w') as out:
        out.write(content)


def main(argv):
    command = get_command(argv)
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument(
        '--measurements',
        help='Path to the JSON files with measurements. If more than one file is provided, model comparison will be generated.',  # noqa: E501
        type=Path,
        nargs='+',
        required=True
    )
    parser.add_argument(
        'reportname',
        help='Name of the report',
        type=str
    )
    parser.add_argument(
        'output',
        help='Path to the output MyST file',
        type=Path
    )
    parser.add_argument(
        '--root-dir',
        help='Path to root directory for documentation (paths in the MyST file are relative to this directory)',  # noqa: E501
        required=True,
        type=Path
    )
    parser.add_argument(
        '--report-types',
        help='List of types that implement this report',
        nargs='+',
        required=True,
        type=str
    )
    parser.add_argument(
        '--img-dir',
        help='Path to the directory where images will be stored',
        type=Path
    )
    parser.add_argument(
        '--model-names',
        help='Names of the models used to create measurements in order',
        nargs='+',
        type=str
    )
    parser.add_argument(
        '--only-png-images',
        help="Forcing to generate images only in PNG format, if not specified also images in HTML will be generated",  # noqa: E501
        action="store_true"
    )
    parser.add_argument(
        '--use-default-theme',
        help="If this flag is specified, custom theme (defining colors for e.g. labels, backgrounds or gird) won't be used and plots' colors won't be adjusted to documentation theme",  # noqa: E501
        action='store_true'
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    args = parser.parse_args(argv[1:])

    if not args.img_dir:
        img_dir = args.root_dir / "img"
    else:
        img_dir = args.img_dir
    img_dir.mkdir(parents=True, exist_ok=True)

    if args.model_names is not None and \
            len(args.measurements) != len(args.model_names):
        log.warning("Number of model names differ from number of measurements! Ignoring --model-names argument")  # noqa: E501
        args.model_names = None

    image_formats = {'png'}
    if not args.only_png_images:
        image_formats |= {'html'}

    measurementsdata = []
    for i, measurementspath in enumerate(args.measurements):
        with open(measurementspath, 'r') as measurementsfile:
            measurements = json.load(measurementsfile)
        if args.model_names is not None:
            modelname = args.model_names[i]
        else:
            modelname = get_model_name(measurementspath)
        measurements['modelname'] = modelname
        measurements['reportname'] = args.reportname
        measurements['reportname_simple'] = re.sub(
            r'[\W]',
            '',
            args.reportname.lower().replace(' ', '_')
        )
        measurementsdata.append(measurements)

    for measurements in measurementsdata:
        if 'build_cfg' in measurements:
            measurements['build_cfg'] = json.dumps(
                measurements['build_cfg'],
                indent=4
            ).split('\n')

    cmap, colors = None, None
    if not args.use_default_theme:
        SERVIS_PLOT_OPTIONS['colormap'] = IMMATERIAL_COLORS
        cmap = RED_GREEN_CMAP
        colors = IMMATERIAL_COLORS
    elif 'colormap' in SERVIS_PLOT_OPTIONS:
        del SERVIS_PLOT_OPTIONS['colormap']

    with choose_theme(
        custom_bokeh_theme=not args.use_default_theme,
        custom_matplotlib_theme=not args.use_default_theme,
    ):
        generate_report(
            args.reportname,
            measurementsdata,
            args.output,
            img_dir,
            args.report_types,
            args.root_dir,
            image_formats,
            command,
            cmap=cmap,
            colors=colors,
            draw_titles=args.use_default_theme,
        )


if __name__ == '__main__':
    main(sys.argv)
