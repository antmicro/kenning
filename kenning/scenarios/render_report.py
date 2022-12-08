#!/usr/bin/env python

"""
A script that generates report files based on Measurements JSON output.

It requires providing the report type and JSON file to extract data from.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List
import json
import numpy as np

if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

from kenning.resources import reports
from kenning.core.drawing import time_series_plot
from kenning.core.drawing import draw_confusion_matrix
from kenning.core.drawing import recall_precision_curves
from kenning.core.drawing import recall_precision_gradients
from kenning.core.drawing import true_positive_iou_histogram
from kenning.core.drawing import true_positives_per_iou_range_histogram
from kenning.core.drawing import draw_plot
from kenning.core.drawing import draw_violin_comparison_plot
from kenning.core.drawing import draw_multiple_time_series
from kenning.core.drawing import draw_radar_chart
from kenning.core.drawing import draw_bubble_plot
from kenning.utils import logger
from kenning.core.report import create_report_from_measurements
from kenning.utils.class_loader import get_command

log = logger.get_logger()


def get_model_name(filepath: Path) -> str:
    """
    Generates the name of the model. Path to the measurements file is used for
    name generation.

    Parameters
    ----------
    filepath: Path
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
        rootdir: Path) -> str:
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

    Returns
    -------
    str :
        content of the report in RST format
    """
    log.info(f'Running performance_report for {measurementsdata["modelname"]}')

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
        usepath = imgdir / f'{imgprefix}inference_time.png'
        time_series_plot(
            str(usepath),
            'Inference time',
            'Time', 's',
            'Inference time', 's',
            measurementsdata[f'{inference_step}_timestamp'],
            measurementsdata[inference_step],
            skipfirst=True)
        measurementsdata['inferencetimepath'] = str(
            usepath.relative_to(rootdir)
        )
        measurementsdata['inferencetime'] = \
            measurementsdata[inference_step]

    if 'session_utilization_mem_percent' in measurementsdata:
        log.info('Using target measurements memory usage percentage')
        usepath = imgdir / f'{imgprefix}cpu_memory_usage.png'
        time_series_plot(
            str(usepath),
            'Memory usage',
            'Time', 's',
            'Memory usage', '%',
            measurementsdata['session_utilization_timestamp'],
            measurementsdata['session_utilization_mem_percent'])
        measurementsdata['memusagepath'] = str(
            usepath.relative_to(rootdir)
        )
    else:
        log.warning('No memory usage measurements in the report')

    if 'session_utilization_cpus_percent' in measurementsdata:
        log.info('Using target measurements CPU usage percentage')
        usepath = imgdir / f'{imgprefix}cpu_usage.png'
        measurementsdata['session_utilization_cpus_percent_avg'] = [
            np.mean(cpus) for cpus in
            measurementsdata['session_utilization_cpus_percent']
        ]
        time_series_plot(
            str(usepath),
            'Mean CPU usage',
            'Time', 's',
            'Mean CPU usage', '%',
            measurementsdata['session_utilization_timestamp'],
            measurementsdata['session_utilization_cpus_percent_avg'])
        measurementsdata['cpuusagepath'] = str(
            usepath.relative_to(rootdir)
        )
    else:
        log.warning('No memory usage measurements in the report')

    if 'session_utilization_gpu_mem_utilization' in measurementsdata:
        log.info('Using target measurements GPU memory usage percentage')
        usepath = imgdir / f'{imgprefix}gpu_memory_usage.png'
        gpumemmetric = 'session_utilization_gpu_mem_utilization'
        if len(measurementsdata[gpumemmetric]) == 0:
            log.warning(
                'Incorrectly collected data for GPU memory utilization'
            )
        else:
            time_series_plot(
                str(usepath),
                'GPU memory usage',
                'Time', 's',
                'GPU memory usage', '%',
                measurementsdata['session_utilization_gpu_timestamp'],
                measurementsdata[gpumemmetric])
            measurementsdata['gpumemusagepath'] = str(
                usepath.relative_to(rootdir)
            )
    else:
        log.warning('No GPU memory usage measurements in the report')

    if 'session_utilization_gpu_utilization' in measurementsdata:
        log.info('Using target measurements GPU utilization')
        usepath = imgdir / f'{imgprefix}gpu_usage.png'
        if len(measurementsdata['session_utilization_gpu_utilization']) == 0:
            log.warning('Incorrectly collected data for GPU utilization')
        else:
            time_series_plot(
                str(usepath),
                'GPU Utilization',
                'Time', 's',
                'Utilization', '%',
                measurementsdata['session_utilization_gpu_timestamp'],
                measurementsdata['session_utilization_gpu_utilization'])
            measurementsdata['gpuusagepath'] = str(
                usepath.relative_to(rootdir)
            )
    else:
        log.warning('No GPU utilization measurements in the report')

    with path(reports, 'performance.rst') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            measurementsdata
        )


def comparison_performance_report(
        measurementsdata: List[Dict],
        imgdir: Path,
        rootdir: Path) -> str:
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

    Returns
    -------
    str : content of the report in RST format
    """
    log.info('Running comparison_performance_report')

    metric_names = {
        'inference_step': ('Inference time', 's'),
        'session_utilization_mem_percent': ('Memory usage', '%'),
        'session_utilization_cpus_percent': ('CPU usage', '%'),
        'session_utilization_gpu_mem_utilization': ('GPU memory usage', '%'),
        'session_utilization_gpu_utilization': ('GPU usage', '%')
    }
    common_metrics = set(metric_names.keys())
    hardware_usage_metrics = common_metrics - {'inference_step'}
    report_variables = {
        'reportname': measurementsdata[0]['reportname']
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
            data['session_utilization_cpus_percent'] = [
                np.mean(cpus) for cpus in
                data['session_utilization_cpus_percent']
            ]

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
        timestamps = {
            data['modelname']: data[timestamp_key]
            for data in measurementsdata
        }

        for data in measurementsdata:
            if metric in data:
                metric_data[data['modelname']] = data[metric]
        if len(metric_data) > 1:
            usepath = imgdir / f"{metric}_comparison.png"
            draw_multiple_time_series(
                usepath,
                f"{metric_name} comparison",
                timestamps,
                "Time [s]",
                metric_data,
                f"{metric_name} [{unit}]",
                smooth=101
            )
            report_variables[f"{metric}_path"] = str(
                usepath.relative_to(rootdir)
            )

    common_metrics = sorted(list(common_metrics))
    visualizationdata = {}
    for data in measurementsdata:
        visualizationdata[data['modelname']] = [
            data[metric] for metric in common_metrics
        ]

    usepath = imgdir / 'mean_performance_comparison.png'
    draw_violin_comparison_plot(
        usepath,
        "Performance comparison plot",
        [f'{metric_names[metric][0]} [{metric_names[metric][1]}]'
         for metric in common_metrics],
        visualizationdata
    )
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

    usepath = imgdir / "hardware_usage_comparison.png"
    draw_radar_chart(
        usepath,
        "Resource usage comparison",
        usage_visualization,
        [metric_names[metric][0] for metric in hardware_usage_metrics]
    )
    report_variables["hardwareusagepath"] = str(
        usepath.relative_to(rootdir)
    )

    with path(reports, 'performance_comparison.rst') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            report_variables
        )


def classification_report(
        measurementsdata: Dict[str, List],
        imgdir: Path,
        imgprefix: str,
        rootdir: Path) -> str:
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

    Returns
    -------
    str : content of the report in RST format
    """
    log.info(f'Running classification report for {measurementsdata["modelname"]}')  # noqa: E501

    if 'eval_confusion_matrix' not in measurementsdata:
        log.error('Confusion matrix not present for classification report')
        return ''
    log.info('Using confusion matrix')
    confusionpath = imgdir / f'{imgprefix}confusion_matrix.png'
    draw_confusion_matrix(
        measurementsdata['eval_confusion_matrix'],
        str(confusionpath),
        'Confusion matrix',
        measurementsdata['class_names']
    )
    measurementsdata['confusionpath'] = str(
        confusionpath.relative_to(rootdir)
    )
    with path(reports, 'classification.rst') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            measurementsdata
        )


def comparison_classification_report(
        measurementsdata: List[Dict],
        imgdir: Path,
        rootdir: Path) -> str:
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

    Returns
    -------
    str : content of the report in RST format
    """
    log.info('Running comparison_classification_report')

    report_variables = {
        'reportname': measurementsdata[0]['reportname']
    }
    metric_visualization = {}
    accuracy, mean_inference_time, ram_usage, names = [], [], [], []
    for data in measurementsdata:
        if 'target_inference_step' in data:
            inference_step = 'target_inference_step'
        elif 'protocol_inference_step' in data:
            inference_step = 'protocol_inference_step'
        else:
            log.warning("Placeholder")
            return ""

        eval_matrix = np.array(data['eval_confusion_matrix'])
        model_accuracy = np.trace(eval_matrix)/data['total']
        accuracy.append(model_accuracy)
        mean_inference_time.append(np.mean(data[inference_step]))
        ram_usage.append(np.mean(data['session_utilization_mem_percent']))
        names.append(data['modelname'])

        # Accuracy, precision, recall
        metric_visualization[data['modelname']] = [
            model_accuracy,
            np.mean(eval_matrix.diagonal()/np.sum(eval_matrix, axis=0)),
            np.mean(eval_matrix.diagonal()/np.sum(eval_matrix, axis=1))
        ]

    usepath = imgdir / "accuracy_vs_inference_time.png"
    draw_bubble_plot(
        usepath,
        "Accuracy vs Mean inference time",
        mean_inference_time,
        "Mean inference time [s]",
        accuracy,
        "Accuracy",
        ram_usage,
        names
    )
    report_variables['bubbleplotpath'] = str(
        usepath.relative_to(rootdir)
    )

    usepath = imgdir / "classification_metric_comparison.png"
    draw_radar_chart(
        usepath,
        "Metric comparison",
        metric_visualization,
        ["Accuracy", "Mean precision", "Mean recall"]
    )
    report_variables['radarchartpath'] = str(
        usepath.relative_to(rootdir)
    )
    report_variables['modelnames'] = names
    report_variables = {
        **report_variables,
        **metric_visualization,
    }

    with path(reports, 'classification_comparison.rst') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            report_variables
        )


def detection_report(
        measurementsdata: Dict[str, List],
        imgdir: Path,
        imgprefix: str,
        rootdir: Path) -> str:
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

    Returns
    -------
    str : content of the report in RST format
    """

    from kenning.datasets.helpers.detection_and_segmentation import \
        get_recall_precision, \
        compute_ap, \
        compute_map_per_threshold

    log.info(f'Running detection report for {measurementsdata["modelname"]}')

    lines = get_recall_precision(measurementsdata, 0.5)

    aps = []
    for line in lines:
        aps.append(compute_ap(line[0], line[1]))

    measurementsdata['mAP'] = compute_map_per_threshold(
        measurementsdata,
        [0.0]
    )[0]

    curvepath = imgdir / f'{imgprefix}recall_precision_curves.png'
    recall_precision_curves(
        str(curvepath),
        'Recall-Precision curves',
        lines,
        measurementsdata['class_names']
    )
    measurementsdata['curvepath'] = str(
        curvepath.relative_to(rootdir)
    )

    gradientpath = imgdir / f'{imgprefix}recall_precision_gradients.png'
    recall_precision_gradients(
        str(gradientpath),
        'Average precision plots',
        lines,
        measurementsdata['class_names'],
        aps,
        measurementsdata['mAP']
    )
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
    tpioupath = imgdir / f'{imgprefix}true_positive_iou_histogram.png'
    iouhistpath = imgdir / f'{imgprefix}histogram_tp_iou_values.png'

    true_positive_iou_histogram(
        str(tpioupath),
        'Average True Positive IoU values',
        tp_iou,
        measurementsdata['class_names'],
    )
    measurementsdata['tpioupath'] = str(
        tpioupath.relative_to(rootdir)
    )

    if len(all_tp_ious) > 0:
        true_positives_per_iou_range_histogram(
            str(iouhistpath),
            "Histogram of True Positive IoU values",
            all_tp_ious
        )
        measurementsdata['iouhistpath'] = str(
            iouhistpath.relative_to(rootdir)
        )

    thresholds = np.arange(0.2, 1.05, 0.05)
    mapvalues = compute_map_per_threshold(measurementsdata, thresholds)

    mappath = imgdir / f'{imgprefix}map.png'
    draw_plot(
        str(mappath),
        'mAP value change over objectness threshold values',
        'threshold',
        None,
        'mAP',
        None,
        [[thresholds, mapvalues]]
    )
    measurementsdata['mappath'] = str(
        mappath.relative_to(rootdir)
    )
    measurementsdata['max_mAP'] = max(mapvalues)
    measurementsdata['max_mAP_index'] = thresholds[np.argmax(mapvalues)].round(2)  # noqa: E501

    with path(reports, 'detection.rst') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            measurementsdata
        )


def comparison_detection_report(
        measurementsdata: List[Dict],
        imgdir: Path,
        rootdir: Path) -> str:
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

    Returns
    -------
    str : content of the report in RST format
    """
    log.info('Running comparison_detection_report')

    from kenning.datasets.helpers.detection_and_segmentation import \
        compute_map_per_threshold

    report_variables = {
        'reportname': measurementsdata[0]['reportname'],
        'modelnames': []
    }

    visualization_data = []
    for data in measurementsdata:
        thresholds = np.arange(0.2, 1.05, 0.05)
        mapvalues = compute_map_per_threshold(data, thresholds)
        visualization_data.append((thresholds, mapvalues))
        report_variables['modelnames'].append(data['modelname'])

    usepath = imgdir / "detection_map_thresholds.png"
    draw_plot(
        usepath,
        "mAP values comparison over different threshold values",
        'threshold',
        None,
        'mAP',
        None,
        visualization_data,
        report_variables['modelnames']
    )
    report_variables['mapcomparisonpath'] = str(
        usepath.relative_to(rootdir)
    )

    with path(reports, 'detection_comparison.rst') as reporttemplate:
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
        command: List[str] = []) -> str:
    """
    Generates an RST report based on Measurements data.

    The report is saved to the file in ``outputpath``.

    Parameters
    ----------
    reportname : str
        Name for the report
    data : List[Dict]
        Data for each model coming from the Measurements object,
        loaded i.e. from JSON files
    outputpath : Path
        Path to the RST file where the report will be saved
    imgdir : Path
        Path to the directory where the report plots should be stored
    report_types : List[str]
        List of report types that define the project, i.e.
        performance, classification
    rootdir : Path
        When the report is a part of a larger RST document (i.e. Sphinx docs),
        the rootdir parameter defines thte root directory of the document.
        It is used to compute relative paths in the document's references.
    command : List[str]
        Full command used to render this report, split into separate lines.
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

    with path(reports, 'header.rst') as reporttemplate:
        content = create_report_from_measurements(
            reporttemplate,
            header_data
        )

    for typ in report_types:
        for model_data in data:
            if len(data) > 1:
                imgprefix = model_data["modelname"] + "_"
            else:
                imgprefix = ""
            content += reptypes[typ](model_data, imgdir, imgprefix, rootdir)
        if len(data) > 1:
            content += comparereptypes[typ](data, imgdir, rootdir)

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
        help='Path to the output RST file',
        type=Path
    )
    parser.add_argument(
        '--root-dir',
        help='Path to root directory for documentation (paths in the RST file are relative to this directory)',  # noqa: E501
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
        measurementsdata.append(measurements)

    for measurements in measurementsdata:
        if 'build_cfg' in measurements:
            measurements['build_cfg'] = json.dumps(
                measurements['build_cfg'],
                indent=4
            ).split('\n')

    generate_report(
        args.reportname,
        measurementsdata,
        args.output,
        img_dir,
        args.report_types,
        args.root_dir,
        command
    )


if __name__ == '__main__':
    main(sys.argv)
