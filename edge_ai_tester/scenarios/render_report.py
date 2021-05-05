#!/usr/bin/env python

"""
A script that generates report files based on Measurements JSON output.

It requires providing the report type and JSON file to extract data from.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

from edge_ai_tester.resources import reports
from edge_ai_tester.core.drawing import time_series_plot
from edge_ai_tester.core.drawing import draw_confusion_matrix
from edge_ai_tester.utils import logger
from edge_ai_tester.core.report import create_report_from_measurements
import numpy as np


log = logger.get_logger()


def performance_report(
        reportname: str,
        measurementsdata: Dict[str, List],
        imgdir: Path,
        reportpath: Path,
        rootdir: Optional[Path] = None):
    """
    Creates performance section of the report.
    """
    log.info('Running performance_report')

    if rootdir is None:
        rootdir = reportpath.parent

    if 'target_inference_step' in measurementsdata:
        log.info('Using target measurements for inference time')
        usepath = imgdir / f'{reportpath.stem}_inference_time.png'
        time_series_plot(
            str(usepath),
            f'Inference time for {reportname}',
            'Time', 's',
            'Inference time', 's',
            measurementsdata['target_inference_step_timestamp'],
            measurementsdata['target_inference_step'],
            skipfirst=True)
        measurementsdata['inferencetimepath'] = str(
            usepath.relative_to(rootdir)
        )
        measurementsdata['inferencetime'] = \
            measurementsdata['target_inference_step']
    elif 'protocol_inference_step' in measurementsdata:
        log.info('Using protocol measurements for inference time')
        usepath = imgdir / f'{reportpath.stem}_inference_time.png'
        time_series_plot(
            str(usepath),
            f'Inference time for {reportname}',
            'Time', 's',
            'Inference time', 's',
            measurementsdata['protocol_inference_step_timestamp'],
            measurementsdata['protocol_inference_step'],
            skipfirst=True)
        measurementsdata['inferencetimepath'] = str(
            usepath.relative_to(rootdir)
        )
        measurementsdata['inferencetime'] = \
            measurementsdata['protocol_inference_step']
    else:
        log.warning('No inference time measurements in the report')

    if 'session_utilization_mem_percent' in measurementsdata:
        log.info('Using target measurements memory usage percentage')
        usepath = imgdir / f'{reportpath.stem}_cpu_memory_usage.png'
        time_series_plot(
            str(usepath),
            f'Memory usage for {reportname}',
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
        usepath = imgdir / f'{reportpath.stem}_cpu_usage.png'
        measurementsdata['session_utilization_cpus_percent_avg'] = [
            np.mean(cpus) for cpus in
            measurementsdata['session_utilization_cpus_percent']
        ]
        time_series_plot(
            str(usepath),
            f'Mean CPU usage for {reportname}',
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
        usepath = imgdir / f'{reportpath.stem}_gpu_memory_usage.png'
        time_series_plot(
            str(usepath),
            f'GPU memory usage for {reportname}',
            'Time', 's',
            'GPU memory usage', 'MB',
            measurementsdata['session_utilization_gpu_timestamp'],
            measurementsdata['session_utilization_gpu_mem_utilization'])
        measurementsdata['gpumemusagepath'] = str(
            usepath.relative_to(rootdir)
        )
    else:
        log.warning('No GPU memory usage measurements in the report')

    if 'session_utilization_gpu_utilization' in measurementsdata:
        log.info('Using target measurements GPU utilization')
        usepath = imgdir / f'{reportpath.stem}_gpu_usage.png'
        time_series_plot(
            str(usepath),
            f'GPU Utilization for {reportname}',
            'Time', 's',
            'Utilization', '%',
            measurementsdata['session_utilization_gpu_timestamp'],
            measurementsdata['session_utilization_gpu_utilization'])
        measurementsdata['gpuusagepath'] = str(
            usepath.relative_to(rootdir)
        )
    else:
        log.warning('No GPU memory usage measurements in the report')

    with path(reports, 'performance.rst') as reporttemplate:
        return create_report_from_measurements(
            reporttemplate,
            measurementsdata
        )


def classification_report(
        reportname: str,
        measurementsdata: Dict[str, List],
        imgdir: Path,
        reportpath: Path,
        rootdir: Optional[Path] = None):
    """
    Creates classification quality section of the report.
    """
    log.info('Running classification report')

    if rootdir is None:
        rootdir = reportpath.parent

    if 'eval_confusion_matrix' not in measurementsdata:
        log.error('Confusion matrix not present for classification report')
        return ''
    log.info('Using confusion matrix')
    confusionpath = imgdir / f'{reportpath.stem}_confusion_matrix.png'
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


def generate_report(
        reportname: str,
        data: Dict,
        outputpath: Path,
        imgdir: Path,
        report_types: List[str],
        rootdir: Optional[Path]):
    reptypes = {
        'performance': performance_report,
        'classification': classification_report
    }

    content = ''
    data['reportname'] = [reportname]
    for typ in report_types:
        content += reptypes[typ](reportname, data, imgdir, outputpath, rootdir)

    with open(outputpath, 'w') as out:
        out.write(content)


def main(argv):
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument(
        'measurements',
        help='Path to the JSON file with measurements',
        type=Path
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
        help='Path to root directory for documentation (paths in RST files are in respect to this directory)',  # noqa: E501
        type=Path,
        default=None
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
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    args = parser.parse_args(argv[1:])

    if args.img_dir is None:
        args.img_dir = args.output.parent / 'img'

    args.img_dir.mkdir(parents=True, exist_ok=True)

    with open(args.measurements, 'r') as measurements:
        measurementsdata = json.load(measurements)

    generate_report(
        args.reportname,
        measurementsdata,
        args.output,
        args.img_dir,
        args.report_types,
        args.root_dir
    )


if __name__ == '__main__':
    main(sys.argv)
