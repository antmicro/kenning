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
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

from edge_ai_tester.resources import reports
from edge_ai_tester.core.drawing import create_line_plot
from edge_ai_tester.core.drawing import draw_confusion_matrix
from edge_ai_tester.utils import logger
from edge_ai_tester.core.report import create_report_from_measurements


log = logger.get_logger()


def performance_report(reportname, measurementsdata: Dict[str, List], imgdir: Path):
    log.info('Running performance_report')

    if 'target_inference_step' in measurementsdata:
        log.info('Using target measurements for inference time')
        usepath = str(imgdir / f'{reportname}_inference_time.png')
        create_line_plot(
            usepath,
            f'Inference time for {reportname}',
            'Time', 's',
            'Inference time', 's',
            measurementsdata['target_inference_step_timestamp'],
            measurementsdata['target_inference_step'],
            skipfirst=True)
        measurementsdata['inferencetimepath'] = usepath
        measurementsdata['inferencetime'] = measurementsdata['target_inference_step']
    elif 'protocol_inference_step' in measurementsdata:
        log.info('Using protocol measurements for inference time')
        usepath = str(imgdir / f'{reportname}_inference_time.png')
        create_line_plot(
            usepath,
            f'Inference time for {reportname}',
            'Time', 's',
            'Inference time', 's',
            measurementsdata['protocol_inference_step_timestamp'],
            measurementsdata['protocol_inference_step'],
            skipfirst=True)
        measurementsdata['inferencetimepath'] = usepath
        measurementsdata['inferencetime'] = measurementsdata['protocol_inference_step']
    else:
        log.warning('No inference time measurements in the report')

    if 'session_utilization_mem_percent' in measurementsdata:
        log.info('Using target measurements memory usage percentage')
        usepath = str(imgdir / f'{reportname}_cpu_memory_usage.png')
        create_line_plot(
            usepath,
            f'Memory usage for {reportname}',
            'Time', 's',
            'Memory usage', '%',
            measurementsdata['session_utilization_timestamp'],
            measurementsdata['session_utilization_mem_percent'])
        measurementsdata['memusagepath'] = usepath
    else:
        log.warning('No memory usage measurements in the report')

    if 'session_utilization_gpu_mem_utilization' in measurementsdata:
        log.info('Using target measurements GPU memory usage percentage')
        usepath = str(imgdir / f'{reportname}_gpu_memory_usage.png')
        create_line_plot(
            usepath,
            f'GPU memory usage for {reportname}',
            'Time', 's',
            'GPU memory usage', 'MB',
            measurementsdata['session_utilization_gpu_timestamp'],
            measurementsdata['session_utilization_gpu_mem_utilization'])
        measurementsdata['gpumemusagepath'] = usepath
    else:
        log.warning('No GPU memory usage measurements in the report')

    if 'session_utilization_gpu_utilization' in measurementsdata:
        log.info('Using target measurements GPU utilization')
        usepath = str(imgdir / f'{reportname}_gpu_usage.png')
        create_line_plot(
            usepath,
            f'GPU Utilization for {reportname}',
            'Time', 's',
            'Utilization', '%',
            measurementsdata['session_utilization_gpu_timestamp'],
            measurementsdata['session_utilization_gpu_utilization'])
        measurementsdata['gpuusagepath'] = usepath
    else:
        log.warning('No GPU memory usage measurements in the report')

    with path(reports, 'performance.rst') as reportpath:
        return create_report_from_measurements(
            reportpath,
            measurementsdata
        )


def classification_report(reportname, measurementsdata: Dict[str, List], imgdir: Path):
    log.info('Running classification report')

    if 'eval_confusion_matrix' not in measurementsdata:
        log.error('Confusion matrix not present for classification report')
        return ''
    log.info('Using confusion matrix')
    confusionpath = str(imgdir / f'{reportname}_confusion_matrix.png')
    draw_confusion_matrix(
        measurementsdata['eval_confusion_matrix'],
        confusionpath,
        f'Confusion matrix',
        measurementsdata['class_names']
    )
    measurementsdata['confusionpath'] = confusionpath
    with path(reports, 'classification.rst') as reportpath:
        return create_report_from_measurements(
            reportpath,
            measurementsdata
        )


def generate_report(
        reportname: str,
        data: Dict,
        outputpath: Path,
        imgdir: Path,
        report_types: List[str]):
    reptypes = {
        'performance': performance_report,
        'classification': classification_report
    }

    content = ''
    data['reportname'] = [reportname]
    for typ in report_types:
        content += reptypes[typ](reportname, data, imgdir)

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
        args.report_types
    )


if __name__ == '__main__':
    main(sys.argv)
