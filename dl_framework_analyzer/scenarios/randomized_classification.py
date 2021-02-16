"""
The sample benchmark for classification problem.

It works with Imagenet-trained models, provides 224x224x3 float tensors as
numpy arrays. It also expects 1000-element float vector as an output.

It provides random data, so it is not applicable for the quality measures.
This test is only for performance tests.
"""

import sys
import numpy as np
import argparse
from pathlib import Path
from importlib.resources import path

from dl_framework_analyzer.core.dataset import Dataset
from dl_framework_analyzer.utils.class_loader import load_class
from dl_framework_analyzer.core.measurements import Measurements
from dl_framework_analyzer.core.measurements import MeasurementsCollector
from dl_framework_analyzer.core.measurements import systemstatsmeasurements
from dl_framework_analyzer.utils import logger
from dl_framework_analyzer.core.report import create_report_from_measurements
from dl_framework_analyzer.resources import reports
from dl_framework_analyzer.core.drawing import create_line_plot


class RandomizedClassificationDataset(Dataset):
    """
    Creates a sample classification dataset.

    It is a mock dataset with randomized inputs and outputs.
    """

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            samplescount: int = 1000,
            inputdims: list = (224, 224, 3),
            outputdims: list = (1000,)):
        self.samplescount = samplescount
        self.inputdims = inputdims
        self.outputdims = outputdims
        super().__init__(root, batch_size)

    def prepare(self):
        self.dataX = [i for i in range(self.samplescount)]
        self.dataY = [i for i in range(self.samplescount)]

    def download_dataset(self):
        pass

    def prepare_input_samples(self, samples):
        result = []
        for sample in samples:
            np.random.seed(sample)
            result.append(np.random.randint(0, 255, size=self.inputdims))
        return result

    def prepare_output_samples(self, samples):
        result = []
        for sample in samples:
            np.random.seed(sample)
            result.append(np.random.rand(*self.outputdims))
        return result

    def evaluate(self, predictions, truth):
        return Measurements()


@systemstatsmeasurements('full_run_statistics')
def run_classification(
        modelwrappercls,
        batch_size: int = 1,
        samplescount: int = 1000):
    """
    Runs classification speed benchmark for a given model.

    Parameters
    ----------
    modelwrappercls : class
        The class variable that inherits from ModelWrapper and implements
        virtual methods for a given model
    batch_size : int
        The batch size of processing
    samplecount : int
        The number of input samples to process

    Returns
    -------
    Measurements : the benchmark results
    """
    dataset = RandomizedClassificationDataset(
        '',
        batch_size=batch_size,
        samplescount=samplescount
    )

    inferenceobj = modelwrappercls(dataset)

    frameworktuple = inferenceobj.get_framework_and_version()

    MeasurementsCollector.measurements += {
        'framework': frameworktuple[0],
        'version': frameworktuple[1]
    }

    return inferenceobj.test_inference()


def main(argv):
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument(
        'modelwrappercls',
        help='ModelWrapper-based class with inference implementation to import',  # noqa: E501
    )
    parser.add_argument(
        'output',
        help='The path to the output directory',
        type=Path)
    parser.add_argument(
        'reportname',
        help='The name of the report, used as RST name and resources prefix',
        type=str)
    parser.add_argument(
        '--resources-dir',
        help='The path to the directory with resources',
        type=Path
    )
    parser.add_argument(
        '--batch-size',
        help='The batch size of the inference',
        type=int,
        default=1
    )
    parser.add_argument(
        '--num-samples',
        help='Number of samples to process',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    args = parser.parse_args(argv[1:])

    if args.resources_dir is None:
        args.resources_dir = Path(args.output / 'img')

    args.output.mkdir(parents=True, exist_ok=True)
    args.resources_dir.mkdir(parents=True, exist_ok=True)

    logger.set_verbosity(args.verbosity)
    logger.get_logger()

    cls = load_class(args.modelwrappercls)

    run_classification(
        cls,
        args.batch_size,
        args.num_samples)

    reportname = args.reportname

    measurementsdata = MeasurementsCollector.measurements.data

    with path(reports, 'classification-performance.rst') as reportpath:
        batchtime = args.resources_dir / f'{reportname}-batchtime.png'
        memusage = args.resources_dir / f'{reportname}-memoryusage.png'
        gpumemusage = args.resources_dir / f'{reportname}-gpumemoryusage.png'
        gpuusage = args.resources_dir / f'{reportname}-gpuutilization.png'
        create_line_plot(
            batchtime,
            'Inference time for batches',
            'Time', 'ns',
            'Inference time', 'ns',
            measurementsdata['inference_step_timestamp'],
            measurementsdata['inference_step'])
        create_line_plot(
            memusage,
            'Memory usage over benchmark',
            'Time', 'ns',
            'Memory usage', '%',
            measurementsdata['full_run_statistics_timestamp'],
            measurementsdata['full_run_statistics_mem_percent'])
        create_line_plot(
            gpumemusage,
            'GPU Memory usage over benchmark',
            'Time', 'ns',
            'Memory usage', '%',
            measurementsdata['full_run_statistics_timestamp'],
            measurementsdata['full_run_statistics_gpu_mem_utilization'])
        create_line_plot(
            gpuusage,
            'GPU usage over benchmark',
            'Time', 'ns',
            'Memory usage', '%',
            measurementsdata['full_run_statistics_timestamp'],
            measurementsdata['full_run_statistics_gpu_utilization'])
        MeasurementsCollector.measurements += {
            'reportname': [reportname],
            'memusagepath': [memusage],
            'batchtimepath': [batchtime]
        }
        create_report_from_measurements(
            reportpath,
            measurementsdata,
            args.output / f'{reportname}.rst')


if __name__ == '__main__':
    main(sys.argv)
