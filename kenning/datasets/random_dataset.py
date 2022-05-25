import numpy as np
from pathlib import Path

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements


class RandomizedClassificationDataset(Dataset):
    """
    Creates a sample randomized classification dataset.

    It is a mock dataset with randomized inputs and outputs.

    It can be used only for speed and utilization metrics, no quality metrics.
    """

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            samplescount: int = 1000,
            inputdims: list = (224, 224, 3),
            outputdims: list = (1000,)):
        """
        Creates randomized dataset.

        Parameters
        ----------
        root : Path
            Deprecated argument, not used in this dataset
        batch_size : int
            The size of batches of data delivered during inference
        samplescount : int
            The number of samples in the dataset
        inputdims : list
            The dimensionality of the inputs
        outputdims : list
            The dimensionality of the outputs
        """
        self.samplescount = samplescount
        self.inputdims = inputdims
        self.outputdims = outputdims
        super().__init__(root, batch_size)

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--num-samples',
            help='Number of samples to process',
            type=int,
            default=1000
        )
        group.add_argument(
            '--input-dims',
            help='Dimensionality of the inputs',
            type=int,
            nargs='+',
            default=[224, 224, 3]
        )
        group.add_argument(
            '--output-dims',
            help='Dimensionality of the outputs',
            type=int,
            nargs='+',
            default=[1000]
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.dataset_root,
            args.inference_batch_size,
            args.num_samples,
            args.input_dims,
            args.output_dims
        )

    def prepare(self):
        self.dataX = [[i for i in range(self.samplescount)] for _ in range(3)]
        self.dataY = [[i for i in range(self.samplescount)] for _ in range(3)]

    def download_dataset_fun(self):
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

    def calibration_dataset_generator(
            self,
            percentage: float = 0.25,
            seed: int = 12345):
        for _ in range(int(self.samplescount * percentage)):
            yield [np.random.randint(0, 255, size=self.inputdims)]
