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

    arguments_structure = {
        'samplescount': {
            'argparse_name': '--num-samples',
            'description': 'Number of samples to process',
            'type': int,
            'default': 1000
        },
        'numclasses': {
            'argparse_name': '--num-classes',
            'description': 'Number of classes in inputs',
            'type': int,
            'default': 3
        },
        'inputdims': {
            'argparse_name': '--input-dims',
            'description': 'Dimensionality of the inputs',
            'type': int,
            'default': [224, 224, 3],
            'is_list': True
        },
        'outputdims': {
            'argparse_name': '--output-dims',
            'description': 'Dimensionality of the outputs',
            'type': int,
            'default': [1000, ],
            'is_list': True
        }
    }

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            samplescount: int = 1000,
            numclasses: int = 3,
            inputdims: list = [224, 224, 3],
            outputdims: list = [1000, ],
            download_dataset: bool = False):
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
        numclasses : int
            The number of classes in the dataset
        inputdims : list
            The dimensionality of the inputs
        outputdims : list
            The dimensionality of the outputs
        """
        self.samplescount = samplescount
        self.inputdims = inputdims
        self.outputdims = outputdims
        self.numclasses = numclasses
        super().__init__(root, batch_size, download_dataset)

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.dataset_root,
            args.inference_batch_size,
            args.num_samples,
            args.numclasses,
            args.input_dims,
            args.output_dims
        )

    def get_class_names(self):
        return [str(i) for i in range(self.numclasses)]

    def get_input_mean_std(self):
        return (0.0, 1.0)

    def prepare(self):
        self.dataX = [[i for i in range(self.numclasses)] for j in range(self.samplescount)]    # noqa: E501
        self.dataY = [[i for i in range(self.numclasses)] for j in range(self.samplescount)]    # noqa: E501

    def download_dataset_fun(self):
        pass

    def prepare_input_samples(self, samples):
        result = []
        for sample in samples:
            np.random.seed(sample)
            result.append(np.random.randn(*self.inputdims))
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
