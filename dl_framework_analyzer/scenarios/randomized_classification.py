import sys
import numpy as np
import importlib
import argparse

from ..core.dataset import Dataset
from ..utils.class_loader import load_class
from ..core.measurements import Measurements

class RandomizedClassificationDataset(Dataset):
    """
    Creates a sample classification dataset.

    It is a mock dataset with randomized inputs and outputs.
    """

    def __init__(
            self,
            root: str,
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
    
def run_classification(
    inferencetestercls,
    batch_size=1,
    samplescount=1000):
    dataset = RandomizedClassificationDataset(
        '',
        batch_size,
        samplescount
    )

    inferenceobj = inferencetestercls(dataset)

    inferenceobj.test_inference()
        
    
def main(argv):
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument(
        'inferencetestercls',
        help='InferenceTester-based class with inference implementation to import',
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

    args = parser.parse_args(argv[1:])

    cls = load_class(args.inferencetestercls)

    run_classification(
        cls,
        args.batch_size,
        args.num_samples)

if __name__ == '__main__':
    main(sys.argv)
