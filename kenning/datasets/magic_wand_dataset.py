"""
The Tensorflow Magic Wand dataset
"""
from typing import Tuple, Any, List, Optional
from pathlib import Path
import tarfile
import tempfile
import glob
import os
import numpy as np

from kenning.core.dataset import Dataset
from kenning.utils.logger import download_url
from kenning.core.measurements import Measurements


class MagicWandDataset(Dataset):
    """
    The Tensorflow Magic Wand dataset

    It is a classification dataset with 4 classes representing different
    gestures captured by accelerometer and gyroscope.
    """

    arguments_structure = {
        'window_size': {
            'argparse_name': '--window-size',
            'description': 'Determines the size of single sample window',
            'default': 128,
            'type': int
        },
        'window_shift': {
            'argparse_name': '--window-shift',
            'description': 'Determines the shift of single sample window',
            'default': 128,
            'type': int
        },
        'noise_level': {
            'argparse_name': '--noise-level',
            'description': 'Determines the level of noise added as padding',
            'default': 20,
            'type': int
        }
    }

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            external_calibration_dataset: Optional[Path] = None,
            window_size: int = 128,
            window_shift: int = 128,
            noise_level: int = 20):
        """
        Prepares all structures and data required for providing data samples.

        Parameters
        ----------
        root : Path
            The path to the dataset data
        batch_size : int
            The batch size
        download_dataset : bool
            True if dataset should be downloaded first
        external_calibration_dataset : Optional[Path]
            Path to the external calibration dataset that can be used for
            quantizing the model. If it is not provided, the calibration
            dataset is generated from the actual dataset.
        windows_size : int
            Size of single sample window
        window_shift : int
            Shift of single sample window
        noise_level : int
            Noise level of padding added to sample
        """
        self.window_size = window_size
        self.window_shift = window_shift
        self.noise_level = noise_level
        super().__init__(
            root,
            batch_size,
            download_dataset,
            external_calibration_dataset
        )

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.dataset_root,
            args.inference_batch_size,
            args.download_dataset,
            args.external_calibration_dataset,
            args.window_size,
            args.window_shift,
            args.noise_level
        )

    def rev_class_id(self, classname: str) -> int:
        """
        Returns an integer representing a class based on a class name

        It generates a reversed dictionary from the self.classnames and
        it gets the ID that is assigned to that name.

        Parameters
        ----------
        classname : str
            The name of the class for which the ID will be returned
        Returns
        -------
        Int :
            the class id
        """
        return {v: k for k, v in self.classnames.items()}[classname]

    def prepare(self):
        self.classnames = {
            0: 'wing',
            1: 'ring',
            2: 'slope',
            3: 'negative'
        }
        self.numclasses = 4
        tmp_dataX = []
        tmp_dataY = []
        for class_name in self.classnames.values():
            path = self.root / class_name
            if not path.is_dir():
                raise FileNotFoundError
            class_id = self.rev_class_id(class_name)
            for file in glob.glob(str(path / '*.txt')):
                data_frame = []
                with open(file) as f:
                    for line in f:
                        line_split = line.strip().split(',')
                        if len(line_split) != 3:
                            continue
                        try:
                            values = [float(i) for i in line_split]
                            data_frame.append(values)
                        except ValueError:
                            if data_frame:
                                tmp_dataX.append(data_frame)
                                tmp_dataY.append(class_id)
                                data_frame = []

        self.dataX = []
        self.dataY = []
        for data, label in zip(tmp_dataX, tmp_dataY):
            padded_data = np.array(self.split_sample_to_windows(
                self.generate_padding(data)
            ))
            for sample in padded_data:
                self.dataX.append(sample)
                self.dataY.append(np.eye(self.numclasses)[label])

        assert len(self.dataX) == len(self.dataY)

    def download_dataset_fun(self):
        dataset_url = r'http://download.tensorflow.org/models/tflite/magic_wand/data.tar.gz'  # noqa: E501
        with tempfile.TemporaryDirectory() as tempdir:
            tarpath = Path(tempdir) / 'data.tar.gz'
            download_url(dataset_url, tarpath)
            data = tarfile.open(tarpath)
            data.extractall(self.root)

        # cleanup MacOS-related hidden metadata files present in the dataset
        for macos_dotfile in (
            glob.glob(str(self.root) + '/**/._*')
            + glob.glob(str(self.root) + '/._*')
        ):
            os.remove(macos_dotfile)

    def _generate_padding(
            self,
            noise_level: int,
            amount: int,
            neighbor: List) -> List:
        """
        Generates noise padding of given length.

        Parameters
        ----------
        noise_level : int
            Level of generated noise
        amount : int
            Length of generated noise
        neighbor : List
            Neighbor data

        Returns
        -------
        List :
            Neighbor data with noise padding
        """
        padding = (np.round((np.random.rand(amount, 3) - 0.5)*noise_level, 1)
                   + neighbor)
        return [list(i) for i in padding]

    def generate_padding(
            self,
            data_frame: List) -> List:
        """
        Generates neighbor-based padding around a given data frame

        Parameters
        ----------
        data_frame : List
            A frame of data to be padded

        Returns
        -------
        List :
            The padded data frame
        """
        pre_padding = self._generate_padding(
            self.noise_level,
            abs(self.window_size - len(data_frame)) % self.window_size,
            data_frame[0]
        )
        unpadded_len = len(pre_padding) + len(data_frame)
        post_len = (-unpadded_len) % self.window_shift

        post_padding = self._generate_padding(
            self.noise_level,
            post_len,
            data_frame[-1]
        )
        return pre_padding + data_frame + post_padding

    def get_class_names(self) -> List[str]:
        return list(self.classnames.values())

    def evaluate(self, predictions, truth):
        confusion_matrix = np.zeros((self.numclasses, self.numclasses))
        for prediction, label in zip(predictions, truth):
            confusion_matrix[np.argmax(label), np.argmax(prediction)] += 1
        measurements = Measurements()
        measurements.accumulate(
            'eval_confusion_matrix',
            confusion_matrix,
            lambda: np.zeros((self.numclasses, self.numclasses))
        )
        measurements.accumulate('total', len(predictions), lambda: 0)
        return measurements

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        return (
            np.array([-219.346, 198.207, 854.390]),
            np.array([430.269, 326.288, 447.666])
        )

    def split_sample_to_windows(
            self,
            data_frame: List) -> np.ndarray:
        """
        Splits given data sample into windows.

        Parameters
        ----------
        data_frame : List
            Data sample to be split

        Returns
        -------
        np.ndarray :
            Data sample split into windows
        """
        return np.array(np.array_split(
            data_frame, len(data_frame) // self.window_size, axis=0
        ))
