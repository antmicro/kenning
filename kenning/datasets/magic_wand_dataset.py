from typing import Tuple, Any, List
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
                self.dataY.append(label)

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
            data_frame: List,
            window_size: int = 128,
            window_shift: int = 128,
            noise_level: int = 20) -> List:
        """
        Generates neighbor-based padding around a given data frame

        Parameters
        ----------
        data_frame : List
            A frame of data to be padded
        window_size: int
            Size of the data window
        window_shift: int
            Shift of the data window
        noise_level : int
            Level of noise (window in which the neighbor data will vary in each
            axis)

        Returns
        -------
        List :
            The padded data frame
        """
        pre_padding = self._generate_padding(
            noise_level,
            abs(window_size - len(data_frame)) % window_size,
            data_frame[0]
        )
        unpadded_len = len(pre_padding) + len(data_frame)
        post_len = (-unpadded_len) % window_shift

        post_padding = self._generate_padding(
            noise_level,
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
            data_frame: List,
            window_size: int = 128) -> np.ndarray:
        """
        Splits given data sample into windows.

        Parameters
        ----------
        data_frame : List
            Data sample to be split
        window_size : int
            Size of the window

        Returns
        -------
        np.ndarray :
            Data sample split into windows
        """
        return np.array(np.array_split(
            data_frame, len(data_frame) // window_size, axis=0
        ))

    def train_test_split_representations(
            self,
            test_fraction: float = 0.25,
            seed: int = 1234,
            validation: bool = False,
            validation_fraction: float = 0.1) -> Tuple[List, ...]:
        """
        Splits the data representations into train dataset and test dataset.

        Parameters
        ----------
        test_fraction : float
            The fraction of data to leave for model validation
        seed : int
            The seed for random state
        validation: bool
            Whether to return a third, validation dataset
        validation_fraction: float
            The fraction (of the total size) that should be split out of
            the training set

        Returns
        -------
        Tuple[List, ...] :
            Data splitted into train, test and optionally validation subsets
        """
        from sklearn.model_selection import train_test_split
        dataXtrain, dataXtest, dataYtrain, dataYtest = train_test_split(
            self.dataX,
            self.dataY,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
            stratify=self.dataY
        )
        if validation:
            dataXtrain, dataXval, dataYtrain, dataYval = train_test_split(
                dataXtrain,
                dataYtrain,
                test_size=validation_fraction/(1 - test_fraction),
                random_state=seed,
                shuffle=True,
                stratify=dataYtrain
            )
            return (
                dataXtrain,
                dataXtest,
                dataYtrain,
                dataYtest,
                dataXval,
                dataYval
            )
        return (dataXtrain, dataXtest, dataYtrain, dataYtest)

    def prepare_tf_dataset(self, in_features: List, in_labels: List):
        """
        Converts data to tensorflow Dataset class.

        Parameters
        ----------
        in_features : List
            Dataset features
        in_labels : List
            Dataset labels

        Returns
        -------
        tensorflow.data.Dataset :
            Dataset in tensorflow format
        """
        assert len(in_features) == len(in_labels)
        from tensorflow.data import Dataset
        features = np.array(in_features)
        labels = np.array(in_labels)
        dataset = Dataset.from_tensor_slices(
            (features, labels.astype(np.int32))
        )
        return dataset
