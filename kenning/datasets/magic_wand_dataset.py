from kenning.core.dataset import Dataset
from kenning.utils.logger import download_url

from pathlib import Path
import tarfile
import tempfile
import glob
import os
import re
from copy import deepcopy
import numpy as np


class MagicWandDataset(Dataset):
    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False):
        self.dataset_root = root
        self.batch_size = batch_size
        if download_dataset:
            self.download_dataset_fun()
        super().__init__(root, batch_size, download_dataset)

    def rev_class_id(self, classname: str) -> int:
        """
        Returns an integer representing a class based on a class name

        It generates a reversed dictionary from the self.classnames and
        it gets the ID that is assigned to that name
        Parameters
        ----------
        classname : str
            The name of the class for which the ID will be returned
        Returns
        -------
        Int : the class id
        """
        return {v: k for k, v in self.classnames.items()}[classname]

    def prepare(self):
        self.classnames = {
            0: "wing",
            1: "ring",
            2: "slope",
            3: "negative"
        }
        self.numclasses = 4
        self.dataX = []
        self.dataY = []
        for i in self.classnames.values():
            path = self.root / i
            for file in glob.glob(str(path / "*.txt")):
                data_frame = []
                with open(file) as f:
                    for line in f:
                        # match if line begins with '-', a number or a space
                        # and later shows only visible characters up to the end
                        if re.match(r"^[-0-9 ][\x20-\x7E]{1,}$", line):
                            if re.search("-,-,-", line):
                                if data_frame != []:
                                    self.dataX.append(deepcopy(data_frame))
                                    self.dataY.append(self.rev_class_id(i))
                                    data_frame = []
                            else:
                                data = line.rstrip()
                                data_frame.append(
                                    [float(i) for i in data.split(',')]
                                )
        # print(self.dataX)
        # print(self.dataY)
        assert len(self.dataX) == len(self.dataY)

    def download_dataset_fun(self):
        dataset_url = "http://download.tensorflow.org/models/tflite/magic_wand/data.tar.gz"  # noqa: E501
        with tempfile.TemporaryDirectory() as tempdir:
            tarpath = Path(tempdir) / "data.tar.gz"
            download_url(dataset_url, tarpath)
            data = tarfile.open(tarpath)
            data.extractall(self.dataset_root)
        # cleanup MacOS dotfiles with its internal metadata
        for macos_dotfile in glob.glob(str(self.dataset_root)+"/**/._*") + \
                glob.glob(str(self.dataset_root)+"/._*"):
            os.remove(macos_dotfile)

    def _generate_padding(self, noise_level, amount, neighbor: list) -> list:
        return [list(i) for i in np.round((np.random.rand(
            amount,
            3
        ) - 0.5) * noise_level, 1)+neighbor]

    def generate_padding(
            self,
            data_frame: list,
            window_size: int = 128,
            window_shift: int = 128,
            noise_level: int = 20) -> list:
        """
        Generates neighbor-based padding around a given data frame

        Parameters
        ----------
        data_frame : List
            A frame of data to be padded
        noise_level : Int
            Level of noise (window in which the
            neighbor data will vary in each axis)

        Returns
        -------
        List : The padded data frame

        """
        pre_padding = self._generate_padding(
                noise_level,
                abs(window_size - len(data_frame)) % window_size,
                data_frame[0]
        )
        unpadded_len = len(pre_padding) + len(data_frame)
        post_len = (window_shift - (unpadded_len %
                    window_shift)) % window_shift

        post_padding = self._generate_padding(
                noise_level,
                post_len,
                data_frame[-1]
        )
        return pre_padding+data_frame+post_padding

    def get_class_names(self):
        return [self.classnames[i] for i in self.classnames.keys()]

    def evaluate(self, predictions, truth):
        pass

    def get_input_mean_std(self):
        pass

    def split_sample_to_windows(self, data_frame, window_size=128):
        return np.array(np.array_split(
                data_frame,
                len(data_frame) // window_size, axis=0)
            )

    def train_test_split_representations(
            self,
            test_fraction: float = 0.25,
            seed: int = 1234,
            validation: bool = False,
            validation_fraction: float = 0.1):
        """
        Splits the data representations into train dataset and test dataset.

        Parameters
        ----------
        test_fraction : float
            The fraction of data to leave for model validation
        seed : int
            The seed for random state
        validation: bool
            Whehther to return a third, validation dataset
        validation_fraction: float
            The fraction (of the total size) that should be split out of
            the training set
        """
        from sklearn.model_selection import train_test_split
        dataXtrain, dataXtest, dataYtrain, dataYtest = train_test_split(
            self.dataX,
            self.dataY,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
        )
        if validation:
            dataXtrain, dataXvalid, dataYtrain, dataYvalid = train_test_split(
                self.dataX,
                self.dataY,
                test_size=test_fraction*(1-test_fraction),
                random_state=seed,
                shuffle=True,
            )
            return (
                dataXtrain,
                dataXtest,
                dataYtrain,
                dataYtest,
                dataXvalid,
                dataYvalid
            )
        return (dataXtrain, dataXtest, dataYtrain, dataYtest)

    def prepare_tf_dataset(self, in_features: list, in_labels: list):
        from tensorflow.data import Dataset
        for i, data in enumerate(in_features):
            in_features[i] = np.array(data)
        features = np.zeros((len(in_features), 128, 3))
        labels = np.zeros(len(in_features))
        for i, (data_frag, label) in enumerate(zip(in_features, in_labels)):
            pdata_frag = self.generate_padding(data_frag[0].tolist())
            features[i] = self.split_sample_to_windows(pdata_frag)[0]
            labels[i] = label
        dataset = Dataset.from_tensor_slices(
            (features, labels.astype("int32"))
        )
        return dataset
