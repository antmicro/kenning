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
            window_size: int = 128,
            window_shift: int = 128,
            batch_size: int = 1,
            download_dataset: bool = False):
        self.dataset_root = root
        self.batch_size = batch_size
        self.window_size = window_size
        self.window_shift = window_shift
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
                                    self.dataX.append(deepcopy(self.generate_padding(data_frame)))  # noqa: E501
                                    self.dataY.append(self.rev_class_id(i))
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
                self.window_size - 1,
                data_frame[0]
        )
        unpadded_len = len(pre_padding) + len(data_frame)
        post_len = (self.window_shift - (unpadded_len %
                    self.window_shift)) % self.window_shift

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
