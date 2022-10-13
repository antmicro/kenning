from kenning.core.dataset import Dataset
from kenning.utils.logger import download_url

from pathlib import Path
import tarfile
import tempfile
import glob
import os
import re
from copy import deepcopy


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
                                if data_frame is not []:
                                    self.dataX.append(deepcopy(data_frame))
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

    def get_class_names(self):
        return [self.classnames[i] for i in self.classnames.keys()]

    def evaluate(self, predictions, truth):
        pass

    def get_input_mean_std(self):
        pass
