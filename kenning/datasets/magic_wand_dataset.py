from kenning.core.dataset import Dataset
from kenning.utils.logger import download_url

from pathlib import Path
import tarfile
import tempfile
import glob
import os


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
            for file in glob.glob(str(path) / "*.txt"):
                filedata = []
                with open(file) as f:
                    for line in f:
                        pass
        pass

    def download_dataset_fun(self):
        dataset_url = "http://download.tensorflow.org/models/tflite/magic_wand/data.tar.gz"  # noqa: E501
        with tempfile.TemporaryDirectory() as tempdir:
            tarpath = Path(tempdir) / "data.tar.gz"
            download_url(dataset_url, tarpath)
            data = tarfile.open(tarpath)
            data.extractall(self.dataset_root)
        # cleanup MacOS dotfiles with metadata
        for macos_dotfile in glob.glob(str(self.dataset_root)+"/**/._*") + \
                glob.glob(str(self.dataset_root)+"/._*"):
            os.remove(macos_dotfile)
