from kenning.core.dataset import Dataset
from pathlib import Path
import tarfile
from kenning.utils.logger import download_url
import pandas as pd


class CommonVoiceDataset(Dataset):
    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            language: str = 'english',
            annotations_type: str = "test"):
        self.language = language
        self.annotations_type = annotations_type
        super().__init__(root, batch_size, download_dataset)

    def download_dataset(self, lang: str = 'english'):
        self.root.mkdir(parents=True, exist_ok=True)
        # Mozilla made sure that machines cannot download this dataset
        # in it's most recent form. however, the version 6.1 has a
        # not-very-public download link that can be used to download them all

        # 7.0 has it blocked because GDPR for now
        # (I will do some additional digging to maybe find something else)
        url_format = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-6.1-2020-12-11/{}.tar.gz"  # noqa: E501

        with self.root as directory:
            tarpath = Path(directory) / 'dataset.tar.gz'
            download_url(url_format.format('en'), tarpath)
            tf = tarfile.open(tarpath)
            unpacked = (self.root / 'unpack')
            unpacked.mkdir(parents=True, exist_ok=True)
            tf.extractall(unpacked)

    def prepare(self):
        pass

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            "--dataset-root",
            type=Path,
            required=True
        )
        group.add_argument(
            "--batch-size",
            type=int,
            default=1
        )
        group.add_argument(
            "--language",
            type=str,
            choices=['english', 'polish'],
            default='english'
        )
        group.add_argument(
            '--annotations-type',
            help='Type of annotations to load',
            choices=['train', 'validated', 'test'],
            default='test'
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.dataset_root,
            args.batch_size,
            args.language
        )
