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
            language: str = 'en',
            annotations_type: str = "test"):
        self.language = language
        self.annotations_type = annotations_type
        super().__init__(root, batch_size, download_dataset)

    def download_dataset(self, lang: str = 'en'):
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
            unpacked = (self.root / 'unpacked')
            unpacked.mkdir(parents=True, exist_ok=True)
            tf.extractall(unpacked)

    def prepare(self):
        voice_folder = Path(self.root / "unpacked").glob("*")
        # take the first found folder inside unpacked tar archive
        # it will be the dataset
        voice_folder = Path([i for i in voice_folder][0] / self.language)
        metadata = pd.read_csv(
            Path(voice_folder / "{}.tsv".format(self.annotations_type)),
            sep='\t'
        )

        # since the data needs to be parsed into model's specific framework
        # and for example TorchAudio does only load from a file path, there is
        # no need to load data inside of the dataset and instead leave it to
        # the modelwrapper and it's later conversion functions.
        self.dataX, self.dataY = metadata['path'], metadata['sentence']
        self.dataX = [
            str(Path(voice_folder / 'clips' / i).absolute().resolve())
            for i in self.dataX
        ]
        self.dataY = [str(i) for i in self.dataY]

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
            choices=['en', 'pl'],
            default='en'
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
            args.language,
            args.annotations_type
        )

    def evaluate(self, prediction, ground_truth):
        # Due to poor documentation of the torchaudio framework, there
        # is no way to determine what the exact shape and structure
        # of the prediction is.
        print(prediction)
        print(ground_truth)
