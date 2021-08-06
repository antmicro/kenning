from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from pathlib import Path
import tarfile
from kenning.utils.logger import download_url
import pandas as pd
import string
from copy import copy
from typing import Any, List, Tuple, Union


def dynamic_levenshtein_distance(a: str, b: str) -> int:
    """
    Computes the Levenshtein Distance metric between strings

    Parameters
    ----------
    a : str
        first string
    b : str
        second string

    Returns
    -------
    int : Levenshtein Distance
    """
    la, lb = len(a), len(b)
    dynamic_array = [0 for i in range(la+1)]
    dynamic_array = [copy(dynamic_array) for i in range(lb+1)]
    for i in range(1, la+1):
        dynamic_array[0][i] = i

    for i in range(1, lb+1):
        dynamic_array[i][0] = i

    for j in range(1, lb+1):
        for i in range(1, la+1):
            if a[i-1] == b[j-1]:
                cost = 0
            else:
                cost = 1
            dynamic_array[j][i] = min(
                dynamic_array[j][i-1] + 1,
                dynamic_array[j-1][i] + 1,
                dynamic_array[j-1][i-1] + cost
            )
    return dynamic_array[lb][la]


def char_eval(pred: str, gt: str) -> float:
    """
    Evaluates the prediction on a character basis

    The algorithm used to determine the distance between the
    strings is a dynamic programming implementation of the
    Levenshtein Distance metric
    Parameters
    ----------
    pred : str
        Prediction string
    gt : str
        Ground truth string

    Returns
    -------
    float : the ratio of the Levenshtein Distance to the ground truth length
    """
    # sanitize the Ground Truth from punctuation and uppercase letters
    gt = gt.translate(
        str.maketrans('', '', string.punctuation)
    ).lower().strip()
    pred = pred.strip()
    dld = dynamic_levenshtein_distance(pred, gt)
    return 1-float(dld)/float(len(gt))


class CommonVoiceDataset(Dataset):
    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            language: str = 'en',
            annotations_type: str = "test",
            sample_size: int = 1000,
            selection_method: str = 'accent'):
        self.language = language
        self.annotations_type = annotations_type
        self.selection_method = selection_method
        self.sample_size = sample_size
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
        self.dataY = [str(i) for i in self.dataY]

        if self.selection_method:
            if self.selection_method == 'length':
                metric = [len(i.strip().split(' ')) for i in self.dataY]
            elif self.selection_method == 'accent':
                metric = metadata['accent']
                metric = [str(i) for i in metric]
                # filter empty values
                new_dataX, new_dataY, new_metric = [], [], []
                assert len(metric) == len(self.dataX)
                for i, j, k in zip(self.dataX, self.dataY, metric):
                    if k != 'nan':
                        new_dataX.append(i)
                        new_dataY.append(j)
                        new_metric.append(k)
                self.dataX = new_dataX
                self.dataY = new_dataY
                metric = new_metric

        self.dataX = [
            str(Path(voice_folder / 'clips' / i).absolute().resolve())
            for i in self.dataX
        ]
        if self.selection_method is not None:
            assert self.sample_size <= len(self.dataX)
            self.select_representative_sample(metric)

    def select_representative_sample(self, metric: List[Any]):
        # select the representative sample of the metric
        from random import sample
        metric_sample = sample(metric, self.sample_size)
        tupled_data = zip(self.dataX, self.dataY, metric)
        tupled_data = list(tupled_data)
        sampled_dataset = []
        print(set(metric_sample))
        for i in set(metric_sample):
            how_many_of_length = metric_sample.count(i)
            matching = [x for x in tupled_data if x[2] == i]
            chosen = sample(matching, how_many_of_length)
            sampled_dataset += chosen
        self.dataX = []
        self.dataY = []
        for i, j, k in sampled_dataset:
            self.dataX.append(i)
            self.dataY.append((j, k))

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
        group.add_argument(
            "--sample-size",
            type=int,
            default=1000
        )
        group.add_argument(
            '--selection-metric',
            help='Metric to group the data',
            choices=['length', 'accent']
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.dataset_root,
            args.batch_size,
            args.language,
            args.annotations_type,
            args.sample_size,
            args.selection_method
        )

    def evaluate(
            self,
            predictions: List[str],
            ground_truths: List[Union[str, Tuple[str, Any]]]) -> Measurements:

        # This minimal score is based on the assumption that a 'good'
        # prediction for each word would have a Levenshtein Distance of at most
        # 2 and the average word length in english is around 5 characters
        # making 1-(2/5) = 0.6
        MIN_SCORE_FOUNDGT = 0.6
        measurements = Measurements()
        for pred, gt in zip(predictions, ground_truths):
            if isinstance(gt, tuple):
                gt, metric = gt
            else:
                metric = hash(gt)
            score = char_eval(pred, gt)
            found_gt = 1 if score >= MIN_SCORE_FOUNDGT else 0
            measurements.add_measurement(
                f'eval_stt/{metric}',  # not a detector therefore no confidence score is given # noqa: E501
                                       # so a new render report method will need to be added  # noqa: E501
                                       # for STT Models
                [[
                    float(found_gt),
                    float(score)
                ]],
                lambda: list()
            )
            measurements.accumulate(
                f'eval_gtcount/{metric}',
                1,
                lambda: 0
            )
        return measurements
