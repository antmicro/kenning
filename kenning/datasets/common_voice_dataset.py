# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Mozilla Common Voice Dataset wrapper
"""

from typing import Any, List, Tuple, Union, Optional
from pathlib import Path
import tarfile
import tempfile
import string
import pandas as pd

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.utils.logger import download_url


def dynamic_levenshtein_distance(a: str, b: str) -> int:
    """
    Computes the Levenshtein Distance metric between strings.

    Parameters
    ----------
    a : str
        First string
    b : str
        Second string

    Returns
    -------
    int :
        Levenshtein Distance
    """
    la, lb = len(a), len(b)
    dynamic_array = [[0 for _ in range(la+1)] for _ in range(lb+1)]
    for i in range(1, la+1):
        dynamic_array[0][i] = i

    for i in range(1, lb+1):
        dynamic_array[i][0] = i

    for j in range(1, lb+1):
        for i in range(1, la+1):
            cost = int(a[i-1] != b[j-1])
            dynamic_array[j][i] = min(
                dynamic_array[j][i-1] + 1,
                dynamic_array[j-1][i] + 1,
                dynamic_array[j-1][i-1] + cost
            )
    return dynamic_array[lb][la]


def char_eval(pred: str, gt: str) -> float:
    """
    Evaluates the prediction on a character basis.

    The algorithm used to determine the distance between the
    strings is a dynamic programming implementation of the
    Levenshtein Distance metric.

    Parameters
    ----------
    pred : str
        Prediction string
    gt : str
        Ground truth string

    Returns
    -------
    float :
        the ratio of the Levenshtein Distance to the ground truth length
    """
    # sanitize the Ground Truth from punctuation and uppercase letters
    gt = gt.translate(
        str.maketrans('', '', string.punctuation)
    ).lower().strip()
    pred = pred.strip()
    dld = dynamic_levenshtein_distance(pred, gt)
    return 1 - float(dld)/float(len(gt))


class CommonVoiceDataset(Dataset):

    languages = ['en', 'pl']
    annotations_types = ['train', 'validation', 'test']
    selection_methods = ['length', 'accent']

    arguments_structure = {
        'language': {
            'argparse_name': '--language',
            'description': 'Determines language of recordings',
            'default': 'en',
            'enum': languages
        },
        'annotation_type': {
            'argparse_name': '--annotation-type',
            'description': 'Type of annotations to load',
            'default': 'test',
            'enum': annotations_types
        },
        'sample_size': {
            'argparse_name': '--sample-size',
            'description': 'Size of sample',
            'type': int,
            'default': 10
        },
        'selection_method': {
            'argparse_name': '--selection-method',
            'description': 'Method to group the data',
            'default': 'accent',
            'enum': selection_methods
        }
    }

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            external_calibration_dataset: Optional[Path] = None,
            language: str = 'en',
            annotations_type: str = 'test',
            sample_size: int = 1000,
            selection_method: str = 'accent'):
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
        language : str
            Determines language of recordings.
        annotation_type : str
            Type of annotations to load.
        sample_size : int
            Size of sampled data.
        selection_method : str
            Method to group the data.
        """
        assert language in ['en', 'pl'], (
            f'Unsupported language {language}, should be one'
            f'of {self.languages}')
        assert annotations_type in self.annotations_types, (
            f'Unsupported annotations type {annotations_type}, should be one'
            f'of {self.annotations_types}')
        assert selection_method in ['length', 'accent'], (
            f'Unsupported selection method {selection_method}, should be one'
            f'of {self.selection_methods}')
        self.language = language
        self.annotations_type = annotations_type
        self.sample_size = sample_size
        self.selection_method = selection_method
        super().__init__(
            root,
            batch_size,
            download_dataset,
            external_calibration_dataset
        )

    def download_dataset_fun(self):
        self.root.mkdir(parents=True, exist_ok=True)
        # Mozilla made sure that machines cannot download this dataset
        # in it's most recent form. however, the version 6.1 has a
        # not-very-public download link that can be used to download them all

        # 7.0 has it blocked because GDPR for now
        # TODO: find a way to obtain dataset version 7.0
        url_format = 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-6.1-2020-12-11/{}.tar.gz'  # noqa: E501

        with tempfile.TemporaryDirectory() as tmpdir:
            tarpath = Path(tmpdir) / 'dataset.tar.gz'
            download_url(url_format.format(self.language), tarpath)
            tf = tarfile.open(tarpath)
            tf.extractall(self.root)

    def prepare(self):
        # take the first found folder containing language subfolder inside
        # unpacked tar archive - it will be the dataset
        voice_folder = next(self.root.glob(f'*/{self.language}'))
        metadata = pd.read_csv(
            voice_folder / f'{self.annotations_type}.tsv',
            sep='\t'
        )

        # since the data needs to be parsed into model's specific framework
        # and for example TorchAudio does only load from a file path, there is
        # no need to load data inside of the dataset and instead leave it to
        # the modelwrapper and it's later conversion functions.
        self.dataX, self.dataY = metadata['path'], metadata['sentence']
        self.dataY = [str(y) for y in self.dataY]

        if self.selection_method:
            if self.selection_method == 'length':
                metric_values = [len(i.strip().split()) for i in self.dataY]
            elif self.selection_method == 'accent':
                metric_values = [str(i) for i in metadata['accent']]
                # filter empty values
                new_dataX, new_dataY, new_metric_values = [], [], []
                assert len(metric_values) == len(self.dataX)
                for x, y, m in zip(self.dataX, self.dataY, metric_values):
                    if m != 'nan':
                        new_dataX.append(x)
                        new_dataY.append(y)
                        new_metric_values.append(m)
                self.dataX = new_dataX
                self.dataY = new_dataY
                metric_values = new_metric_values

        self.dataX = [
            str(Path(voice_folder / 'clips' / x).resolve())
            for x in self.dataX
        ]
        if self.selection_method is not None:
            assert self.sample_size <= len(self.dataX)
            self.select_representative_sample(metric_values)

    def select_representative_sample(self, metric_values: List[Any]):
        """
        Returns the representative sample from dataset based on provided metric
        values.

        Parameters
        ----------
        metric_values : List[Any]
            Metric value for each data sample
        """
        # select the representative sample of the metric
        from random import sample
        assert len(self.dataX) == len(metric_values)
        metric_values_sample = sample(metric_values, self.sample_size)
        tupled_data = list(zip(self.dataX, self.dataY, metric_values))
        sampled_dataset = []
        for i in set(metric_values_sample):
            how_many_of_length = metric_values_sample.count(i)
            matching = [x for x in tupled_data if x[2] == i]
            chosen = sample(matching, how_many_of_length)
            sampled_dataset += chosen
        self.dataX = []
        self.dataY = []
        for x, *y in sampled_dataset:
            self.dataX.append(x)
            self.dataY.append(y)

    @classmethod
    def from_argparse(cls, args):
        return cls(
            root=args.dataset_root,
            batch_size=args.batch_size,
            download_dataset=args.download_dataset,
            language=args.language,
            annotations_type=args.annotations_type,
            sample_size=args.sample_size,
            selection_method=args.selection_method
        )

    def evaluate(
            self,
            predictions: List[str],
            ground_truths: List[Union[str, Tuple[str, Any]]]) -> Measurements:
        # This minimal score is based on the assumption that a 'good'
        # prediction for each word would have a Levenshtein Distance of at most
        # 2 and the average word length in english is around 5 characters
        # making 1-(2/5) = 0.6
        MIN_SCORE_FOUND_GT = 0.6
        measurements = Measurements()
        for pred, gt in zip(predictions, ground_truths):
            if isinstance(gt, tuple):
                gt, metric = gt
            else:
                metric = hash(gt)
            score = char_eval(pred, gt)
            found_gt = 1 if score >= MIN_SCORE_FOUND_GT else 0
            # not a detector therefore no confidence score is given so a new
            # render report method will need to be added for STT Models
            measurements.add_measurement(
                f'eval_stt/{metric}',
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
