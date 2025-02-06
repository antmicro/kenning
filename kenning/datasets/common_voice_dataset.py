# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Mozilla Common Voice Dataset wrapper.
"""

import string
import wave
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.utils.resource_manager import Resources, extract_tar


def dynamic_levenshtein_distance(a: str, b: str) -> int:
    """
    Computes the Levenshtein Distance metric between strings.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    int
        Levenshtein Distance.
    """
    la, lb = len(a), len(b)
    dynamic_array = np.zeros((lb + 1, la + 1), dtype=int)

    dynamic_array[0, :] = np.arange(1, la + 2)
    dynamic_array[:, 0] = np.arange(1, lb + 2)

    for j in range(1, lb + 1):
        for i in range(1, la + 1):
            cost = int(a[i - 1] != b[j - 1])
            dynamic_array[j, i] = min(
                dynamic_array[j, i - 1] + 1,
                dynamic_array[j - 1, i] + 1,
                dynamic_array[j - 1, i - 1] + cost,
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
        Prediction string.
    gt : str
        Ground truth string.

    Returns
    -------
    float
        The ratio of the Levenshtein Distance to the ground truth length.
    """
    # sanitize the Ground Truth from punctuation and uppercase letters
    gt = (
        gt.translate(str.maketrans("", "", string.punctuation)).lower().strip()
    )
    pred = pred.strip()
    dld = dynamic_levenshtein_distance(pred, gt)
    return 1 - float(dld) / float(len(gt))


def resample_wave(
    input_wave: np.ndarray, orig_sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    """
    Resamples provided wave.

    Parameters
    ----------
    input_wave : np.ndarray
        Wave to be resampled.
    orig_sample_rate : int
        Sample rate of provided wave.
    target_sample_rate : int
        Sample rate of the resampled wave.

    Returns
    -------
    np.ndarray
        Resampled wave.
    """
    return librosa.resample(
        input_wave.astype(np.float32),
        orig_sr=orig_sample_rate,
        target_sr=target_sample_rate,
    ).astype(np.int16)


def convert_mp3_to_wav(abspath: Path, subdir: str) -> Path:
    """
    Convert mp3 file at abspath to wav placed in subdir directory.

    Parameters
    ----------
    abspath : Path
        Absolute path to the .mp3 file.
    subdir : str
        A name of the subdirectory that will contain the converted file(s).

    Returns
    -------
    Path
        The string-typed path to the converted file.
    """
    from pydub import AudioSegment

    sound = AudioSegment.from_mp3(str(abspath))
    dst_folder = Path(abspath.parent / subdir)
    dst_folder.mkdir(parents=True, exist_ok=True)
    dst = str(Path(dst_folder / (abspath.stem + ".wav")))
    sound.export(dst, format="wav")
    return dst


def _init_resources():
    """
    Initializes Common Voice Dataset resources.
    """
    languages = ["en"]
    url_per_version = {
        "1": "cv-corpus-1/",
        "2": "cv-corpus-2/",
        "3": "cv-corpus-3/",
        "4": "cv-corpus-4-2019-12-10/",
        "5.1": "cv-corpus-5.1-2020-06-22/",
        "6.1": "cv-corpus-6.1-2020-12-11/",
        "7.0": "cv-corpus-7.0-2021-07-21/cv-corpus-7.0-2021-07-21-",
        "8.0": "cv-corpus-8.0-2022-01-19/cv-corpus-8.0-2022-01-19-",
        "9.0": "cv-corpus-9.0-2022-04-27/cv-corpus-9.0-2022-04-27-",
        "10.0": "cv-corpus-10.0-2022-07-04/cv-corpus-10.0-2022-07-04-",
        "11.0": "cv-corpus-11.0-2022-09-21/cv-corpus-11.0-2022-09-21-",
        "12.0": "cv-corpus-12.0-2022-12-07/cv-corpus-12.0-2022-12-07-",
    }
    url_format = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/{version_url}{language}.tar.gz"

    return Resources(
        {
            version: {
                language: url_format.format(
                    version_url=version_url, language=language
                )
                for language in languages
            }
            for version, version_url in url_per_version.items()
        }
    )


class CommonVoiceDataset(Dataset):
    """
    The Mozilla Common Voice Dataset.

    https://commonvoice.mozilla.org/

    The Common Voice dataset consists of a unique MP3 and corresponding text
    file. Many of the 9,283 recorded hours in the dataset also include
    demographic metadata like age, sex, and accent that can help train the
    accuracy of speech recognition engines.

    *License*: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication License.

    *Page*: `Common Voice site <https://commonvoice.mozilla.org/>`_.
    """

    resources = _init_resources()
    annotations_types = ["train", "validation", "test"]
    selection_methods = ["none", "length", "accent"]

    arguments_structure = {
        "language": {
            "argparse_name": "--language",
            "description": "Determines language of recordings",
            "default": "en",
            "enum": list(set(key[1] for key in resources.keys())),
        },
        "annotation_type": {
            "argparse_name": "--annotation-type",
            "description": "Type of annotations to load",
            "default": "test",
            "enum": annotations_types,
        },
        "sample_size": {
            "argparse_name": "--sample-size",
            "description": "Size of sample",
            "type": int,
            "default": 10,
        },
        "sample_rate": {
            "argparse_name": "--sample-rate",
            "description": "Recording sample rate",
            "type": int,
            "default": 16000,
        },
        "selection_method": {
            "argparse_name": "--selection-method",
            "description": "Method to group the data",
            "default": "accent",
            "enum": selection_methods,
        },
        "dataset_version": {
            "argparse_name": "--dataset-version",
            "description": "Version of the dataset",
            "default": "12.0",
            "enum": list(set(key[0] for key in resources.keys())),
        },
    }

    def __init__(
        self,
        root: Path,
        batch_size: int = 1,
        download_dataset: bool = True,
        force_download_dataset: bool = False,
        external_calibration_dataset: Optional[Path] = None,
        split_fraction_test: float = 0.2,
        split_fraction_val: Optional[float] = None,
        split_seed: int = 1234,
        dataset_percentage: float = 1,
        language: str = "en",
        annotations_type: str = "test",
        sample_size: int = 1000,
        sample_rate: int = 16000,
        selection_method: str = "accent",
        dataset_version: str = "12.0",
    ):
        """
        Prepares all structures and data required for providing data samples.

        Parameters
        ----------
        root : Path
            The path to the dataset data.
        batch_size : int
            The batch size.
        download_dataset : bool
            Downloads the dataset before taking any action. If the dataset
            files are already downloaded then they are not downloaded again.
        force_download_dataset : bool
            Forces dataset download.
        external_calibration_dataset : Optional[Path]
            Path to the external calibration dataset that can be used for
            quantizing the model. If it is not provided, the calibration
            dataset is generated from the actual dataset.
        split_fraction_test : float
            Default fraction of data to leave for model testing.
        split_fraction_val : Optional[float]
            Default fraction of data to leave for model validation.
        split_seed : int
            Default seed used for dataset split.
        dataset_percentage : float
            Use given percentage of the dataset.
        language : str
            Determines language of recordings.
        annotations_type : str
            Type of annotations to load.
        sample_size : int
            Size of sampled data.
        sample_rate : int
            Rate of recoding sample.
        selection_method : str
            Method to group the data.
        dataset_version : str
            Version of the dataset.
        """
        assert language in set(key[1] for key in self.resources.keys()), (
            f"Unsupported language {language}, should be one"
            f"of {set(key[1] for key in self.resources.keys())}"
        )
        assert annotations_type in self.annotations_types, (
            f"Unsupported annotations type {annotations_type}, should be one"
            f"of {self.annotations_types}"
        )
        assert selection_method in self.selection_methods, (
            f"Unsupported selection method {selection_method}, should be one"
            f"of {self.selection_methods}"
        )
        self.language = language
        self.annotations_type = annotations_type
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.selection_method = selection_method
        self.dataset_version = dataset_version
        super().__init__(
            root,
            batch_size,
            download_dataset,
            force_download_dataset,
            external_calibration_dataset,
            split_fraction_test,
            split_fraction_val,
            split_seed,
            dataset_percentage,
        )

    def download_dataset_fun(self):
        self.root.mkdir(parents=True, exist_ok=True)
        extract_tar(
            self.root, self.resources[self.dataset_version, self.language]
        )

    def prepare(self):
        # take the first found folder containing language subfolder inside
        # unpacked tar archive - it will be the dataset
        try:
            voice_folder = next(self.root.glob(f"*/{self.language}"))
        except StopIteration:
            raise FileNotFoundError
        metadata = pd.read_csv(
            voice_folder / f"{self.annotations_type}.tsv", sep="\t"
        )

        # since the data needs to be parsed into model's specific framework
        # and for example TorchAudio does only load from a file path, there is
        # no need to load data inside of the dataset and instead leave it to
        # the modelwrapper and it's later conversion functions.
        self.dataX, self.dataY = metadata["path"], metadata["sentence"]
        self.dataY = [str(y) for y in self.dataY]

        if self.selection_method == "length":
            metric_values = [len(i.strip().split()) for i in self.dataY]
        elif self.selection_method == "accent":
            if "accent" in metadata.columns:
                metadata_accent = metadata["accent"]
            elif "accents" in metadata.columns:
                metadata_accent = metadata["accents"]
            else:
                raise ValueError("Missing column with accents")

            metric_values = [str(i) for i in metadata_accent]
            # filter empty values
            new_dataX, new_dataY, new_metric_values = [], [], []
            assert len(metric_values) == len(self.dataX)
            for x, y, m in zip(self.dataX, self.dataY, metric_values):
                if m != "nan":
                    new_dataX.append(x)
                    new_dataY.append(y)
                    new_metric_values.append(m)
            self.dataX = new_dataX
            self.dataY = new_dataY
            metric_values = new_metric_values

        self.dataX = [
            str(Path(voice_folder / "clips" / x).resolve()) for x in self.dataX
        ]
        if self.selection_method != "none":
            assert self.sample_size <= len(self.dataX)
            self.select_representative_sample(metric_values)

    def prepare_input_samples(
        self, samples: List[str]
    ) -> List[List[np.ndarray]]:
        result = []
        for sample in samples:
            x = Path(sample)
            # check file type
            if x.suffix == ".mp3":
                audio_path = str(convert_mp3_to_wav(x, "waves"))
            else:
                audio_path = str(x)
            loaded_wav = wave.open(audio_path, "rb")
            audio = np.frombuffer(
                loaded_wav.readframes(loaded_wav.getnframes()), np.int16
            )
            if loaded_wav.getframerate() != self.sample_rate:
                audio = resample_wave(
                    audio, loaded_wav.getframerate(), self.sample_rate
                )
            result.append(audio)
        return [result]

    def select_representative_sample(self, metric_values: List[Any]):
        """
        Selects the representative sample from dataset based on provided metric
        values.

        Parameters
        ----------
        metric_values : List[Any]
            Metric value for each data sample.
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

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[Union[str, Tuple[str, Any]]],
    ) -> Measurements:
        # This minimal score is based on the assumption that a 'good'
        # prediction for each word would have a Levenshtein Distance of at most
        # 2 and the average word length in english is around 5 characters
        # making 1-(2/5) = 0.6
        MIN_SCORE_FOUND_GT = 0.6
        measurements = Measurements()
        currindex = self._dataindex - len(predictions)
        for pred, gt in zip(predictions, ground_truths):
            if isinstance(gt, tuple):
                gt, metric = gt
                type_suffix = f"/{metric}"
            else:
                type_suffix = ""
            score = char_eval(pred, gt)
            found_gt = 1 if score >= MIN_SCORE_FOUND_GT else 0
            # not a detector therefore no confidence score is given so a new
            # render report method will need to be added for STT Models
            measurements.add_measurement(
                f"eval_stt{type_suffix}",
                [
                    {
                        "found_ground_truth": float(found_gt),
                        "score": float(score),
                        "true_text": gt,
                        "predicted_text": pred,
                        "audio_path": str(
                            Path(
                                self.dataX[self._indexes[currindex]]
                            ).relative_to(self.root)
                        ),
                    }
                ],
                lambda: list(),
            )
            measurements.accumulate(f"eval_gtcount{type_suffix}", 1, lambda: 0)
        return measurements

    def train_test_split_representations(
        self,
        *args,
        stratify: bool = True,
        **kwargs,
    ) -> Tuple[List, ...]:
        return super().train_test_split_representations(
            *args,
            stratify=False,
            **kwargs,
        )

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        raise NotImplementedError

    def get_class_names(self) -> List[str]:
        raise NotImplementedError
