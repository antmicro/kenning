# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Classes and methods for CNN Dailymail dataset.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.utils.resource_manager import Resources


class CNNDailymailDataset(Dataset):
    """
    Dataset wrapper for cnn_dailymail that utilizes
    HuggingFace preprocessed dataset.

    CNNDailyMail is a non-anonymized summarization dataset that consists of
    online news articles from CNN and Daily Mail. Each article is paired with
    highlights written by the original editors.

    https://huggingface.co/datasets/cnn_dailymail

    *License*: Apache License, Version 2.0

    *Page*: https://huggingface.co/datasets/cnn_dailymail.

    Attributes
    ----------
    ds : Optional[datasets.Dataset]
        Huggingface dataset wrapper that is indexed by three attributes:
        `id`, `highlight` and `article`.
    """

    system_message = (
        "Your task is to generate a short summary of the article below. "
        + "Keep the summarization concise and under thirty words."
    )

    resources = Resources(
        {
            "cnn_dailymail": "hf://datasets/cnn_dailymail",
        }
    )

    arguments_structure = {
        "gather_predictions": {
            "description": "Determines whether returned evaluations should include target and predicted sentences",  # noqa: E501
            "type": bool,
            "default": False,
        },
        "metrics": {
            "description": "Types of rouge metrics gathered during evaluation",
            "type": str,
            "default": ["rouge1", "rouge2", "rouge3", "rougeL"],
            "is_list": True,
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
        gather_predictions: bool = False,
        metrics: List[str] = ["rouge1", "rouge2", "rouge3", "rougeL"],
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
        gather_predictions : bool
            Determines whether returned evaluations should
            include target and predicted sentences
        metrics : List[str]
            Types of rouge metrics gathered during evaluation
        """
        self.gather_predictions = gather_predictions
        self.metrics = metrics

        self.ds = None

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

    def prepare_input_samples(self, samples: List[int]) -> List[List[str]]:
        result = []
        for id in samples:
            result.append(self.ds[id]["article"])
        return [result]

    def prepare_output_samples(self, samples: List[int]) -> List[List[str]]:
        result = []
        for id in samples:
            result.append(self.ds[id]["highlights"])
        return [result]

    def prepare(self):
        from datasets import load_from_disk

        self.ds = load_from_disk(str(self.root.resolve()))

        # Input and output data are lists of Ids
        self.dataX = list(range(self.ds.num_rows))
        self.dataY = list(range(self.ds.num_rows))

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        # Text summarization is not a numerical task.
        # Every input is a list of words, that are later tokenized,
        # therefore, mean and std are not defined.
        raise NotImplementedError

    def get_class_names(self) -> List[str]:
        # Text summarization is not a classification task.
        # There are no classes to be classified.
        raise NotImplementedError

    def download_dataset_fun(self):
        from datasets import load_dataset

        dataset_path = self.resources["cnn_dailymail"]
        ds = load_dataset(
            path=str(dataset_path), name="3.0.0", split="train+test+validation"
        )
        ds.save_to_disk(str(self.root))

    def get_data(self) -> Tuple[List, List]:
        raise NotImplementedError

    def get_data_unloaded(self) -> Tuple[List, List]:
        raise NotImplementedError

    def train_test_split_representations(
        self,
        test_fraction: Optional[float] = None,
        val_fraction: Optional[float] = None,
        seed: Optional[int] = None,
        stratify: bool = True,
        append_index: bool = False,
    ) -> Tuple[List, ...]:
        """
        Splits the data representations into train dataset and test dataset.

        CNNDailyMail is not a classification dataset. Its `dataY`
        does not convey information about labels, but about rows in the
        dataset, hence `dataY` values are unique.
        Because of that `stratify` argument of sklearn function
        `train_test_split` has to be fixed to False.

        Parameters
        ----------
        test_fraction : Optional[float]
            The fraction of data to leave for model testing.
        val_fraction : Optional[float]
            The fraction of data to leave for model validation.
        seed : Optional[int]
            The seed for random state.
        stratify : bool
            Whether to stratify the split.
        append_index : bool
            Whether to return the indices of the split. If True, the returned
            tuple will have indices appended at the end.
            For example, if the split is (X_train, X_test, y_train, y_test),
            the returned tuple will be (X_train, X_test, y_train, y_test,
            train_indices, test_indices).

        Returns
        -------
        Tuple[List, ...]
            Split data into train, test and optionally validation subsets.
        """
        return super().train_test_split_representations(
            test_fraction,
            val_fraction,
            seed,
            False,
            append_index,
        )

    def evaluate(
        self, predictions: List[List[str]], truth: List[List[str]]
    ) -> Measurements:
        from rouge_score import rouge_scorer

        measurements = Measurements()

        for p, t in zip(predictions[0], truth[0]):
            scorer = rouge_scorer.RougeScorer(self.metrics, use_stemmer=True)
            scores = scorer.score(t, p)

            for metric in self.metrics:
                measurements.accumulate(
                    metric, scores[metric].precision, lambda: 0
                )

            if self.gather_predictions:
                measurements.add_measurement(
                    "predictions",
                    [
                        {
                            "target": t,
                            "prediction": p,
                        }
                    ],
                )
        measurements.accumulate("total", len(truth[0]), lambda: 0)
        return measurements
