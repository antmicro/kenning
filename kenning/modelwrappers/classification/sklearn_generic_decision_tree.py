# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Generic model-wrapper for a decision tree classifier. Designed to work with
TabularDataset.
"""

from math import prod
from typing import Any, Dict, List, Optional

import numpy as np

from kenning.cli.command_template import TRAIN
from kenning.core.dataset import Dataset, DatasetIterator
from kenning.modelwrappers.frameworks.sklearn import SKLearnModelWrapper
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


class SKLearnGenericDecisionTreeClassifier(SKLearnModelWrapper):
    """
    Generic model-wrapper for a decision tree classifier. Designed to work with
    TabularDataset.
    """

    arguments_structure = {
        "criterion": {
            "argparse_name": "--criterion",
            "description": "Criterion for tree optimization. Options: 'gini',"
            " 'entropy', 'log_loss'.",
            "type": str,
            "default": "gini",
            "subcommands": [TRAIN],
        },
        "splitter": {
            "argparse_name": "--splitter",
            "description": "Node splitting strategy. Options: 'random' - best,"
            " random split, 'best' - overall best split.",
            "type": str,
            "default": "best",
            "subcommands": [TRAIN],
        },
        "max_depth": {
            "argparse_name": "--max-depth",
            "description": "Max tree depth.",
            "type": int,
            "default": None,
            "nullable": True,
            "subcommands": [TRAIN],
        },
        "min_samples_leaf": {
            "argparse_name": "--min-samples-leaf",
            "description": "The minimum number of samples required to be at a"
            " leaf node.",
            "type": int,
            "default": 1,
            "subcommands": [TRAIN],
        },
        "min_samples_split": {
            "argparse_name": "--min-samples-split",
            "description": "The minimum number of samples required to split an"
            " internal node.",
            "type": int,
            "default": 2,
            "subcommands": [TRAIN],
        },
        "max_leaf_nodes": {
            "argparse_name": "--max-leaf-nodes",
            "description": "Grow a tree with in best-first fashion.",
            "type": int,
            "default": None,
            "nullable": True,
            "subcommands": [TRAIN],
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool = False,
        model_name: Optional[str] = None,
        dtype: str = "int16",
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        max_leaf_nodes: Optional[int] = None,
    ):
        super().__init__(model_path, dataset, from_file, model_name, dtype)
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_leaf_nodes = max_leaf_nodes

    def _train_for_n_samples(
        self, model: Any, n: int, dataset: DatasetIterator
    ) -> float:
        """
        Run training on the given model for first n samples in the dataset.

        Parameters
        ----------
        model: Any
            Model object
        n: int
            Number of samples
        dataset: DatasetIterator
            Data source

        Returns
        -------
        float
            Model accuracy
        """
        iterator = iter(dataset)
        X = []
        y = []
        for _ in range(n):
            _x, _y = next(iterator)
            X.append(np.asarray(_x).flatten().tolist())
            y.append(np.asarray(_y).flatten().tolist())
        model.fit(X, y)
        return model.score(X, y)

    def get_model(self):
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
        )
        # We have to train the model a bit to give it proper input/output shape
        self._train_for_n_samples(model, 1, self.dataset)
        return model

    def get_io_specification_from_model(self) -> Dict[str, List[Dict]]:
        x, y = next(self.dataset)
        return {
            "input": [
                {
                    "name": "input_1",
                    "shape": [
                        (1, 1, prod(np.asarray(x).shape)),
                    ],
                    "dtype": self.dtype,
                }
            ],
            "processed_input": [
                {
                    "name": "input_1",
                    "shape": (1, prod(np.asarray(x).shape)),
                    "dtype": self.dtype,
                }
            ],
            "output": [
                {
                    "name": "classification_result",
                    "shape": (1, prod(np.asarray(y).shape)),
                    "dtype": self.dtype,
                },
                {
                    "name": "class_probabilities",
                    "shape": (prod(np.asarray(y).shape), 1, 2),
                    "dtype": self.dtype,
                },
            ],
        }

    def _preprocess_input(
        self,
        X: List[Any],
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ) -> List[Any]:
        X = [np.expand_dims(np.asarray(X).flatten(), axis=0)]
        return X

    def run_inference(self, X: List[Any]) -> List[Any]:
        y = [self.model.predict(X[0]), self.model.predict_proba(X[0])]
        # If the model didn't see all possible values of all outputs during
        # training, shapes of the probability distribution it returns will be
        # invalid. It needs to be corrected.
        for i in range(len(y[1])):
            distribution = y[1][i]
            if distribution.shape == (1, 1):
                if y[0][0][i] == 1:
                    y[1][i] = np.asarray(
                        [[distribution[0][0], 1 - distribution[0][0]]]
                    )
                else:
                    y[1][i] = np.asarray(
                        [[1 - distribution[0][0], distribution[0][0]]]
                    )
        return y

    def train_model(self):
        KLogger.info(
            "Commencing training of for SKLearnDecisionTreeClassifier..."
        )
        iter = self.dataset.iter_train()
        score = self._train_for_n_samples(self.model, len(iter), iter)
        KLogger.info(f"Training finished, accuracy: {score}.")
