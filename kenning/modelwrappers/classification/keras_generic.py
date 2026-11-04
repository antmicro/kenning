# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing a generic Keras classification model wrapper.
"""

import importlib
import sys
from typing import Any, List, Optional, Tuple

import numpy as np

from kenning.core.dataset import Dataset
from kenning.datasets.tabular_dataset import TabularDataset
from kenning.modelwrappers.classification import generic_templates
from kenning.modelwrappers.frameworks.tensorflow import TensorFlowWrapper
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI

if sys.version_info.minor < 9:
    from importlib_resources import files
else:
    from importlib.resources import files


class KerasGenericClassification(TensorFlowWrapper):
    """
    Wrapper for generic Keras classification models.
    """

    arguments_structure = {
        "batch_size": {
            "argparse_name": "--training-batch-size",
            "description": "The batch size for providing the input data",
            "type": int,
            "default": 1,
        },
        "model_source": {
            "argparse_name": "--model-source",
            "description": "Python script with functions for model generation",
            "type": str,
            "nullable": True,
            "default": None,
        },
        "learning_rate": {
            "argparse_name": "--learning-rate",
            "description": "Learning rate",
            "type": float,
            "default": 0.01,
        },
        "num_epochs": {
            "argparse_name": "--num-epochs",
            "description": "Number of epochs",
            "type": int,
            "default": 10,
        },
        "normalization_method": {
            "argparse_name": "--normalization",
            "description": "Method used for input data normalization",
            "type": str,
            "enum": ["mean", "min_max", "robust"],
            "default": "mean",
        },
    }

    default_dataset = TabularDataset

    default_template_path = (
        files(generic_templates) / "keras_anomaly_detection.py"
    )

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset] = None,
        batch_size: int = 1,
        from_file: bool = True,
        model_name: Optional[str] = None,
        model_source: Optional[str] = None,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        normalization_method: str = "mean",
    ):
        self.learning_rate = 0.001 if learning_rate is None else learning_rate
        self.num_epochs = 100 if num_epochs is None else num_epochs
        self.batch_size = batch_size

        if model_source is not None:
            self.model_source = model_source
        else:
            self.model_source = self.default_template_path

        if normalization_method == "mean":
            self.zero, self.scale = dataset.get_input_mean_std()
        elif normalization_method == "min_max":
            self.scale, self.zero = dataset.get_min_max_scaler_params()
        else:
            self.scale, self.zero = dataset.get_robust_scaler_params()

        self.scale = np.maximum(np.asarray(self.scale, dtype=np.float32), 1e-6)
        self.zero = np.asarray(self.zero, dtype=np.float32)

        KLogger.debug(f"Scale values: {self.scale} , zero values: {self.zero}")

        super().__init__(model_path, dataset, from_file, model_name)

    def prepare_model(self):
        if self.model_prepared:
            return None

        if self.from_file:
            self.load_model(self.model_path)
            self.model_prepared = True
        else:
            self.create_model_structure()
            self.model_prepared = True
            KLogger.debug(f"Saving model at {self.model_path}")
            self.save_model(self.model_path)

    def create_model_structure(self):
        if self.model_source is None:
            return

        spec = importlib.util.spec_from_file_location(
            "model", self.model_source
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        input_shape = self.get_input_shape(self.dataset)

        self.model = module.get_model(input_shape)

    def postprocess_outputs(self, y: List[Any]) -> List[np.ndarray]:
        predictions = np.argmax(np.asarray(y[0]), axis=-1).reshape(-1)

        return [predictions.astype(np.int64)]

    def preprocess_input(self, X: List[Any]) -> List[Any]:
        X = np.asarray(X[0], dtype=np.float32)
        X = np.arcsinh((X - self.zero) / self.scale)
        return [X.astype(np.float32)]

    def train_model(self):
        import tensorflow as tf
        from sklearn.utils import compute_class_weight

        (
            train_x,
            test_x,
            train_y,
            test_y,
        ) = self.dataset.train_test_split_representations()

        train_x = self.dataset.prepare_input_samples(train_x)[0]
        train_y = self.dataset.prepare_output_samples(train_y)[0]
        test_x = self.dataset.prepare_input_samples(test_x)[0]
        test_y = self.dataset.prepare_output_samples(test_y)[0]

        train_x = np.array(train_x, dtype=np.float32)
        train_y = np.array(train_y, dtype=np.float32)
        test_x = np.array(test_x, dtype=np.float32)
        test_y = np.array(test_y, dtype=np.float32)

        train_x = np.arcsinh((train_x - self.zero) / self.scale)
        test_x = np.arcsinh((test_x - self.zero) / self.scale)

        KLogger.debug(f"Train x shape: {train_x.shape}")
        KLogger.debug(f"Train y shape: {train_y.shape}")
        KLogger.debug(f"Test x shape: {test_x.shape}")
        KLogger.debug(f"Test y shape: {test_y.shape}")

        classes = np.unique(train_y)
        weights = compute_class_weight("balanced", classes=classes, y=train_y)

        KLogger.debug(f"Class weighs: {weights}")

        class_number = len(self.dataset.get_class_names())

        train_y = tf.one_hot(train_y, class_number)
        test_y = tf.one_hot(test_y, class_number)

        optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [tf.keras.metrics.F1Score(average="macro", name="F1Score")]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model.fit(
            train_x,
            train_y,
            validation_data=(test_x, test_y),
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            class_weight=dict(zip(classes, weights)),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_F1Score",
                    mode="max",
                    patience=50,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_F1Score",
                    mode="max",
                    factor=0.5,
                    patience=10,
                ),
            ],
        )

        self.model.summary()
        self.save_model(self.model_path)

    def get_io_specification_from_model(self):
        return self.get_io_specification_from_dataset(self.dataset)

    @classmethod
    def get_input_shape(cls, dataset: TabularDataset) -> Tuple:
        """
        Get input shape of the model from dataset.

        Parameters
        ----------
        dataset : TabularDataset
            Tabular dataset from which input shape is deduced.

        Returns
        -------
        Tuple
            Tuple with input's dimensions.
        """
        if getattr(dataset, "window_size", None) is not None:
            input_shape = (
                dataset.window_size,
                dataset.num_features,
            )
        else:
            input_shape = (
                (dataset.num_features,)
                if hasattr(dataset, "num_features")
                else (dataset.inputdims)
            )

        return input_shape

    @classmethod
    def get_io_specification_from_dataset(
        cls,
        dataset: TabularDataset,
    ):
        input_shape = cls.get_input_shape(dataset)

        return {
            "input": [
                {
                    "name": "input_1",
                    "shape": (dataset.batch_size, *input_shape),
                    "dtype": "float64",
                }
            ],
            "processed_input": [
                {
                    "name": "input_1",
                    "shape": (dataset.batch_size, *input_shape),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "probs",
                    "shape": (
                        dataset.batch_size,
                        len(dataset.get_class_names()),
                    ),
                    "dtype": "float32",
                }
            ],
            "processed_output": [
                {
                    "name": "class",
                    "shape": (dataset.batch_size,),
                    "dtype": "int64",
                }
            ],
        }

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        return cls.get_io_specification()
