# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains TFLite model for MagicWand dataset.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf

from kenning.cli.command_template import TRAIN
from kenning.core.dataset import Dataset
from kenning.datasets.magic_wand_dataset import MagicWandDataset
from kenning.modelwrappers.frameworks.tensorflow import TensorFlowWrapper
from kenning.utils.logger import KLogger, LoggerProgressBar
from kenning.utils.resource_manager import PathOrURI


class MagicWandModelWrapper(TensorFlowWrapper):
    """
    Model wrapper for Magic Wand model.
    """

    default_dataset = MagicWandDataset
    pretrained_model_uri = "kenning:///models/classification/magic_wand.h5"
    arguments_structure = {
        "window_size": {
            "argparse_name": "--window-size",
            "description": "Determines the size of single sample window",
            "default": 128,
            "type": int,
        },
        "batch_size": {
            "argparse_name": "--batch-size",
            "description": "Batch size for training. If not assigned, dataset batch size will be used.",  # noqa: E501
            "type": int,
            "default": 64,
            "subcommands": [TRAIN],
        },
        "learning_rate": {
            "description": "Learning rate for training",
            "type": float,
            "default": 0.001,
            "subcommands": [TRAIN],
        },
        "num_epochs": {
            "argparse_name": "--num-epochs",
            "description": "Number of epochs to train for",
            "type": int,
            "default": 50,
            "subcommands": [TRAIN],
        },
        "logdir": {
            "argparse_name": "--logdir",
            "description": "Path to the logging directory",
            "type": Path,
            "default": Path("/tmp/tflite_magic_wand_logs"),
            "subcommands": [TRAIN],
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool,
        model_name: Optional[str] = None,
        window_size: int = 128,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        logdir: Path = Path("/tmp/tflite_magic_wand_logs"),
    ):
        """
        Creates the Magic Wand model wrapper.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        dataset : Dataset
            The dataset to verify the inference.
        from_file : bool
            True if the model should be loaded from file.
        model_name : Optional[str]
            Name of the model used for the report
        window_size : int
            Size of single sample window.
        batch_size : int
            Batch size for training.
        learning_rate : float
            Learning rate for training.
        num_epochs : int
            Number of epochs to train for.
        logdir : Path
            Path to the logging directory.
        """
        super().__init__(model_path, dataset, from_file, model_name)
        self.window_size = window_size
        if dataset is not None:
            self.class_names = self.dataset.get_class_names()
            self.numclasses = len(self.class_names)
            self.save_io_specification(self.model_path)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.logdir = logdir

    @classmethod
    def _get_io_specification(
        cls, window_size, numclasses=-1, class_names=None, batch_size=1
    ):
        io_spec = {
            "input": [
                {
                    "name": "input_1",
                    "shape": [
                        (batch_size, window_size, 3),
                        (batch_size, window_size, 3, 1),
                    ],
                    "dtype": "float32",
                }
            ],
            "processed_input": [
                {
                    "name": "input_1",
                    "shape": (batch_size, window_size, 3, 1),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "out_layer",
                    "shape": (batch_size, numclasses),
                    "dtype": "float32",
                }
            ],
        }
        if class_names is not None:
            io_spec["output"][0]["class_names"] = class_names
        return io_spec

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        return cls._get_io_specification(json_dict["window_size"])

    def get_io_specification_from_model(self):
        if self.dataset:
            return self._get_io_specification(
                self.window_size,
                self.numclasses,
                self.class_names,
                self.dataset.batch_size,
            )

        return self._get_io_specification(
            self.window_size, self.numclasses, self.class_names
        )

    def preprocess_input(self, X: List[np.ndarray]) -> List[np.ndarray]:
        X = super().preprocess_input(X)
        if X[0].shape[-1] != 1:
            return [np.resize(X[0], (*X[0].shape, 1))]
        return X

    def prepare_model(self):
        if self.model_prepared:
            return None
        # https://github.com/tensorflow/tflite-micro/blob/dde75de483faa8d5e42b875cef3aaf26f6c63101/tensorflow/lite/micro/examples/magic_wand/train/train.py#L51
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    input_shape=(self.window_size, 3, 1), name="input_1"
                ),
                tf.keras.layers.Conv2D(
                    8, (4, 3), padding="same", activation="relu"
                ),
                tf.keras.layers.MaxPool2D((3, 3)),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Conv2D(
                    16, (4, 1), padding="same", activation="relu"
                ),
                tf.keras.layers.MaxPool2D((3, 1), padding="same"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(
                    4, activation="softmax", name="out_layer"
                ),
            ]
        )
        self.model.summary()

        if self.from_file:
            self.model.load_weights(self.model_path)
            self.model_prepared = True
        else:
            self.model_prepared = True
            self.save_model(self.model_path)

    def train_model(self):
        def convert_to_tf_dataset(features: List, labels: List):
            return tf.data.Dataset.from_tensor_slices(
                (
                    np.array(self.dataset.prepare_input_samples(features)[0]),
                    np.array(self.dataset.prepare_output_samples(labels)[0]),
                )
            )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        (
            train_data,
            test_data,
            train_labels,
            test_labels,
            val_data,
            val_labels,
        ) = self.dataset.train_test_split_representations(
            test_fraction=0.2, val_fraction=0.1
        )

        train_dataset = (
            convert_to_tf_dataset(train_data, train_labels)
            .batch(self.batch_size)
            .repeat()
        )
        val_dataset = convert_to_tf_dataset(val_data, val_labels).batch(
            self.batch_size
        )
        test_dataset = convert_to_tf_dataset(test_data, test_labels).batch(
            self.batch_size
        )

        with LoggerProgressBar(capture_stdout=True):
            self.model.fit(
                train_dataset,
                epochs=self.num_epochs,
                validation_data=val_dataset,
                steps_per_epoch=1000,
                validation_steps=int(
                    (len(val_data) - 1) / self.batch_size + 1
                ),
                callbacks=[
                    tf.keras.callbacks.TensorBoard(log_dir=str(self.logdir))
                ],
            )

        loss, acc = self.model.evaluate(test_dataset)
        with LoggerProgressBar(capture_stdout=True):
            pred = np.argmax(self.model.predict(test_dataset), axis=1)

        confusion = tf.math.confusion_matrix(
            labels=tf.constant(np.argmax(test_labels, axis=1)),
            predictions=tf.constant(pred),
            num_classes=4,
        )
        KLogger.info(f"confusion matrix:\n {confusion}")
        KLogger.info(f"loss: {loss}, accuracy: {acc}")
        self.model.save(self.model_path)
