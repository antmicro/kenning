# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for TensorFlow optimizers.
"""

import tensorflow as tf
import zipfile

from typing import List, Literal, Tuple, Optional

from kenning.core.dataset import Dataset
from kenning.core.optimizer import Optimizer
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


class TensorFlowOptimizer(Optimizer):
    """
    The TensorFlow optimizer.
    """

    arguments_structure = {
        "epochs": {
            "description": "Number of epochs for the training",
            "type": int,
            "default": 3,
        },
        "batch_size": {
            "description": "The size of a batch for the training",
            "type": int,
            "default": 32,
        },
        "optimizer": {
            "description": "Optimizer used during the training",
            "type": str,
            "default": "adam",
            "enum": ["adam", "SGD", "RMSprop"],
        },
        "disable_from_logits": {
            "description": "Determines whether output of the model is "
            "normalized",
            "type": bool,
            "default": False,
        },
        "save_to_zip": {
            "description": "Determines whether optimized model should "
            "additionally be saved in ZIP format",
            "type": bool,
            "default": False,
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        epochs: int = 10,
        batch_size: int = 32,
        optimizer: str = "adam",
        disable_from_logits: bool = False,
        save_to_zip: bool = False,
    ):
        """
        TensorFlowOptimizer framework.

        This class adds a functionality for classification models fine-tuning
        using a given dataset and compiler options.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for quantization or
            fine-tuning.
        compiled_model_path : PathOrURI
            Path or URI where compiled model will be saved.
        location : Literal['host', 'target']
            Specifies where optimization should be performed in client-server
            scenario.
        epochs : int
            Number of epochs used to fine-tune the model.
        batch_size : int
            The size of a batch used for the fine-tuning.
        optimizer : str
            Optimizer used during the training.
        disable_from_logits : bool
            Determines whether output of the model is normalized.
        save_to_zip : bool
            Determines whether optimized model should additionally be saved in
            ZIP format.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.disable_from_logits = disable_from_logits
        self.save_to_zip = save_to_zip
        super().__init__(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            location=location,
        )
        assert (
            not self.save_to_zip or self.compiled_model_path.suffix != ".zip"
        ), "Please use different extension than `.zip`, it will be used by archived model"  # noqa: E501

    def prepare_train_validation(self) -> Tuple:
        """
        Prepares train and validation datasets of the model
        and splits them into batches.

        Returns
        -------
        Tuple
            Batched train and validation datasets.
        """
        Xt, Xv, Yt, Yv = self.dataset.train_test_split_representations()

        Xt = self.dataset.prepare_input_samples(Xt)
        Yt = self.dataset.prepare_output_samples(Yt)
        traindataset = tf.data.Dataset.from_tensor_slices((Xt, Yt))
        traindataset = traindataset.batch(
            self.batch_size, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        Xv = self.dataset.prepare_input_samples(Xv)
        Yv = self.dataset.prepare_output_samples(Yv)
        validdataset = tf.data.Dataset.from_tensor_slices((Xv, Yv))
        validdataset = validdataset.batch(
            self.batch_size, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return traindataset, validdataset

    def train_model(
        self, model: tf.keras.Model, callbacks: Optional[List] = None
    ) -> tf.keras.Model:
        """
        Compiles and trains the given model.

        The function can be used to retrain the model if needed.

        Parameters
        ----------
        model : tf.keras.Model
            The keras model to retrain.
        callbacks : Optional[List]
            List of callback function to use during the training.

        Returns
        -------
        tf.keras.Model
            Trained keras model.
        """
        traindataset, validdataset = self.prepare_train_validation()
        KLogger.info("Dataset prepared")

        if len(traindataset.element_spec[1].shape) == 1:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=not self.disable_from_logits
            )
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=not self.disable_from_logits
            )
            metrics = [tf.keras.metrics.CategoricalAccuracy()]

        model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)

        model.fit(
            traindataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1,
            validation_data=validdataset,
        )

        return model

    def compress_model_to_zip(self):
        """
        Compress saved model to ZIP archive.
        """
        with zipfile.ZipFile(
            str(self.compiled_model_path) + ".zip",
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as zfd:
            zfd.write(self.compiled_model_path)

    def save_model(self, model: tf.keras.Model):
        """
        Save Keras model to compiled_model_path
        and optionally archive it into ZIP.

        Parameters
        ----------
        model : tf.keras.Model
            Model that will be saved.
        """
        model.save(
            self.compiled_model_path, include_optimizer=False, save_format="h5"
        )

        if self.save_to_zip:
            self.compress_model_to_zip()

    def get_framework_and_version(self):
        return ("tensorflow", tf.__version__)
