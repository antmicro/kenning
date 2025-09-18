# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains Tensorflow models for the pet classification.

Pretrained on ImageNet dataset, trained on Pet Dataset.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from kenning.cli.command_template import TRAIN
from kenning.core.dataset import Dataset
from kenning.core.exceptions import TrainingParametersMissingError
from kenning.datasets.pet_dataset import PetDataset
from kenning.interfaces.io_interface import IOInterface
from kenning.modelwrappers.frameworks.tensorflow import TensorFlowWrapper
from kenning.utils.logger import KLogger, LoggerProgressBar
from kenning.utils.resource_manager import PathOrURI


class TensorFlowPetDatasetMobileNetV2(TensorFlowWrapper):
    """
    Model wrapper for pet classification in TensorFlow.
    """

    default_dataset = PetDataset
    pretrained_model_uri = "kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5"
    arguments_structure = {
        "batch_size": {
            "argparse_name": "--batch-size",
            "description": "Batch size for training. If not assigned, dataset batch size will be used.",  # noqa: E501
            "type": int,
            "default": None,
            "subcommands": [TRAIN],
        },
        "learning_rate": {
            "description": "Learning rate for training",
            "type": float,
            "default": None,
            "subcommands": [TRAIN],
        },
        "num_epochs": {
            "argparse_name": "--num-epochs",
            "description": "Number of epochs to train for",
            "type": int,
            "default": None,
            "subcommands": [TRAIN],
        },
        "logdir": {
            "argparse_name": "--logdir",
            "description": "Path to the logging directory",
            "type": Path,
            "default": None,
            "subcommands": [TRAIN],
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file=True,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        logdir: Optional[Path] = None,
    ):
        super().__init__(model_path, dataset, from_file, model_name)
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                KLogger.warning(f"Couldn't enable memory growth for {gpu}")

        if dataset is not None:
            self.numclasses = dataset.numclasses
            self.mean, self.std = dataset.get_input_mean_std()
            self.class_names = dataset.get_class_names()
        else:
            io_spec = self.load_io_specification(model_path)
            input_1 = IOInterface.find_spec(io_spec, "input", "input_1")
            out_layer = IOInterface.find_spec(io_spec, "output", "out_layer")

            self.mean = input_1["mean"]
            self.std = input_1["std"]
            self.class_names = out_layer["class_names"]
            self.numclasses = len(self.class_names)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.logdir = logdir

    @classmethod
    def _get_io_specification(
        cls, numclasses, class_names=None, mean=None, std=None, batch_size=1
    ):
        io_spec = {
            "input": [
                {
                    "name": "input_1",
                    "shape": (batch_size, 224, 224, 3),
                    "dtype": "float32",
                    "mean": mean,
                    "std": std,
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
        if mean is not None:
            io_spec["input"][0]["mean"] = mean
        if std is not None:
            io_spec["input"][0]["std"] = std
        return io_spec

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        return cls._get_io_specification(-1)

    def get_io_specification_from_model(self):
        mean = self.mean
        std = self.std

        if isinstance(mean, np.ndarray):
            mean = mean.tolist()
        if isinstance(std, np.ndarray):
            std = std.tolist()

        if self.dataset:
            return self._get_io_specification(
                self.numclasses,
                self.class_names,
                mean,
                std,
                self.dataset.batch_size,
            )

        return self._get_io_specification(
            self.numclasses, self.class_names, mean, std
        )

    def prepare_model(self):
        if self.model_prepared:
            return None
        import tensorflow as tf

        if self.from_file:
            self.load_model(self.model_path)
            self.model_prepared = True
        else:
            self.base = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights="imagenet",
            )
            self.base.trainable = False
            avgpool = tf.keras.layers.GlobalAveragePooling2D()(
                self.base.output
            )
            layer1 = tf.keras.layers.Dense(1024, activation="relu")(avgpool)
            d1 = tf.keras.layers.Dropout(0.5)(layer1)
            layer2 = tf.keras.layers.Dense(512, activation="relu")(d1)
            d2 = tf.keras.layers.Dropout(0.5)(layer2)
            layer3 = tf.keras.layers.Dense(128, activation="relu")(d2)
            d3 = tf.keras.layers.Dropout(0.5)(layer3)
            output = tf.keras.layers.Dense(self.numclasses, name="out_layer")(
                d3
            )
            self.model = tf.keras.models.Model(
                inputs=self.base.input, outputs=output
            )
            self.model_prepared = True
            self.save_model(self.model_path)
            self.model.summary()

    def train_model(self):
        import tensorflow as tf

        if not self.batch_size:
            self.batch_size = self.dataset.batch_size

        missing_params = []
        if not self.learning_rate:
            missing_params.append("learning_rate")

        if not self.num_epochs:
            missing_params.append("num_epochs")

        if not self.logdir:
            missing_params.append("logdir")
        else:
            self.logdir.mkdir(exist_ok=True, parents=True)

        if missing_params:
            raise TrainingParametersMissingError(missing_params)

        self.prepare_model()

        def preprocess_input(path, onehot):
            data = tf.io.read_file(path)
            img = tf.io.decode_jpeg(data, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)
            img /= 255.0
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.7, 1.0)
            img = tf.image.random_flip_left_right(img)
            img = (img - self.mean) / self.std
            return img, tf.convert_to_tensor(onehot)

        Xt, Xv, Yt, Yv = self.dataset.train_test_split_representations(0.25)
        Yt = self.dataset.prepare_output_samples(Yt)[0]
        Yv = self.dataset.prepare_output_samples(Yv)[0]
        traindataset = tf.data.Dataset.from_tensor_slices((Xt, Yt))
        traindataset = traindataset.map(
            preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.batch_size)
        validdataset = tf.data.Dataset.from_tensor_slices((Xv, Yv))
        validdataset = validdataset.map(
            preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.batch_size)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            str(self.logdir), histogram_freq=1
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.logdir),
            monitor="val_categorical_accuracy",
            mode="max",
            save_best_only=True,
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        with LoggerProgressBar(capture_stdout=True):
            self.model.fit(
                traindataset,
                epochs=self.num_epochs,
                callbacks=[tensorboard_callback, model_checkpoint_callback],
                validation_data=validdataset,
            )
