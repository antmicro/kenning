# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains Flax model for MagicWand dataset.
"""

from typing import List, Optional

import numpy as np
from tqdm import tqdm

from kenning.cli.command_template import TRAIN
from kenning.core.dataset import Dataset
from kenning.datasets.magic_wand_dataset import MagicWandDataset
from kenning.modelwrappers.frameworks.flax import FlaxWrapper
from kenning.utils.logger import LoggerProgressBar
from kenning.utils.resource_manager import PathOrURI


class FlaxMagicWandModelWrapper(FlaxWrapper):
    """
    Model wrapper for Magic Wand model.
    """

    default_dataset = MagicWandDataset
    pretrained_model_uri = (
        "kenning:///models/classification/flax_magic_wand.tar"
    )
    arguments_structure = {
        "window_size": {
            "argparse_name": "--window-size",
            "description": "Number of sensor samples",
            "type": int,
            "default": 128,
            "subcommands": [TRAIN],
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool = True,
        model_name: Optional[str] = None,
        window_size: int = 128,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
    ):
        super().__init__(
            model_path,
            dataset,
            from_file,
            model_name,
            batch_size,
            learning_rate,
            num_epochs,
        )
        self.window_size = window_size
        if self.dataset is not None:
            self.class_names = self.dataset.get_class_names()
            self.numclasses = len(self.class_names)
            self.save_io_specification(self.model_path)
            if self.batch_size is None:
                self.batch_size = self.dataset.batch_size

        if self.batch_size is None:
            self.batch_size = 1

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
        x = X[0]
        if type(x) is not np.ndarray:
            x = np.array(x, "float32")
        if len(x.shape) == 3:
            x = x.reshape((*x.shape, 1))
        return [x]

    def create_model_structure(self):
        from flax import nnx

        class MagicWandFlax(nnx.Module):
            def __init__(self, num_classes: int, rngs: nnx.Rngs):
                self.conv1 = nnx.Conv(1, 8, (4, 3), padding="SAME", rngs=rngs)
                self.dropout1 = nnx.Dropout(0.1, rngs=rngs)
                self.conv2 = nnx.Conv(8, 16, (4, 1), padding="SAME", rngs=rngs)
                self.dropout2 = nnx.Dropout(0.1, rngs=rngs)

                self.fc1 = nnx.Linear(224, 16, rngs=rngs)
                self.dropout3 = nnx.Dropout(0.1, rngs=rngs)
                self.fc2 = nnx.Linear(16, num_classes, rngs=rngs)

            def __call__(self, x):
                x = self.conv1(x)
                x = nnx.relu(x)
                x = nnx.max_pool(x, window_shape=(3, 3), strides=(3, 3))
                x = self.dropout1(x)

                x = self.conv2(x)
                x = nnx.relu(x)
                x = nnx.max_pool(x, window_shape=(3, 1), strides=(3, 1))
                x = self.dropout2(x)

                x = x.reshape(x.shape[0], -1)

                x = self.fc1(x)
                x = nnx.relu(x)
                x = self.dropout3(x)
                x = self.fc2(x)

                return x

        self.model = MagicWandFlax(
            self.numclasses, nnx.Rngs(params=0, dropout=1)
        )

    def train_model(self):
        import jax.numpy as jnp
        import optax
        import torch
        from flax import nnx
        from torch.utils.data import Dataset as TorchDataset

        if not self.batch_size:
            self.batch_size = self.dataset.batch_size

        optimizer = nnx.Optimizer(
            self.model,
            optax.adam(learning_rate=self.learning_rate),
            wrt=nnx.Param,
        )

        @nnx.jit
        def train_step(model, optimizer, x, y):
            def loss_fn(model):
                y_pred = model(x)
                loss = optax.softmax_cross_entropy(
                    logits=y_pred,
                    labels=y,
                ).mean()
                return loss, y_pred

            (loss, y_pred), grads = nnx.value_and_grad(
                loss_fn,
                has_aux=True,
            )(model)
            optimizer.update(model, grads)

            pred_class = jnp.argmax(y_pred, axis=-1)
            true_class = jnp.argmax(y, axis=-1)
            accuracy = jnp.mean(pred_class == true_class)
            return loss, accuracy

        @nnx.jit
        def eval_step(model, x, y):
            y_pred = model(x)
            loss = optax.softmax_cross_entropy(
                logits=y_pred,
                labels=y,
            ).mean()

            pred_class = jnp.argmax(y_pred, axis=-1)
            true_class = jnp.argmax(y, axis=-1)
            accuracy = jnp.mean(pred_class == true_class)
            return loss, accuracy

        class MagicWandDatasetPytorch(TorchDataset):
            def __init__(self, inputs, labels, dataset):
                self.inputs = inputs
                self.labels = labels
                self.dataset = dataset

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                x = self.dataset.prepare_input_samples([self.inputs[idx]])[0][
                    0
                ]
                y = np.asarray(
                    self.dataset.prepare_output_samples(self.labels[idx])
                )

                x = torch.from_numpy(x.astype("float32"))
                y = torch.from_numpy(y)
                return x, y

        (
            train_x,
            val_x,
            train_y,
            val_y,
        ) = self.dataset.train_test_split_representations(0.25)
        train_dataset = MagicWandDatasetPytorch(
            train_x,
            train_y,
            self.dataset,
        )
        validation_dataset = MagicWandDatasetPytorch(
            val_x,
            val_y,
            self.dataset,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

        with LoggerProgressBar() as logger_progress_bar:
            bar = tqdm(range(self.num_epochs), **logger_progress_bar.kwargs)
            for epoch in bar:
                train_losses = []
                train_accuracies = []
                validation_losses = []
                validation_accuracies = []

                self.model.train()

                for x_batch, y_batch in train_loader:
                    x_batch = jnp.asarray(x_batch)
                    y_batch = jnp.asarray(y_batch)
                    x_batch = self.preprocess_input([x_batch])[0]

                    loss, accuracy = train_step(
                        self.model,
                        optimizer,
                        x_batch,
                        y_batch,
                    )
                    train_losses.append(loss)
                    train_accuracies.append(accuracy)

                self.model.eval()

                for x_batch, y_batch in validation_loader:
                    x_batch = jnp.asarray(x_batch)
                    y_batch = jnp.asarray(y_batch)
                    x_batch = self.preprocess_input([x_batch])[0]

                    loss, accuracy = eval_step(
                        self.model,
                        x_batch,
                        y_batch,
                    )
                    validation_losses.append(loss)
                    validation_accuracies.append(accuracy)

                bar.set_description(
                    f"epoch={epoch} "
                    f"loss={jnp.mean(jnp.asarray(train_losses)):.4f} "
                    f"acc={jnp.mean(jnp.asarray(train_accuracies)):.4f} "
                    f"val_loss={jnp.mean(jnp.asarray(validation_losses)):.4f} "
                    f"val_acc={jnp.mean(jnp.asarray(validation_accuracies)):.4f}"
                )

    def prepare_model(self):
        self.create_model_structure()

        if self.from_file:
            self.load_model(self.model_path)
