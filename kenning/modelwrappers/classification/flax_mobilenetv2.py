# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains a Flax NNX MobileNetV2 model for the Oxford-IIIT Pet Dataset.
"""

from typing import List, Optional

import numpy as np
from tqdm import tqdm

from kenning.core.dataset import Dataset
from kenning.datasets.pet_dataset import PetDataset
from kenning.interfaces.io_interface import IOInterface
from kenning.modelwrappers.frameworks.flax import FlaxWrapper
from kenning.utils.logger import LoggerProgressBar
from kenning.utils.resource_manager import PathOrURI


class FlaxPetDatasetMobileNetV2(FlaxWrapper):
    """
    Model wrapper for pet classification using MobileNetV2 in Flax NNX.

    The model uses the standard MobileNetV2 inverted-residual configuration
    and expects 224 x 224 RGB images in the NHWC layout internally.
    """

    default_dataset = PetDataset
    pretrained_model_uri = (
        "kenning:///models/classification/flax_pet_dataset_mobilenetv2.tar"
    )

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset],
        from_file: bool = True,
        model_name: Optional[str] = None,
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

        if self.dataset is not None:
            self.class_names = self.dataset.get_class_names()
            self.numclasses = len(self.class_names)
            self.mean, self.std = self.dataset.get_input_mean_std()

            if self.batch_size is None:
                self.batch_size = self.dataset.batch_size
        else:
            io_spec = self.load_io_specification(self.model_path)
            input_spec = IOInterface.find_spec(io_spec, "input", "input_1")
            output_spec = IOInterface.find_spec(io_spec, "output", "out_layer")

            self.mean = np.asarray(input_spec["mean"], dtype=np.float32)
            self.std = np.asarray(input_spec["std"], dtype=np.float32)
            self.class_names = output_spec["class_names"]
            self.numclasses = len(self.class_names)

        if self.batch_size is None:
            self.batch_size = 1

    @classmethod
    def _get_io_specification(
        cls,
        numclasses: int,
        class_names=None,
        mean=None,
        std=None,
        batch_size: int = 1,
    ):
        io_spec = {
            "input": [
                {
                    "name": "input_1",
                    "shape": [
                        (batch_size, 224, 224, 3),
                        (batch_size, 3, 224, 224),
                    ],
                    "dtype": "float32",
                    "mean": mean,
                    "std": std,
                }
            ],
            "processed_input": [
                {
                    "name": "input_1",
                    "shape": (batch_size, 224, 224, 3),
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
        if mean is not None:
            io_spec["input"][0]["mean"] = mean
        if std is not None:
            io_spec["input"][0]["std"] = std

        return io_spec

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        return cls._get_io_specification(-1)

    def get_io_specification_from_model(self):
        mean = (
            self.mean.tolist()
            if isinstance(self.mean, np.ndarray)
            else self.mean
        )
        std = (
            self.std.tolist() if isinstance(self.std, np.ndarray) else self.std
        )
        batch_size = (
            self.dataset.batch_size
            if self.dataset is not None
            else self.batch_size
        )

        return self._get_io_specification(
            self.numclasses,
            self.class_names,
            mean,
            std,
            batch_size,
        )

    def preprocess_input(self, X: List[np.ndarray]) -> List[np.ndarray]:
        X = super().preprocess_input(X)
        x = np.asarray(X[0], dtype=np.float32)

        if x.ndim != 4:
            raise ValueError(
                f"MobileNetV2 expects a 4D input tensor, got shape {x.shape}"
            )

        if x.shape[-1] == 3:
            return [x]

        if x.shape[1] == 3:
            return [np.transpose(x, (0, 2, 3, 1))]

        raise ValueError(
            "MobileNetV2 expects RGB images in NHWC or NCHW layout, "
            f"got shape {x.shape}"
        )

    def create_model_structure(self):
        import jax
        import jax.numpy as jnp
        from flax import nnx

        def make_divisible(
            value: float,
            divisor: int = 8,
            min_value: Optional[int] = None,
        ) -> int:
            if min_value is None:
                min_value = divisor

            rounded = max(
                min_value,
                int(value + divisor / 2) // divisor * divisor,
            )
            if rounded < 0.9 * value:
                rounded += divisor
            return rounded

        def relu6(x):
            return jnp.minimum(jnp.maximum(x, 0.0), 6.0)

        glorot_uniform = jax.nn.initializers.glorot_uniform()

        class DepthwiseConv2D(nnx.Module):
            def __init__(
                self,
                channels: int,
                kernel_size: int,
                stride: int,
                rngs: nnx.Rngs,
            ):
                self.channels = channels
                self.stride = stride
                self.padding = (
                    ((0, 1), (0, 1))
                    if stride == 2 and kernel_size == 3
                    else "SAME"
                )

                # [kernel_h, kernel_w, input_channels, depth_multiplier].
                kernel_key = rngs.params()
                self.kernel = nnx.Param(
                    glorot_uniform(
                        kernel_key,
                        (kernel_size, kernel_size, channels, 1),
                        jnp.float32,
                    )
                )

            def __call__(self, x):
                # JAX grouped convolution expects HWIO with one input channel
                # per group and `channels` output channels.
                kernel = jnp.transpose(self.kernel.value, (0, 1, 3, 2))
                return jax.lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(self.stride, self.stride),
                    padding=self.padding,
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                    feature_group_count=self.channels,
                )

        class ConvNormActivation(nnx.Module):
            def __init__(
                self,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int,
                rngs: nnx.Rngs,
                depthwise: bool = False,
                activate: bool = True,
            ):
                padding = (
                    ((0, 1), (0, 1))
                    if stride == 2 and kernel_size == 3
                    else "SAME"
                )

                if depthwise:
                    if in_channels != out_channels:
                        raise ValueError(
                            "Depthwise convolution requires equal input and "
                            "output channels when depth_multiplier=1"
                        )
                    self.conv = DepthwiseConv2D(
                        in_channels,
                        kernel_size,
                        stride,
                        rngs,
                    )
                else:
                    self.conv = nnx.Conv(
                        in_channels,
                        out_channels,
                        (kernel_size, kernel_size),
                        strides=(stride, stride),
                        padding=padding,
                        use_bias=False,
                        kernel_init=glorot_uniform,
                        rngs=rngs,
                    )

                self.norm = nnx.BatchNorm(
                    out_channels,
                    axis=-1,
                    momentum=0.999,
                    epsilon=1e-3,
                    rngs=rngs,
                )
                self.activate = activate

            def __call__(self, x):
                x = self.conv(x)
                x = self.norm(x, use_running_average=True)
                return relu6(x) if self.activate else x

        class InvertedResidual(nnx.Module):
            def __init__(
                self,
                in_channels: int,
                out_channels: int,
                stride: int,
                expand_ratio: int,
                rngs: nnx.Rngs,
            ):
                if stride not in (1, 2):
                    raise ValueError(f"Unsupported stride: {stride}")

                hidden_channels = in_channels * expand_ratio
                self.use_residual = stride == 1 and in_channels == out_channels

                self.expand = (
                    ConvNormActivation(
                        in_channels,
                        hidden_channels,
                        kernel_size=1,
                        stride=1,
                        rngs=rngs,
                    )
                    if expand_ratio != 1
                    else None
                )
                self.depthwise = ConvNormActivation(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    depthwise=True,
                    rngs=rngs,
                )
                self.project = ConvNormActivation(
                    hidden_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    activate=False,
                    rngs=rngs,
                )

            def __call__(self, x):
                residual = x

                if self.expand is not None:
                    x = self.expand(x)

                x = self.depthwise(x)
                x = self.project(x)

                if self.use_residual:
                    x = x + residual

                return x

        class MobileNetV2Backbone(nnx.Module):
            def __init__(
                self,
                rngs: nnx.Rngs,
                width_multiplier: float = 1.0,
            ):
                inverted_residual_setting = (
                    # expand ratio, output channels, repeats, first stride
                    (1, 16, 1, 1),
                    (6, 24, 2, 2),
                    (6, 32, 3, 2),
                    (6, 64, 4, 2),
                    (6, 96, 3, 1),
                    (6, 160, 3, 2),
                    (6, 320, 1, 1),
                )

                input_channels = make_divisible(
                    32 * width_multiplier,
                    8,
                )
                last_channels = (
                    make_divisible(1280 * width_multiplier, 8)
                    if width_multiplier > 1.0
                    else 1280
                )
                self.output_channels = last_channels

                self.stem = ConvNormActivation(
                    3,
                    input_channels,
                    kernel_size=3,
                    stride=2,
                    rngs=rngs,
                )

                blocks = []
                for (
                    expand_ratio,
                    channels,
                    repeats,
                    first_stride,
                ) in inverted_residual_setting:
                    output_channels = make_divisible(
                        int(channels * width_multiplier),
                        8,
                    )
                    for block_index in range(repeats):
                        stride = first_stride if block_index == 0 else 1
                        blocks.append(
                            InvertedResidual(
                                input_channels,
                                output_channels,
                                stride,
                                expand_ratio,
                                rngs,
                            )
                        )
                        input_channels = output_channels

                self.blocks = (
                    nnx.List(blocks) if hasattr(nnx, "List") else blocks
                )
                self.final_conv = ConvNormActivation(
                    input_channels,
                    last_channels,
                    kernel_size=1,
                    stride=1,
                    rngs=rngs,
                )

            def __call__(self, x):
                x = self.stem(x)

                for block in self.blocks:
                    x = block(x)

                return self.final_conv(x)

        class MobileNetV2(nnx.Module):
            def __init__(self, num_classes: int, rngs: nnx.Rngs):
                self.base = MobileNetV2Backbone(rngs=rngs)

                self.dense = nnx.Linear(
                    1280,
                    1024,
                    kernel_init=glorot_uniform,
                    rngs=rngs,
                )
                self.dropout = nnx.Dropout(0.5, rngs=rngs)
                self.dense_1 = nnx.Linear(
                    1024,
                    512,
                    kernel_init=glorot_uniform,
                    rngs=rngs,
                )
                self.dropout_1 = nnx.Dropout(0.5, rngs=rngs)
                self.dense_2 = nnx.Linear(
                    512,
                    128,
                    kernel_init=glorot_uniform,
                    rngs=rngs,
                )
                self.dropout_2 = nnx.Dropout(0.5, rngs=rngs)
                self.out_layer = nnx.Linear(
                    128,
                    num_classes,
                    kernel_init=glorot_uniform,
                    rngs=rngs,
                )

            def __call__(self, x):
                x = self.base(x)
                x = jnp.mean(x, axis=(1, 2))

                x = jax.nn.relu(self.dense(x))
                x = self.dropout(x)
                x = jax.nn.relu(self.dense_1(x))
                x = self.dropout_1(x)
                x = jax.nn.relu(self.dense_2(x))
                x = self.dropout_2(x)
                return self.out_layer(x)

        self.model = MobileNetV2(
            self.numclasses,
            nnx.Rngs(params=0, dropout=1),
        )

    def train_model(self):
        import jax.numpy as jnp
        import optax
        import torch
        from flax import nnx
        from torch.utils.data import Dataset as TorchDataset

        if not self.batch_size:
            self.batch_size = self.dataset.batch_size

        trainable_head_params = nnx.All(
            nnx.Param,
            nnx.Not(nnx.PathContains("base")),
        )
        optimizer = nnx.Optimizer(
            self.model,
            optax.adam(learning_rate=self.learning_rate),
            wrt=trainable_head_params,
        )

        @nnx.jit
        def train_step(model, optimizer, x, y):
            model.train()

            def loss_fn(model):
                y_pred = model(x)
                loss = optax.softmax_cross_entropy_with_integer_labels(
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
            accuracy = jnp.mean(pred_class == y)
            return loss, accuracy

        @nnx.jit
        def eval_step(model, x, y):
            model.eval()
            y_pred = model(x)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=y_pred,
                labels=y,
            ).mean()

            pred_class = jnp.argmax(y_pred, axis=-1)
            accuracy = jnp.mean(pred_class == y)
            return loss, accuracy

        class PetDatasetPytorch(TorchDataset):
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
                y = np.asarray(self.labels[idx], dtype=np.int64)

                x = torch.from_numpy(x.astype("float32"))
                y = torch.from_numpy(y)
                return x, y

        def prepare_labels(labels):
            labels = jnp.asarray(labels)
            return jnp.argmax(labels, axis=-1).astype(jnp.int32)

        (
            train_x,
            val_x,
            train_y,
            val_y,
        ) = self.dataset.train_test_split_representations(0.25)
        train_dataset = PetDatasetPytorch(
            train_x,
            train_y,
            self.dataset,
        )
        validation_dataset = PetDatasetPytorch(
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
            shuffle=True,
        )

        with LoggerProgressBar() as logger_progress_bar:
            bar = tqdm(range(self.num_epochs), **logger_progress_bar.kwargs)
            for epoch in bar:
                train_losses = []
                train_accuracies = []
                validation_losses = []
                validation_accuracies = []

                for x_batch, y_batch in train_loader:
                    x_batch = jnp.asarray(x_batch)
                    y_batch = prepare_labels(y_batch)
                    x_batch = self.preprocess_input([x_batch])[0]

                    loss, accuracy = train_step(
                        self.model,
                        optimizer,
                        x_batch,
                        y_batch,
                    )
                    train_losses.append(loss)
                    train_accuracies.append(accuracy)

                for x_batch, y_batch in validation_loader:
                    x_batch = jnp.asarray(x_batch)
                    y_batch = prepare_labels(y_batch)
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
        if self.model_prepared:
            return

        self.create_model_structure()

        if self.from_file:
            self.load_model(self.model_path)

        self.model_prepared = True
