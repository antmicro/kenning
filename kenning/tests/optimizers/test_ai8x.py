# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import os
from itertools import product
from typing import Dict, Optional, Tuple

import pytest
import torch
import torch.nn as nn

from kenning.optimizers.ai8x_fuse import LayerType, fuse_torch_sequential
from kenning.utils.logger import KLogger


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


AI8X_TRAINING_PATH = os.environ["AI8X_TRAINING_PATH"]
AI8X_SYNTHESIS_PATH = os.environ["AI8X_SYNTHESIS_PATH"]

MAX78002_DEVICE_ID = 87

EPS = 1e-9

# Contains tuples of layer types and parameters values
FUSABLE_LAYERS_POOL = (
    (
        nn.MaxPool1d,
        dict(
            kernel_size=((2,), (8,), (16,)),
            dilation=((1,), (16,)),
        ),
    ),
    (
        nn.AvgPool1d,
        dict(
            kernel_size=((2,), (8,), (16,)),
        ),
    ),
    (
        nn.MaxPool2d,
        dict(
            kernel_size=((2, 2), (8, 8), (16, 16)),
            dilation=((1, 1), (16, 16)),
        ),
    ),
    (
        nn.AvgPool2d,
        dict(
            kernel_size=((2, 2), (8, 8), (16, 16)),
        ),
    ),
)
FUSABLE_LAYERS_OP = (
    (
        nn.Conv1d,
        dict(
            in_channels=(3, 32),
            out_channels=(16, 32),
            kernel_size=(1, 5, 9),
            padding=(0, 1, 2),
            bias=(True, False),
        ),
    ),
    (
        nn.Conv2d,
        dict(
            in_channels=(3, 32),
            out_channels=(16, 32),
            kernel_size=(1, 3),
            padding=(0, 1, 2),
            bias=(True, False),
        ),
    ),
    (
        nn.ConvTranspose2d,
        dict(
            in_channels=(16, 32),
            out_channels=(16, 32),
            kernel_size=(3,),
            padding=(0, 1, 2),
            stride=(2,),
            output_padding=(1,),
            bias=(True, False),
        ),
    ),
    (
        nn.Linear,
        dict(
            in_features=(16, 32, 512, 1024),
            out_features=(32, 1024),
            bias=(True, False),
        ),
    ),
)
FUSABLE_LAYERS_BATCHNORM = (
    (
        nn.BatchNorm1d,
        dict(
            affine=(True, False),
        ),
    ),
    (
        nn.BatchNorm2d,
        dict(
            affine=(True, False),
        ),
    ),
)
FUSABLE_LAYERS_ACTIVATION = (
    (
        nn.ReLU,
        dict(),
    ),
    (
        Abs,
        dict(),
    ),
)

# cartesian product of possible layers combinations
FUSABLE_LAYERS_PROD = list(
    product(
        (*FUSABLE_LAYERS_POOL, None),
        FUSABLE_LAYERS_OP,  # OP cannot be None
        (*FUSABLE_LAYERS_BATCHNORM, None),
        (*FUSABLE_LAYERS_ACTIVATION, None),
    )
)


def filter_combinations(layers_tuples: Tuple) -> bool:
    pool_tuple, op_tuple, batchnorm_tuple, _ = layers_tuples

    if batchnorm_tuple is not None:
        if (
            batchnorm_tuple[0] is nn.BatchNorm1d
            and op_tuple[0] is not nn.Conv1d
        ):
            # 1 dimensional batchnorm can only be combined with 1 dimensional
            # convolution
            return False
        if (
            batchnorm_tuple[0] is nn.BatchNorm2d
            and op_tuple[0] is not nn.Conv2d
        ):
            # 2 dimensional batchnorm can only be combined with 2 dimensional
            # convolution
            return False

    if pool_tuple is not None:
        if (
            pool_tuple[0] in (nn.MaxPool1d, nn.AvgPool1d)
            and op_tuple[0] is not nn.Conv1d
        ):
            # 1 dimensional pooling can only be combined with 1 dimensional
            # convolution
            return False
        if (
            pool_tuple[0] in (nn.MaxPool2d, nn.AvgPool2d)
            and op_tuple[0] is not nn.Conv2d
        ):
            # 2 dimensional pooling can only be combined with 2 dimensional
            # convolution
            return False

    return True


FUSABLE_LAYERS_PROD = list(filter(filter_combinations, FUSABLE_LAYERS_PROD))


class TestAi8xOptimizer:
    @pytest.mark.parametrize(
        "pool_tuple,op_tuple,batchnorm_tuple,activation_tuple",
        FUSABLE_LAYERS_PROD,
        ids=[
            "+".join(
                layer[0].__name__ for layer in layers if layer is not None
            )
            for layers in FUSABLE_LAYERS_PROD
        ],
    )
    def test_fuse_torch_sequential(
        self,
        pool_tuple: Optional[Tuple[nn.Module, Dict, Optional[Dict]]],
        op_tuple: Optional[Tuple[nn.Module, Dict, Optional[Dict]]],
        batchnorm_tuple: Optional[Tuple[nn.Module, Dict, Optional[Dict]]],
        activation_tuple: Optional[Tuple[nn.Module, Dict, Optional[Dict]]],
    ):
        # dict with all possible params, for the key we use layer type and
        # param name
        all_params = {}

        if pool_tuple is not None:
            pool_cls, pool_params = pool_tuple
            for param, value in pool_params.items():
                all_params[(LayerType.POOLING, param)] = value

        if op_tuple is not None:
            op_cls, op_params = op_tuple
            for param, value in op_params.items():
                all_params[(LayerType.OP, param)] = value

        if batchnorm_tuple is not None:
            batchnorm_cls, batchnorm_params = batchnorm_tuple

            for param, value in batchnorm_params.items():
                all_params[(LayerType.BATCHNORM, param)] = value

        if activation_tuple is not None:
            activation_cls, activation_params = activation_tuple
            for param, value in activation_params.items():
                all_params[(LayerType.ACTIVATION, param)] = value

        results = []
        n_failed = 0
        # cartesian product of all possible params values
        params_prod = list(product(*all_params.values()))
        for params in params_prod:
            # filter params for each layer type
            pool_params = {
                param: value
                for (layer_type, param), value in zip(
                    all_params.keys(), params
                )
                if layer_type == LayerType.POOLING
            }
            op_params = {
                param: value
                for (layer_type, param), value in zip(
                    all_params.keys(), params
                )
                if layer_type == LayerType.OP
            }
            batchnorm_params = {
                param: value
                for (layer_type, param), value in zip(
                    all_params.keys(), params
                )
                if layer_type == LayerType.BATCHNORM
            }
            activation_params = {
                param: value
                for (layer_type, param), value in zip(
                    all_params.keys(), params
                )
                if layer_type == LayerType.ACTIVATION
            }
            if batchnorm_tuple is not None:
                # AI8X requires bias when using batchnorm
                if not op_params["bias"]:
                    continue
                batchnorm_params["num_features"] = op_params.get(
                    "out_features", op_params.get("out_channels")
                )
            KLogger.info(f"{pool_params=}")
            KLogger.info(f"{op_params=}")
            KLogger.info(f"{batchnorm_params=}")
            KLogger.info(f"{activation_params=}")

            # create layers
            pool = pool_cls(**pool_params) if pool_tuple is not None else None
            op = op_cls(**op_params) if op_tuple is not None else None
            batchnorm = (
                batchnorm_cls(**batchnorm_params)
                if batchnorm_tuple is not None
                else None
            )
            activation = (
                activation_cls(**activation_params)
                if activation_tuple is not None
                else None
            )

            layers = [
                layer
                for layer in (
                    pool,
                    op,
                    batchnorm,
                    activation,
                )
                if layer is not None
            ]

            # compute input shape
            if isinstance(pool, (nn.MaxPool1d, nn.MaxPool2d)):
                dim = pool.kernel_size[0] * pool.dilation[0]
            elif isinstance(pool, (nn.AvgPool1d, nn.AvgPool2d)):
                dim = pool.kernel_size[0]
            else:
                dim = 1

            if isinstance(op, nn.Conv1d):
                sample_input_shape = (
                    op.in_channels,
                    dim * (1 + op.kernel_size[0]),
                )
            elif isinstance(op, (nn.Conv2d, nn.ConvTranspose2d)):
                sample_input_shape = (
                    op.in_channels,
                    dim * (1 + op.kernel_size[0]),
                    dim * (1 + op.kernel_size[1]),
                )
            elif isinstance(op, nn.Linear):
                sample_input_shape = (op.in_features,)
            else:
                # should not happen
                raise TypeError("Invalid op")

            sample_input = torch.randn((4, *sample_input_shape))

            # create simple sequential model
            torch_model = nn.Sequential(*layers)

            sample_output = torch_model(sample_input)

            try:
                fused_model = fuse_torch_sequential(
                    AI8X_TRAINING_PATH,
                    MAX78002_DEVICE_ID,
                    torch_model,
                )

                fused_model_output = fused_model(sample_input)

                # check shape
                assert sample_output.shape == fused_model_output.shape, (
                    "Invalid output shape, "
                    f"{sample_output.shape} != {fused_model_output.shape}, "
                    f"input shape was {sample_input.shape}"
                )
                # check values
                diff = (sample_output - fused_model_output).abs().max()
                assert diff < EPS, f"Invalid output values {diff} >= EPS"

            except Exception as e:
                results.append(f"{e.__class__.__name__}: {e}")
                n_failed += 1
            else:
                results.append(0)

        if n_failed > 0:
            KLogger.error(f"Failed for {n_failed}/{len(results)}")
            for result, params in zip(results, params_prod):
                if result != 0:
                    KLogger.error(f"\t{params} : {result}")

            pytest.fail("Failed for some params combinations, see logs")
