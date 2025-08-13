# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides tools to fuse torch models to ai8x-compatible format.
"""

from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import torch

from kenning.core.exceptions import ConversionError
from kenning.utils.class_loader import append_to_sys_path
from kenning.utils.logger import KLogger


class LayerType(Enum):
    """
    Enum with layer type.
    """

    POOLING = auto()
    OP = auto()
    BATCHNORM = auto()
    ACTIVATION = auto()


def layer_name(layer: torch.nn.Module) -> str:
    """
    Returns name of torch model layer.

    Parameters
    ----------
    layer : torch.nn.Module
        Torch model layer.

    Returns
    -------
    str
        Name of the layer.
    """
    return layer.__class__.__name__


def get_size(size: Union[int, List[int]]) -> int:
    """
    Converts tensor size in list or int format to int. If the provided size is
    in list format, then it should have all elements equal.

    Parameters
    ----------
    size : Union[int, List[int]]
        Size to be converted.

    Returns
    -------
    int
        Size in int format.

    Raises
    ------
    ValueError
        If provided list elements are not equal.
    """
    if isinstance(size, int):
        return size
    if any(size[0] != s for s in size):
        raise ValueError(
            f"Provided list does not have all values equal: {size}"
        )
    return size[0]


def get_fuse_params(layer: torch.nn.Module) -> Dict[str, Any]:
    """
    Returns fused layer parameters extracted from provided torch layer.

    Parameters
    ----------
    layer : torch.nn.Module
        Layer which params should be extracted.

    Returns
    -------
    Dict[str, Any]
        Fused layer params.
    """
    if isinstance(
        layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
    ):
        return dict(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=get_size(layer.kernel_size),
            stride=get_size(layer.stride),
            padding=get_size(layer.padding),
            dilation=get_size(layer.dilation),
            bias=layer.bias is not None,
        )

    elif isinstance(layer, torch.nn.Linear):
        return dict(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
        )

    elif isinstance(layer, (torch.nn.MaxPool1d, torch.nn.MaxPool2d)):
        return dict(
            pool_size=get_size(layer.kernel_size),
            pool_stride=get_size(layer.stride),
            pool_dilation=get_size(layer.dilation),
        )

    elif isinstance(layer, (torch.nn.AvgPool1d, torch.nn.AvgPool2d)):
        return dict(
            pool_size=get_size(layer.kernel_size),
            pool_stride=get_size(layer.stride),
        )

    elif isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        return dict(batchnorm="Affine" if layer.affine else "NoAffine")

    return {}


def copy_params(src: torch.nn.Module, dest: torch.nn.Module):
    """
    Copies parameters between torch layers.

    Parameters
    ----------
    src : torch.nn.Module
        Source layer.
    dest : torch.nn.Module
        Destination layer.

    Raises
    ------
    ValueError
        Raised if provided layers have parameters of different size.
    """
    if dest.weight.size() != src.weight.size():
        raise ValueError(
            "Provided layers have weights of different size, "
            f"{src.weight.size()} != {dest.weight.size()}"
        )
    dest.weight.data = src.weight.view_as(dest.weight)
    if (src.bias is None) != (dest.bias is None):
        raise ValueError(
            "One of provided layers has bias while the other dont"
        )
    if src.bias is not None:
        if dest.bias.size() != src.bias.size():
            raise ValueError(
                "Provided layers have biases of different size, "
                f"{src.bias.size()} != {dest.bias.size()}"
            )
        dest.bias.data = src.bias.view_as(dest.bias)


def fuse_layers(
    fused_cls: Any, layers_types: List[LayerType]
) -> Callable[..., Any]:
    """
    Creates function that fuses torch model layers.

    Parameters
    ----------
    fused_cls : Any
        Class of the fused layer.
    layers_types : List[LayerType]
        List of fused layer types.

    Returns
    -------
    Callable[..., Any]
        Function that fuses torch model layers.
    """

    def fuse(*layers):
        layers = {
            layer_type: layer
            for layer_type, layer in zip(layers_types, layers)
        }
        KLogger.debug("Fusing layers:")
        for layer_type, layer in layers.items():
            KLogger.debug(f"\t{layer_type}: {layer}")

        op = layers.get(LayerType.OP, None)
        pool = layers.get(LayerType.POOLING, None)
        batchnorm = layers.get(LayerType.BATCHNORM, None)

        fuse_params = {}
        if op is not None:
            fuse_params = dict(fuse_params, **get_fuse_params(op))

        if pool is not None:
            fuse_params = dict(fuse_params, **get_fuse_params(pool))

        if batchnorm is not None:
            fuse_params = dict(fuse_params, **get_fuse_params(batchnorm))

        fused_layers = fused_cls(**fuse_params)

        if op is not None:
            copy_params(op, fused_layers.op)

        if batchnorm is not None and batchnorm.affine:
            copy_params(batchnorm, fused_layers.bn)

        KLogger.debug(f"Into: {fused_layers}")

        return fused_layers

    return fuse


def fuse_torch_sequential(
    ai8x_training_path: Path,
    device_id: int,
    torch_model: torch.nn.Sequential,
) -> torch.nn.Sequential:
    """
    Fuses torch sequential model into a model built of AI8X-compatible layers.

    Parameters
    ----------
    ai8x_training_path : Path
        Path to the ai8x-training tool.
    device_id : int
        AI8X device ID.
    torch_model : torch.nn.Sequential
        Torch sequential model.

    Returns
    -------
    torch.nn.Sequential
        Model built of AI8X-compatible layers.

    Raises
    ------
    ConversionError
        When unsupported model is passed.
    """
    with append_to_sys_path([ai8x_training_path]):
        import ai8x

    ai8x.set_device(device_id, False, False)

    fuse_map = {
        "Conv1d": {
            None: fuse_layers(ai8x.Conv1d, [LayerType.OP]),
            "BatchNorm1d": {
                None: fuse_layers(
                    ai8x.Conv1d, [LayerType.OP, LayerType.BATCHNORM]
                ),
                "ReLU": fuse_layers(
                    ai8x.FusedConv1dBNReLU,
                    [LayerType.OP, LayerType.BATCHNORM, LayerType.ACTIVATION],
                ),
                "Abs": fuse_layers(
                    ai8x.FusedConv1dBNAbs,
                    [LayerType.OP, LayerType.BATCHNORM, LayerType.ACTIVATION],
                ),
            },
            "ReLU": fuse_layers(
                ai8x.FusedConv1dReLU, [LayerType.OP, LayerType.ACTIVATION]
            ),
            "Abs": fuse_layers(
                ai8x.FusedConv1dAbs, [LayerType.OP, LayerType.ACTIVATION]
            ),
        },
        "Conv2d": {
            None: fuse_layers(ai8x.Conv2d, [LayerType.OP]),
            "BatchNorm2d": {
                None: fuse_layers(
                    ai8x.FusedConv2dBN, [LayerType.OP, LayerType.BATCHNORM]
                ),
                "ReLU": fuse_layers(
                    ai8x.FusedConv2dBNReLU,
                    [LayerType.OP, LayerType.BATCHNORM, LayerType.ACTIVATION],
                ),
                "Abs": fuse_layers(
                    ai8x.FusedConv2dAbs,
                    [LayerType.OP, LayerType.BATCHNORM, LayerType.ACTIVATION],
                ),
            },
            "ReLU": fuse_layers(
                ai8x.FusedConv2dReLU, [LayerType.OP, LayerType.ACTIVATION]
            ),
            "Abs": fuse_layers(
                ai8x.FusedConv2dAbs, [LayerType.OP, LayerType.ACTIVATION]
            ),
        },
        "ConvTranspose2d": {
            None: fuse_layers(ai8x.ConvTranspose2d, [LayerType.OP]),
            "BatchNorm2d": {
                None: fuse_layers(
                    ai8x.ConvTranspose2d, [LayerType.OP, LayerType.BATCHNORM]
                ),
                "ReLU": fuse_layers(
                    ai8x.FusedConvTranspose2dBNReLU,
                    [LayerType.OP, LayerType.BATCHNORM, LayerType.ACTIVATION],
                ),
                "Abs": fuse_layers(
                    ai8x.FusedConvTranspose2dAbs,
                    [LayerType.OP, LayerType.BATCHNORM, LayerType.ACTIVATION],
                ),
            },
            "ReLU": fuse_layers(
                ai8x.FusedConvTranspose2dReLU,
                [LayerType.OP, LayerType.ACTIVATION],
            ),
            "Abs": fuse_layers(
                ai8x.FusedConvTranspose2dAbs,
                [LayerType.OP, LayerType.ACTIVATION],
            ),
        },
        "MaxPool1d": {
            None: fuse_layers(
                partial(
                    ai8x.FusedMaxPoolConv1d,
                    in_channels=0,
                    out_channels=0,
                    kernel_size=None,
                ),
                [LayerType.POOLING],
            ),
            "Conv1d": {
                None: fuse_layers(
                    ai8x.FusedMaxPoolConv1d, [LayerType.POOLING, LayerType.OP]
                ),
                "BatchNorm1d": {
                    None: fuse_layers(
                        ai8x.FusedMaxPoolConv1dBN,
                        [LayerType.POOLING, LayerType.OP, LayerType.BATCHNORM],
                    ),
                    "ReLU": fuse_layers(
                        ai8x.FusedMaxPoolConv1dBNReLU,
                        [
                            LayerType.POOLING,
                            LayerType.OP,
                            LayerType.BATCHNORM,
                            LayerType.ACTIVATION,
                        ],
                    ),
                    "Abs": fuse_layers(
                        ai8x.FusedMaxPoolConv1dBNAbs,
                        [
                            LayerType.POOLING,
                            LayerType.OP,
                            LayerType.BATCHNORM,
                            LayerType.ACTIVATION,
                        ],
                    ),
                },
                "ReLU": fuse_layers(
                    ai8x.FusedMaxPoolConv1dReLU,
                    [LayerType.POOLING, LayerType.OP, LayerType.ACTIVATION],
                ),
                "Abs": fuse_layers(
                    ai8x.FusedMaxPoolConv1dAbs,
                    [LayerType.POOLING, LayerType.OP, LayerType.ACTIVATION],
                ),
            },
        },
        "AvgPool1d": {
            None: fuse_layers(
                partial(
                    ai8x.FusedAvgPoolConv1d,
                    in_channels=0,
                    out_channels=0,
                    kernel_size=None,
                ),
                [LayerType.POOLING],
            ),
            "Conv1d": {
                None: fuse_layers(
                    ai8x.FusedAvgPoolConv1d, [LayerType.POOLING, LayerType.OP]
                ),
                "BatchNorm1d": {
                    None: fuse_layers(
                        ai8x.FusedAvgPoolConv1d,
                        [LayerType.POOLING, LayerType.OP, LayerType.BATCHNORM],
                    ),
                    "ReLU": fuse_layers(
                        ai8x.FusedAvgPoolConv1dBNReLU,
                        [
                            LayerType.POOLING,
                            LayerType.OP,
                            LayerType.BATCHNORM,
                            LayerType.ACTIVATION,
                        ],
                    ),
                    "Abs": fuse_layers(
                        ai8x.FusedAvgPoolConv1dBNAbs,
                        [
                            LayerType.POOLING,
                            LayerType.OP,
                            LayerType.BATCHNORM,
                            LayerType.ACTIVATION,
                        ],
                    ),
                },
                "ReLU": fuse_layers(
                    ai8x.FusedAvgPoolConv1dReLU,
                    [LayerType.POOLING, LayerType.OP, LayerType.ACTIVATION],
                ),
                "Abs": fuse_layers(
                    ai8x.FusedAvgPoolConv1dAbs,
                    [LayerType.POOLING, LayerType.OP, LayerType.ACTIVATION],
                ),
            },
        },
        "MaxPool2d": {
            None: fuse_layers(
                partial(
                    ai8x.FusedMaxPoolConv2d,
                    in_channels=0,
                    out_channels=0,
                    kernel_size=None,
                ),
                [LayerType.POOLING],
            ),
            "Conv2d": {
                None: fuse_layers(
                    ai8x.FusedMaxPoolConv2d, [LayerType.POOLING, LayerType.OP]
                ),
                "BatchNorm2d": {
                    None: fuse_layers(
                        ai8x.FusedMaxPoolConv2d,
                        [LayerType.POOLING, LayerType.OP, LayerType.BATCHNORM],
                    ),
                    "ReLU": fuse_layers(
                        ai8x.FusedMaxPoolConv2dBNReLU,
                        [
                            LayerType.POOLING,
                            LayerType.OP,
                            LayerType.BATCHNORM,
                            LayerType.ACTIVATION,
                        ],
                    ),
                    "Abs": fuse_layers(
                        ai8x.FusedMaxPoolConv2dBNAbs,
                        [
                            LayerType.POOLING,
                            LayerType.OP,
                            LayerType.BATCHNORM,
                            LayerType.ACTIVATION,
                        ],
                    ),
                },
                "ReLU": fuse_layers(
                    ai8x.FusedMaxPoolConv2dReLU,
                    [LayerType.POOLING, LayerType.OP, LayerType.ACTIVATION],
                ),
                "Abs": fuse_layers(
                    ai8x.FusedMaxPoolConv2dAbs,
                    [LayerType.POOLING, LayerType.OP, LayerType.ACTIVATION],
                ),
            },
            "ConvTranspose2d": {
                None: fuse_layers(
                    ai8x.FusedMaxPoolConvTranspose2d,
                    [LayerType.POOLING, LayerType.OP],
                ),
                "BatchNorm2d": {
                    None: fuse_layers(
                        ai8x.FusedMaxPoolConvTranspose2d,
                        [LayerType.POOLING, LayerType.OP, LayerType.BATCHNORM],
                    ),
                    "ReLU": fuse_layers(
                        ai8x.FusedMaxPoolConvTranspose2dBNReLU,
                        [
                            LayerType.POOLING,
                            LayerType.OP,
                            LayerType.BATCHNORM,
                            LayerType.ACTIVATION,
                        ],
                    ),
                },
                "ReLU": fuse_layers(
                    ai8x.FusedMaxPoolConvTranspose2dReLU,
                    [LayerType.POOLING, LayerType.OP, LayerType.ACTIVATION],
                ),
                "Abs": fuse_layers(
                    ai8x.FusedMaxPoolConvTranspose2dAbs,
                    [LayerType.POOLING, LayerType.OP, LayerType.ACTIVATION],
                ),
            },
        },
        "AvgPool2d": {
            None: fuse_layers(
                partial(
                    ai8x.FusedAvgPoolConv2d,
                    in_channels=0,
                    out_channels=0,
                    kernel_size=None,
                ),
                [LayerType.POOLING],
            ),
            "Conv2d": {
                None: fuse_layers(
                    ai8x.FusedAvgPoolConv2d, [LayerType.POOLING, LayerType.OP]
                ),
                "BatchNorm2d": {
                    None: fuse_layers(
                        ai8x.FusedAvgPoolConv2d,
                        [LayerType.POOLING, LayerType.OP, LayerType.BATCHNORM],
                    ),
                    "ReLU": fuse_layers(
                        ai8x.FusedAvgPoolConv2dBNReLU,
                        [
                            LayerType.POOLING,
                            LayerType.OP,
                            LayerType.BATCHNORM,
                            LayerType.ACTIVATION,
                        ],
                    ),
                    "Abs": fuse_layers(
                        ai8x.FusedAvgPoolConv2dBNAbs,
                        [
                            LayerType.POOLING,
                            LayerType.OP,
                            LayerType.BATCHNORM,
                            LayerType.ACTIVATION,
                        ],
                    ),
                },
                "ReLU": fuse_layers(
                    ai8x.FusedAvgPoolConv2dReLU,
                    [LayerType.POOLING, LayerType.OP, LayerType.ACTIVATION],
                ),
                "Abs": fuse_layers(
                    ai8x.FusedAvgPoolConv2dAbs,
                    [LayerType.POOLING, LayerType.OP, LayerType.ACTIVATION],
                ),
            },
            "ConvTranspose2d": {
                None: fuse_layers(
                    ai8x.FusedAvgPoolConvTranspose2d,
                    [LayerType.POOLING, LayerType.OP],
                ),
                "BatchNorm2d": {
                    None: fuse_layers(
                        ai8x.FusedAvgPoolConvTranspose2d,
                        [LayerType.POOLING, LayerType.OP, LayerType.BATCHNORM],
                    ),
                    "ReLU": fuse_layers(
                        ai8x.FusedAvgPoolConvTranspose2dBNReLU,
                        [
                            LayerType.POOLING,
                            LayerType.OP,
                            LayerType.BATCHNORM,
                            LayerType.ACTIVATION,
                        ],
                    ),
                },
                "ReLU": fuse_layers(
                    ai8x.FusedAvgPoolConvTranspose2dReLU,
                    [LayerType.POOLING, LayerType.OP, LayerType.ACTIVATION],
                ),
                "Abs": fuse_layers(
                    ai8x.FusedAvgPoolConvTranspose2dAbs,
                    [LayerType.POOLING, LayerType.OP, LayerType.ACTIVATION],
                ),
            },
        },
        "Linear": {
            None: fuse_layers(ai8x.Linear, [LayerType.OP]),
            "ReLU": fuse_layers(
                ai8x.FusedLinearReLU, [LayerType.OP, LayerType.ACTIVATION]
            ),
            "Abs": fuse_layers(
                ai8x.FusedLinearAbs, [LayerType.OP, LayerType.ACTIVATION]
            ),
        },
        "Flatten": lambda x: x,
    }

    new_model_layers = []
    layers_to_fuse = []
    fuse_map_elem = None

    try:
        for layer in torch_model:
            if len(layers_to_fuse) == 0:
                layers_to_fuse.append(layer)
                fuse_map_elem = fuse_map[layer_name(layer)]
                continue

            if not isinstance(fuse_map_elem, dict):
                new_model_layers.append(fuse_map_elem(*layers_to_fuse))

                layers_to_fuse = [layer]
                fuse_map_elem = fuse_map[layer_name(layer)]
            elif layer_name(layer) not in fuse_map_elem:
                new_model_layers.append(fuse_map_elem[None](*layers_to_fuse))

                layers_to_fuse = [layer]
                fuse_map_elem = fuse_map[layer_name(layer)]
            else:
                layers_to_fuse.append(layer)
                fuse_map_elem = fuse_map_elem[layer_name(layer)]

        if isinstance(fuse_map_elem, dict):
            fuse_map_elem = fuse_map_elem[None]

        new_model_layers.append(fuse_map_elem(*layers_to_fuse))

    except KeyError:
        raise ConversionError(
            "The model cannot be converted to ai8x format. Found unsupported "
            f"layer {layer}"
        )

    ai8x_model = torch.nn.Sequential(*new_model_layers)

    KLogger.debug("Fused model:")
    for layer in ai8x_model:
        KLogger.debug(f"\t{layer.__class__.__name__}")

    return ai8x_model
