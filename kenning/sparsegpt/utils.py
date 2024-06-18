# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides functionality for preparing a quantized/pruned model
to be serialized into a compressed format.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import coloredlogs
import torch
import torch.nn as nn

from kenning.sparsegpt.quantLinear import QuantLinear

CPU = torch.device("cpu")


def recurse_setattr(module: nn.Module, name: str, value: Any):
    """
    A function to recursively set attributes to a module.

    Parameters
    ----------
    module : nn.Module
        Module to set the attribute
    name : str
        Name of the attribute
    value : Any
        Value to set
    """
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        recurse_setattr(getattr(module, name), rest, value)


def find_layers(
    module: nn.Module, layers: Optional[List[nn.Module]] = None, name: str = ""
) -> Dict[str, nn.Module]:
    """
    Finds layers in a module and returns them as a dictionary.

    Parameters
    ----------
    module : nn.Module
        Module to search for layers
    layers : Optional[List[nn.Module]]
        List of layers to search for
    name : str
        Name of the layer

    Returns
    -------
    Dict[str, nn.Module]
        Dictionary with keys as layer names and values as layers
    """
    if not layers:
        layers = [nn.Linear]
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res


def make_quant(
    module: nn.Module,
    names: Dict,
    bits: int,
    group_size: int,
    prunen: int,
    prunem: int,
    development_mode: bool,
) -> None:
    """
    Replaces the layers in the model with quantized layers.

    Parameters
    ----------
    module : nn.Module
        Module to pack
    names : Dict
        Dictionary with keys as layer names and values
        as quantization parameters
    bits : int
        Target precisioin of the quantized weights
    group_size : int
        Group size for the quantization
    prunen : int
        Number of pruned elements per `prunem` elements
    prunem : int
        Size of sparsity block
    development_mode : bool
        Development mode flag, determines whether the model should serialize
        both compressed and uncompressed weights in case of a sparse model.
        Additional checks to validate the model are also enabled.

    Returns
    -------
    None
    """
    if isinstance(module, QuantLinear):
        return

    for name, submodule in module.named_modules():
        if name in names:
            ori_layer_device = next(submodule.parameters()).device

            in_features = submodule.in_features
            out_features = submodule.out_features
            has_bias = submodule.bias is not None
            weight_dtype = submodule.weight.dtype

            quantization_parameters = names[name]

            # Sparsity metadata is stored as a second element
            is_sparse = quantization_parameters[1] is not None

            new_layer = QuantLinear(
                bits,
                group_size,
                in_features,
                out_features,
                prunen,
                prunem,
                sparse=is_sparse,
                bias=has_bias,
                development_mode=development_mode,
                weight_dtype=weight_dtype,
            )

            new_layer.device = ori_layer_device
            recurse_setattr(module, name, new_layer.to(ori_layer_device))


def pack_model(
    model: nn.Module,
    quantizers: Dict,
    bits: int,
    group_size: int,
    prunen: int,
    prunem: int,
    verbosity: str,
    development_mode: bool,
):
    """
    Pack the model with quantized weights and quantization parameters.

    Parameters
    ----------
    model : nn.Module
        Model to pack
    quantizers : Dict
        Dictionary of quantization parameters
    bits : int
        Target precision of the quantized weights
    group_size : int
        Group size for the quantization
    prunen : int
        Number of pruned elements per `prunem` elements
    prunem : int
        Size of sparsity block
    verbosity : str
        Logging verbosity level
    development_mode : bool
        Development mode flag, determines whether the model should serialize
        both compressed and uncompressed weights in case of a sparse model.
        Additional checks to validate the model are also enabled.
    """
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        level=verbosity,
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("Packing model")
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}

    if development_mode:
        logger.debug(
            "Development mode enabled. Additional information "
            + "may be serialized."
        )

    make_quant(
        model, quantizers, bits, group_size, prunen, prunem, development_mode
    )
    qlayers = find_layers(model, [QuantLinear])

    for name in qlayers:
        quantizers[name], sparsity_metadata, scale, zero, g_idx = quantizers[
            name
        ]
        layer_device = qlayers[name].device
        qlayers[name].to(CPU)

        layers[name], sparsity_metadata, scale, zero, g_idx = (
            layers[name].to(CPU),
            sparsity_metadata.to(CPU)
            if sparsity_metadata is not None
            else None,
            scale.to(CPU),
            zero.to(CPU),
            g_idx.to(CPU),
        )

        logger.debug(f"Packing layer {name}")
        tick = time.time()
        qlayers[name].pack(layers[name], sparsity_metadata, scale, zero, g_idx)
        qlayers[name].to(layer_device)
        logger.debug("Layer packing took: %.2f s" % (time.time() - tick))
    logger.info("Model packed")
