# Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Implements the main class for model optimization that iteratively,
layer by layer, quantizes and/or prunes the model.
"""

import json
import logging
import time
from dataclasses import dataclass, field, fields
from os.path import join
from pprint import pformat
from typing import Dict, List, Union

import coloredlogs
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from kenning.core.exceptions import KenningOptimizerError
from kenning.sparsegpt.quant import Quantizer
from kenning.sparsegpt.sparsegpt import SparseGPT
from kenning.sparsegpt.utils import pack_model


@dataclass
class BaseOptimizationConfig:
    """
    Configuration class for optimization parameters.

    Attributes
    ----------
    sparsity : float
        The desired sparsity level. Must be in the range [0, 1].
    n_samples : int
        The number of samples used for calibration.
    prunen : int
        Number of weights pruned in semi-structured manner.
        Both prunen and prunem have to be non-zero to use semi-structured
        pruning. If semi-structured pruning is used, prunen has to be
        smaller than prunem.
    prunem : int
        Block size in semi-structured pruning.
        Both prunen and prunem have to be non-zero to use semi-structured
        pruning.
    block_size : int
        Number of columns that are pruned and quantized at once.
        It is a trade-off between the speed and the memory usage.
    minlayer : int
        Index of the first layer to be optimized. Cannot be used with
        quantization.
    maxlayer : int
        Index of the last layer to be optimized. Cannot be used with
        quantization.
    bits : int
        Number of bits used for quantization. Can be
        one of the following: 2, 3, 4, 8, 16.
    sym : bool
        Whether to use symmetric quantization.
    """

    sparsity: float = field(default=0.5, metadata={"min": 0, "max": 1})
    n_samples: int = field(default=128)
    prunen: int = field(default=0)
    prunem: int = field(default=0)
    block_size: int = field(default=128)
    minlayer: int = field(default=None)
    maxlayer: int = field(default=None)
    bits: int = field(default=16, metadata={"range": [2, 3, 4, 8, 16]})
    sym: bool = field(default=False)

    def __post_init__(self):
        fields_info = fields(self)

        if (
            self.sparsity < fields_info[0].metadata["min"]
            or self.sparsity > fields_info[0].metadata["max"]
        ):
            raise ValueError(
                "Invalid 'sparsity' value. It has to be in range [0, 1]"
            )

        if (self.prunem == 0 and self.prunen != 0) or (
            self.prunen == 0 and self.prunem != 0
        ):
            raise ValueError("Both 'prunem' and 'prunen' have to be defined")

        if self.prunen > self.prunem:
            raise ValueError(
                "Invalid 'prunen' and 'prunem' values. "
                + "prunen has to be smaller than prunem"
            )

        if (
            self.prunen * self.prunem != 0
            and self.prunen / self.prunem != self.sparsity
        ):
            raise ValueError(
                "When using semi-structured pruning, `sparsity` value "
                + "has to match `prunen` and `prunem` values"
            )

        if self.bits not in fields_info[-2].metadata["range"]:
            raise ValueError(
                f"Invalid quantization precision value {self.bits}. "
                + f"Allowed values: {fields_info[-2].metadata['range']}"
            )

    def save_pretrained(self, save_dir: str, **kwargs):
        # Not all quantization parameters are yet supported, so the
        # values are set statically
        # auto-gptq compatibility
        if self.bits < 16:
            with open(join(save_dir, "quantize_config.json"), "w") as f:
                json.dump(
                    {
                        "bits": self.bits,
                        "group_size": self.block_size,
                        "damp_percent": 0.01,
                        "desc_act": False,
                        "static_groups": False,
                        "sym": self.sym,
                        "true_sequential": True,
                        "model_name_or_path": None,
                        "model_file_base_name": None,
                    },
                    f,
                    indent=4,
                )


class HijackException(Exception):
    """
    Exception used to hijack the first layer of the model.
    """

    pass


class InvalidMetadataError(KenningOptimizerError):
    """
    Exception raised when sparsity metadata is invalid.
    """

    pass


class BaseSparseGPTForCausalML(nn.Module):
    """
    Base class for sparsegpt models that can be used for pruning and
    quantizing the model.

    Attributes
    ----------
    model : PreTrainedModel
        Model to be optimized
    config : BaseOptimizationConfig
        Configuration of the optimization
    dev : str
        Device to run the optimization on. Can be either 'cpu' or 'cuda'
    logger : logging.Logger
        Logger used for logging
    outside_layer_modules : Optional[List[List[str]]]
        List of lists of layer names that are sequential. The order determines
        the order of optimization. It is overridden in the derived classes.
    inside_layer_modules : Optional[List[str]]
        List of layer names that are used for preprocessing. It is overridden
        in the derived classes. It has to be defined as it is needed when
        hijacking the first layer of the model.
    compressible_modules : Optional[List[str]]
        List of layer names that are being pruned and compressed when using
        semi-structured pruning.
    """

    outside_layer_modules = None
    inside_layer_modules = None
    compressible_modules = None

    def __init__(
        self,
        model: PreTrainedModel,
        config: BaseOptimizationConfig,
        dev: str,
        verbosity: str,
        development_mode: bool = False,
    ):
        """
        Initializes the Optimizer.

        Parameters
        ----------
        model : PreTrainedModel
            Model to be optimized
        config : BaseOptimizationConfig
            Optimization config
        dev : str
            Device to run the optimization on. Can be either 'cpu' or 'cuda'
        verbosity : str
            Verbosity level of the logger. Can be one of the following:
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        development_mode : bool
            Determines whether to run additional checks during model
            optimization. If set to True, the model will be optimized
            with additional checks to ensure that the model is optimized
            correctly.
        """
        super().__init__()

        self.model = model
        self.config = config
        self.dev = dev
        self.logger = logging.getLogger(__name__)
        self.verbosity = verbosity
        self.development_mode = development_mode
        coloredlogs.install(
            level=verbosity,
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: BaseOptimizationConfig,
        torch_dtype: torch.dtype = torch.float16,
        dev: str = "cuda:0",
        verbosity: str = "DEBUG",
        development_mode: bool = False,
        **model_init_kwargs,
    ) -> "BaseSparseGPTForCausalML":
        """
        Initializes the Optimizer from a pretrained model.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Name or path of the pretrained model
        config : BaseOptimizationConfig
            Optimization config
        torch_dtype : torch.dtype
            Type of the torch tensors.
        dev : str
            Device to run the optimization on. Can be either 'cpu'
            or 'cuda'.
        verbosity : str
            Verbosity level of the logger. Can be one of the following:
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        development_mode : bool
            Determines whether to run additional checks during model
            optimization. If set to True, the model will be optimized
            with additional checks to ensure that the model is optimized
            correctly.
        **model_init_kwargs :
            Additional arguments passed to the model initializer

        Returns
        -------
        BaseSparseGPTForCausalML
            Optimizer initialized from the pretrained model
        """
        # Setting model init arguments
        model_init_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **model_init_kwargs
        )
        model.eval()

        return cls(model, config, dev, verbosity, development_mode)

    def _find_layers(
        self,
        module: nn.Module,
        layers: List[nn.Module] = [nn.Conv2d, nn.Linear],
        name: str = "",
    ) -> Dict[str, nn.Module]:
        """
        Finds layers defined in `layers` in the model and returns them.

        Parameters
        ----------
        module : nn.Module
            Module to search for layers
        layers : List[nn.Module]
            List of layers to search for
        name : str
            Name of the module

        Returns
        -------
        Dict[str, nn.Module]
            Dictionary of layers found in the model
        """
        res = {}

        for child in module.named_modules():
            child_name, child_module = child
            full_name = f"{name}.{child_name}" if name != "" else child_name
            if type(child_module) in layers:
                res[full_name] = child_module
        return res

    def _prepare_examples(
        self,
        examples: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Converts examples to torch tensors.

        Parameters
        ----------
        examples : List[Dict[str, Union[List[int], torch.Tensor]]]
            List of examples that constitute the calibration dataset

        Returns
        -------
        List[Dict[str, torch.Tensor]]
            List of examples that constitute the calibration dataset
            with converted input_ids and attention_mask to torch tensors
        """

        def convert_to_tensor(
            key: str, example: Union[List[int], torch.Tensor]
        ):
            if isinstance(example, list):
                example = torch.Tensor(example)
                example.unsqueeze(0)

            if example.dim() != 2:
                self.logger.error(
                    f"{key} tensor has to be 2-dimensional. "
                    + f"Got {example.dim()}"
                )
                raise ValueError
            return example

        for example in examples:
            example["input_ids"] = convert_to_tensor(
                "input_ids", example["input_ids"]
            )
            example["attention_mask"] = convert_to_tensor(
                "attention_mask", example["attention_mask"]
            )
        return examples

    @torch.no_grad
    def _validate_prune_n_m(
        self,
        weights: torch.Tensor,
        metadata: torch.Tensor,
        prunen: int,
        prunem: int,
    ):
        """
        Validates the pruning mask in the semi-structured pruning.

        All values indicated by mask have to be zero and for every
        `prunem` values in the mask, at least `prunen` of them have to be zero.

        Parameters
        ----------
        weights : torch.Tensor
            Pruned weights
        metadata : torch.Tensor
            Pruning mask
        prunen : int
            Number of zeroed values in blocks of size `prunem`
        prunem : int
            Pruned block size

        Raises
        ------
        Exception
            Raised if validation fails
        """
        if not (weights[metadata] == 0).all().item():
            raise InvalidMetadataError(
                "All values indicated by mask have to be zero"
            )

        metadata_view = metadata.reshape(-1, prunem)
        if not (metadata_view.sum(dim=1) == prunen).all().item():
            raise InvalidMetadataError(
                f"At least {prunen} values have to be zero "
                + "in every block of size {prunem}"
            )

    @torch.no_grad()
    def optimize(
        self,
        examples: List[Dict[str, Union[List[int], torch.Tensor]]],
    ):
        """
        Optimizes the model based on the calibration dataset provided
        and the configuration specified in the constructor.

        Parameters
        ----------
        examples : List[Dict[str, Union[List[int], torch.Tensor]]]
            List of examples that constitute the calibration dataset.
            Each example has to be a dictionary with the following keys:
            'input_ids' and 'attention_mask'. The values of these keys
            have to be 2-dimensional torch tensors.

        Raises
        ------
        ValueError
            If the number of samples in the calibration dataset is different
            than the number of samples specified in the pruning config
        """
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        start_tick = time.time()

        # Preprocessing modules have to be moved to the same device
        # as the calibration dataset
        for name in self.inside_layer_modules:
            for module_name, module in self.model.named_modules():
                if module_name == name:
                    module = module.to(self.dev)

        if hasattr(self.model, "seqlen"):
            seqlen = self.model.seqlen
        elif hasattr(self.model, "config") and hasattr(
            self.model.config, "max_position_embeddings"
        ):
            seqlen = self.model.config.max_position_embeddings
        else:
            seqlen = 4096
            logging.warning(
                "Model does not have 'seqlen' or 'max_position_embeddings' "
                + "attribute. Using default value of 4096"
            )

        # Too big sequence length can cause out of memory errors
        seqlen = min(seqlen, 4096)

        if self.config.n_samples != len(examples):
            self.logger.error(
                "Number of samples in the calibration dataset "
                + f"({len(examples)}) is different than the number of samples "
                + f"({self.config.n_samples}) specified "
                + "in the pruning config. "
                + "Make sure that the configuration is correct"
            )
            raise ValueError

        dtype = self.model.dtype
        hidden_size = self.model.config.hidden_size

        inps = torch.zeros(
            (self.config.n_samples, seqlen, hidden_size),
            dtype=dtype,
            device=self.dev,
        )

        cache = {"i": 0, "attention_mask": None}

        class HijackModule(nn.Module):
            """
            Wrapper for the first layer to catch the input and attention_mask.
            """

            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                raise HijackException

        layers = self.model.model.layers

        # Hijacking the first layer to catch the input and attention_mask
        layers[0] = layers[0].to(self.dev)
        layers[0] = HijackModule(layers[0])

        examples = self._prepare_examples(examples)
        for example in examples:
            for key in example:
                example[key] = example[key].to(self.dev)

            try:
                self.model(**example)
            except HijackException:
                pass
            except Exception as e:
                self.logger.error("Exception occurred when processing input")
                raise e

        # Removing the HijackModule
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()

        # Preprocessing modules have to be moved to the same device
        # as the calibration dataset
        for name in self.inside_layer_modules:
            for module_name, module in self.model.named_modules():
                if module_name == name:
                    module = module.cpu()

        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache["attention_mask"]

        self.logger.info(
            "Optimizing with configuration: \n" + pformat(self.config.__dict__)
        )
        self.logger.info(f"Found layers: {layers}")

        quantizers = {}
        for i in range(len(layers)):
            tick = time.time()
            if (
                self.config.minlayer is not None
                and self.config.maxlayer is not None
                and not self.config.minlayer <= i < self.config.maxlayer
            ):
                self.logger.debug(f"Skipping layer: {i}")
                continue

            self.logger.debug("-------------------------")
            self.logger.debug(f"Optimizing layer number: {i}")
            self.logger.debug("-------------------------")

            layer = layers[i].to(self.dev)
            full = self._find_layers(layer)

            for names in self.outside_layer_modules:
                subset = {n: full[n] for n in names}

                gpts = {}
                for name in subset:
                    gpts[name] = SparseGPT(subset[name])

                    if self.config.bits < 16:
                        gpts[name].quantizer = Quantizer()
                        gpts[name].quantizer.configure(
                            bits=self.config.bits,
                            perchannel=True,
                            sym=False,
                            mse=False,
                        )

                def add_batch(name):
                    def tmp(_, inp, out):
                        # Adding batch to the sparsegpt
                        # which is used to calculate the Hessian
                        gpts[name].add_batch(inp[0].data)  # noqa: F821

                    return tmp

                handles = []
                for name in subset:
                    handles.append(
                        subset[name].register_forward_hook(add_batch(name))
                    )
                for j in range(self.config.n_samples):
                    layer(inps[j].unsqueeze(0), attention_mask=attention_mask)
                for h in handles:
                    h.remove()

                for name in subset:
                    self.logger.debug(f"Optimizing {name}")
                    sparsity = self.config.sparsity

                    error, sparsity_metadata, scale, zero, g_idx = gpts[
                        name
                    ].optimize(
                        sparsity,
                        prunen=self.config.prunen,
                        prunem=self.config.prunem,
                        blocksize=self.config.block_size,
                    )

                    # If semi-structured pruning is used,
                    # pruning mask is validated in a development mode
                    if self.config.prunem != 0 and self.development_mode:
                        self._validate_prune_n_m(
                            gpts[name].layer.weight,
                            sparsity_metadata,
                            self.config.prunen,
                            self.config.prunem,
                        )

                    if self.config.bits < 16:
                        quantizers[f"model.layers.{i}.{name}"] = (
                            gpts[name].quantizer.to("cpu"),
                            (
                                sparsity_metadata.to("cpu")
                                if name in self.compressible_modules
                                and len(sparsity_metadata)
                                else None
                            ),
                            scale.to("cpu"),
                            zero.to("cpu"),
                            g_idx.to("cpu"),
                        )
                    gpts[name].free()

                    self.logger.debug(f"Error: {error}")

            self.logger.debug(
                "Layer optimization took: %.2f s" % (time.time() - tick)
            )

            for j in range(self.config.n_samples):
                outs[j] = layer(
                    inps[j].unsqueeze(0), attention_mask=attention_mask
                )[0]

            layers[i] = layer.cpu()
            del layer
            del gpts

            torch.cuda.empty_cache()
            inps, outs = outs, inps

        self.model.config.use_cache = use_cache

        if self.config.bits < 16:
            self.logger.info("-------------------------")
            self.logger.info("Optimization finished")
            self.logger.info("Elapsed %.2f s" % (time.time() - start_tick))
            self.logger.info("-------------------------")

            pack_model(
                self.model,
                quantizers,
                self.config.bits,
                self.config.block_size,
                self.config.prunen,
                self.config.prunem,
                self.verbosity,
                self.development_mode,
            )

        self.logger.info("-------------------------")
        self.logger.info("Optimization and packing finished")
        self.logger.info("Elapsed %.2f s" % (time.time() - start_tick))
        self.logger.info("-------------------------")

    def _safetensors_metadata(self) -> Dict[str, str]:
        """
        Returns the metadata that will be saved along with the model.

        Returns
        -------
        Dict[str, str]
            Dictionary with metadata
        """
        safetensors_metadata = {}
        safetensors_metadata["format"] = "pt"
        safetensors_metadata["gptq_bits"] = str(self.config.bits)
        safetensors_metadata["gptq_group_size"] = str(self.config.block_size)
        safetensors_metadata["gptq_desc_act"] = str(False)
        safetensors_metadata["gptq_damp_percent"] = str(0.01)

        return safetensors_metadata

    def save_optimized(self, path: str):
        """
        Saves the optimized model to the specified path.

        Parameters
        ----------
        path : str
            Path where the optimized model will be saved
        """
        import os

        from safetensors.torch import save_file

        safetensors_metadata = self._safetensors_metadata()

        os.makedirs(path, exist_ok=True)
        save_file(
            self.model.state_dict(),
            (path + "/model.safetensors"),
            safetensors_metadata,
        )

        self.config.save_pretrained(path)
        self.model.config.save_pretrained(path)
