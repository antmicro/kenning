# Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Structure of this module is based on AutoGPTQ implementation:
https://github.com/AutoGPTQ/AutoGPTQ.

The original module was licensed under MIT.
"""


import logging
import time
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Union

import coloredlogs
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from kenning.sparsegpt.sparsegpt import SparseGPT


@dataclass
class BasePruningConfig:
    """
    Configuration class for optimization parameters.

    Attributes
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
            Index of the first layer to be pruned. Cannot be used with
            quantization.
        maxlayer : int
            Index of the last layer to be pruned. Cannot be used with
            quantization.
    """

    sparsity: float = field(default=0.5, metadata={"min": 0, "max": 1})
    n_samples: int = field(default=128)
    prunen: Optional[int] = field(default=0)
    prunem: Optional[int] = field(default=0)
    block_size: int = field(default=128)
    minlayer: int = field(default=None)
    maxlayer: int = field(default=None)

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


class HijackException(Exception):
    """
    Exception used to hijack the first layer of the model.
    """

    pass


class BaseSparseGPTForCausalML(nn.Module):
    """
    Base class for sparsegpt models that can be used for pruning
    the model.

    Attributes
    ----------
    model : PreTrainedModel
        Model to be pruned
    pruning_config : BasePruningConfig
        Configuration of the pruning
    dev : str
        Device to run the pruning on. Can be either 'cpu' or 'cuda'
    logger : logging.Logger
        Logger used for logging
    outside_layer_modules : Optional[List[List[str]]]
        List of lists of layer names that are sequential. The order detetmines
        the order of pruning. It is overridden in the derived classes.
    inside_layer_modules : Optional[List[str]]
        List of layer names that are used for preprocessing. It is overridden
        in the derived classes. It has to be defined as it is needed when
        hijacking the first layer of the model.
    """

    outside_layer_modules = None
    inside_layer_modules = None

    def __init__(
        self,
        model: PreTrainedModel,
        pruning_config: BasePruningConfig,
        dev: str,
        verbosity: str,
    ):
        """
        Initializes the Pruner.

        Parameters
        ----------
        model : PreTrainedModel
            Model to be pruned
        pruning_config : BasePruningConfig
            Configuration of the pruning
        dev : str
            Device to run the pruning on. Can be either 'cpu' or 'cuda'
        verbosity : str
            Verbosity level of the logger. Can be one of the following:
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        super().__init__()

        self.model = model
        self.pruning_config = pruning_config
        self.dev = dev
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(
            level=verbosity,
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        pruning_config: BasePruningConfig,
        torch_dtype: torch.dtype = torch.float16,
        dev: str = "cuda:0",
        verbosity: str = "DEBUG",
        **model_init_kwargs,
    ) -> "BaseSparseGPTForCausalML":
        """
        Initializes the Pruner from a pretrained model.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Name or path of the pretrained model
        pruning_config : BasePruningConfig
            Configuration of the pruning
        torch_dtype : torch.dtype, optional
            Type of the torch tensors.
        dev : str, optional
            Device to run the pruning on. Can be either 'cpu'
            or 'cuda'.
        verbosity : str, optional
            Verbosity level of the logger. Can be one of the following:
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        **model_init_kwargs :
            Additional arguments passed to the model initializer

        Returns
        -------
        BaseSparseGPTForCausalML
            Pruner initialized from the pretrained model
        """
        # Setting model init arguments
        model_init_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **model_init_kwargs
        )
        model.eval()

        return cls(model, pruning_config, dev, verbosity)

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

    @torch.no_grad()
    def prune(
        self,
        examples: List[Dict[str, Union[List[int], torch.Tensor]]],
    ):
        """
        Prunes the model based on the calibration dataset provided
        and the pruning configuration specified in the constructor.

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

        # Preprocessing modules have to be moved to the same device
        # as the calibration dataset
        for name in self.inside_layer_modules:
            for module_name, module in self.model.named_modules():
                if module_name == name:
                    module = module.to(self.dev)

        try:
            seqlen = self.model.seqlen
        except AttributeError:
            seqlen = 4096
            logging.warning(
                "Model does not have 'seqlen' attribute. "
                + "Using default value of 4096"
            )

        if self.pruning_config.n_samples != len(examples):
            self.logger.error(
                "Number of samples in the calibration dataset "
                + f"({len(examples)}) is different than the number of samples "
                + f"({self.pruning_config.n_samples}) specified "
                + "in the pruning config. "
                + "Make sure that the configuration is correct"
            )
            raise ValueError

        dtype = self.model.dtype
        hidden_size = self.model.config.hidden_size

        inps = torch.zeros(
            (self.pruning_config.n_samples, seqlen, hidden_size),
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

        self.logger.debug(
            "Starting pruning with configuration: "
            + f"{self.pruning_config.__dict__}"
        )
        self.logger.debug(f"Found layers: {layers}")

        for i in range(len(layers)):
            if (
                self.pruning_config.minlayer is not None
                and self.pruning_config.maxlayer is not None
                and not self.pruning_config.minlayer
                <= i
                < self.pruning_config.maxlayer
            ):
                self.logger.debug(f"Skipping layer: {i}")
                continue

            self.logger.debug("-------------------------")
            self.logger.debug(f"Pruning layer number: {i}")
            self.logger.debug("-------------------------")

            layer = layers[i].to(self.dev)
            full = self._find_layers(layer)

            for names in self.outside_layer_modules:
                subset = {n: full[n] for n in names}

                gpts = {}
                for name in subset:
                    gpts[name] = SparseGPT(subset[name])

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
                for j in range(self.pruning_config.n_samples):
                    layer(inps[j].unsqueeze(0), attention_mask=attention_mask)
                for h in handles:
                    h.remove()

                for name in subset:
                    self.logger.debug(f"Pruning {name} layer")
                    sparsity = self.pruning_config.sparsity
                    tick = time.time()

                    error = gpts[name].optimize(
                        sparsity,
                        prunen=self.pruning_config.prunen,
                        prunem=self.pruning_config.prunem,
                        blocksize=self.pruning_config.block_size,
                    )
                    gpts[name].free()

                    self.logger.debug("Elapsed %.2f s" % (time.time() - tick))

            for j in range(self.pruning_config.n_samples):
                outs[j] = layer(
                    inps[j].unsqueeze(0), attention_mask=attention_mask
                )[0]

            layers[i] = layer.cpu()
            del layer
            del gpts

            torch.cuda.empty_cache()
            inps, outs = outs, inps

        self.model.config.use_cache = use_cache

    def save_pruned_model(self, path: str):
        """
        Saves the pruned model to the specified path.

        Parameters
        ----------
        path : str
            Path where the pruned model will be saved
        """
        self.model.save_pretrained(path)
