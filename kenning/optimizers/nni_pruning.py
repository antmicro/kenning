# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Pruning optimizer implementation with Neural Network Intelligence
"""
import torch
import nni
from nni.compression.pytorch.pruning import (
    ActivationAPoZRankPruner,
    ActivationMeanRankPruner,
    ActivationPruner,
)
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.algorithms.compression.v2.pytorch import TorchEvaluator
from nni.compression.pytorch.speedup.compress_modules import (
    convert_to_coarse_mask,
    replace_module
)
from nni.compression.pytorch.speedup.compressor import _logger as nni_logger
from nni.common.graph_utils import _logger as nni_graph_logger
import numpy as np
from typing import Callable, Dict, Optional, List, Type, Tuple
from enum import Enum
import copy
import dill
import logging
from tqdm import tqdm

from kenning.core.onnxconversion import SupportStatus
from kenning.core.optimizer import Optimizer, CompilationError
from kenning.core.dataset import Dataset
from kenning.onnxconverters.pytorch import PyTorchONNXConversion
from kenning.utils.class_loader import load_class
from kenning.utils.logger import LoggerProgressBar
from kenning.onnxconverters.onnx2torch import convert
from kenning.utils.resource_manager import PathOrURI


def torchconversion(model_path: PathOrURI, device: torch.device, **kwargs):
    loaded = torch.load(str(model_path), map_location=device)
    if not isinstance(loaded, torch.nn.Module):
        raise CompilationError(
            f'Expecting model of type:'
            f' torch.nn.Module, but got {type(loaded).__name__}')
    return loaded


def onnxconversion(model_path: PathOrURI, device: torch.device, **kwargs):
    conversion = PyTorchONNXConversion()
    if conversion.onnx_import(None, model_path) != SupportStatus.SUPPORTED:
        raise CompilationError('Conversion for provided model to PyTorch'
                               ' is not supported')
    model = convert(str(model_path))
    model.to(device)
    return model


class AddOperation(torch.nn.Module):
    """
    PyTorch module for adding and adjusting size of two tensor with masks
    """

    def __init__(self,
                 mask_a: torch.Tensor,
                 mask_b: torch.Tensor,
                 mask_out: torch.Tensor) -> None:
        super().__init__()
        self.original_size = tuple(mask_out.shape)[1:]
        self.register_buffer(
            'channels_a', convert_to_coarse_mask(mask_a, 1)[1])
        self.register_buffer(
            'channels_b', convert_to_coarse_mask(mask_b, 1)[1])
        self.register_buffer(
            'channels_out', convert_to_coarse_mask(mask_out, 1)[1])

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        operation_result = torch.zeros(a.shape[0], *self.original_size,
                                       dtype=a.dtype, device=a.device)
        operation_result[:, self.channels_a] = a
        operation_result[:, self.channels_b] += b
        return operation_result[:, self.channels_out]


def add_replacer(onnx_math_op, masks: Tuple):
    """
    Function converting Addition to AddOperation
    """
    in_masks, out_masks, _ = masks
    return AddOperation(
        in_masks[0],
        in_masks[1],
        out_masks
    )


def reshape_replacer(reshape, masks):
    """
    Function replacing Reshape for pruned model
    """
    in_masks, out_mask, _ = masks
    reshape = copy.deepcopy(reshape)
    if hasattr(reshape.wrapped_module, 'shape'):
        rem = [convert_to_coarse_mask(out_mask, i)[1].shape[0]
               for i in range(1, out_mask.dim())]
        reshape.wrapped_module.shape = tuple(rem)
    return reshape


def expand_conversion(expand, masks):
    """
    Function replacing Expand for pruned model
    """
    in_masks, out_mask, _ = masks
    if hasattr(expand, 'const0'):
        expand = copy.deepcopy(expand)
        expand.const0 = torch.tensor(
            [1]+[convert_to_coarse_mask(out_mask, i)[1].shape[0]
                 for i in range(1, expand.const0.shape[0])])
    return expand


class TmpLoss(torch.nn.modules.loss._Loss):
    """
    Temporary loss for YOLOv4
    """

    def forward(self, input, target) -> torch.Tensor:
        return torch.mean(input[0]) + torch.mean(input[1]) + \
            torch.mean(input[2])


class Modes(str, Enum):
    """
    Enum containing possible pruners's modes.
    """

    NORMAL = "normal"
    """
    In this mode, pruner will not be aware of channel-dependency
    or group-dependency of  the model
    """

    DEPENDENCY = "dependency_aware"
    """
    In this mode, pruner will prune the model according to the activation-
    based metrics and the channel-dependency or group-dependency of the model
    """


class NNIPruningOptimizer(Optimizer):
    """
    The Neural Network Intelligence optimizer for activation base pruners,
    ActivationAPoZRankPruner and ActivationMeanRankPruner.
    """

    outputtypes = ["torch"]

    inputtypes = {"torch": torchconversion, "onnx": onnxconversion}

    prunertypes = {
        "apoz": ActivationAPoZRankPruner,
        "mean_rank": ActivationMeanRankPruner,
    }

    arguments_structure = {
        "modelframework": {
            "argparse_name": "--modelframework",
            "description": "The input type of the model, framework-wise",
            "default": "torch",
            "enum": list(inputtypes.keys()),
        },
        "finetuning_epochs": {
            "argparse_name": "--finetuning-epochs",
            "description": "Number of epochs model will be fine-tuning for",
            "type": int,
            "default": 3,
        },
        "pruner_type": {
            "argparse_name": "--pruner-type",
            "description": "Type of pruning algorithm",
            "type": str,
            "required": True,
            "enum": list(prunertypes.keys()),
        },
        "config_list": {
            "description": "Pruning specification, for more information please see NNI documentation - Compression Config Specification",  # noqa: E501
            "type": list,
            "items": object,
            "required": True,
        },
        "mode": {
            "description": "The mode for the pruner configuration",
            "default": Modes.NORMAL.value,
            "enum": [mode.value for mode in Modes],
        },
        "criterion": {
            "description": "Module path to class calculation loss",
            "type": str,
            "default": "torch.nn.CrossEntropyLoss",
        },
        "optimizer": {
            "description": "Module path to optimizer class",
            "type": str,
            "default": "torch.optim.SGD",
        },
        "finetuning_learning_rate": {
            "argparse_name": "--finetuning-learning-rate",
            "description": "Learning rate for fine-tuning",
            "type": float,
            "default": 0.001,
        },
        "finetuning_batch_size": {
            "argparse_name": "--finetuning-batch-size",
            "description": "Batch size for fine-tuning",
            "type": int,
            "default": 32
        },
        "training_steps": {
            "argparse_name": "--training-steps",
            "description": "The step number used to collect activations",
            "type": int,
            "required": True,
        },
        "activation": {
            "description": "Type of activation used in pruning algorithm",
            "type": str,
            "enum": ["relu", "gelu", "relu6"],
            "default": "relu",
        },
        "confidence": {
            "description": "The confidence coefficient of the sparsity inference, actually used as batch size for NNI model speedup. If not specified, equals finetuning_batch_size.",  # noqa: E501
            "type": int,
            "default": None,
        },
        "pruning_on_cuda": {
            "argparse_name": "--pruning-on-cuda",
            "description": "Allow pruning on CUDA GPU",
            "type": bool,
            "default": True,
        },
        "exclude_last_layer": {
            "argparse_name": "--exclude-last-layer",
            "description": "Exclude last Linear layer to preserve number of outputs",  # noqa: E501
            "type": bool,
            "default": True,
        },
    }

    str_to_dtype = {
        "int": torch.int,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float": torch.float,
        "float16": torch.float16,
        "float32": torch.float32,
        "float63": torch.float64,
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        pruner_type: str = list(prunertypes.keys())[0],
        config_list: List[Dict] = [{"sparsity_per_layer": 0.1, "op_types": ["Conv2d", "Linear"]}],  # noqa: E501
        training_steps: int = 1,
        mode: Optional[str] = Modes.NORMAL.value,
        criterion: str = "torch.nn.CrossEntropyLoss",
        optimizer: str = "torch.optim.SGD",
        finetuning_learning_rate: float = 0.001,
        finetuning_batch_size: int = 32,
        activation: str = "relu",
        model_framework: str = "torch",
        finetuning_epochs: int = 3,
        confidence: Optional[int] = None,
        pruning_on_cuda: bool = True,
        exclude_last_layer: bool = True,
    ):
        """
        The NNIPruning optimizer.

        This compiler applies pruning optimization base
        on Neural Network Intelligence framework.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to prune and fine-tune model.
        compiled_model_path: PathOrURI
            Path or URI where compiled model will be saved.
        pruner_type : str
            'apoz' or 'mean_rank' - to select ActivationAPoZRankPruner
            or ActivationMeanRankPruner.
        config_list : List[Dict]
            List of objects in JSON format, containing
            pruning specification, for more information please see
            NNI documentation - Compression Config Specification.
        training_steps : int
            The step number used to collect activation.
        mode : str
            'normal' or 'dependency_aware' - to select pruner mode.
        criterion : str
            Path to class calculating loss.
        optimizer : str
            Path to optimizer class.
        finetuning_learning_rate : float
            Learning rate for fine-tuning.
        finetuning_batch_size : int
            Batch size for fine-tuning.
        activation : str
            'relu', 'gelu' or 'relu6' - to select activation function
            used by pruner.
        model_framework : str
            Framework of the input model, used to select proper backend.
        finetuning_epochs: int
            Number of epoch used for fine-tuning model.
        confidence : int | None
            The confidence coefficient of the sparsity inference, actually
            used as batch size for NNI model speedup. If not specified, equals
            finetuning_batch_size.
        pruning_on_cuda : bool
            Allow using GPU CUDA for pruning.
        exclude_last_layer : bool
            Condition for excluding last linear layer from pruning.
        """
        super().__init__(
            dataset=dataset,
            compiled_model_path=compiled_model_path
        )

        self.criterion_modulepath = criterion
        self.optimizer_modulepath = optimizer
        self.finetuning_learning_rate = finetuning_learning_rate
        self.finetuning_batch_size = finetuning_batch_size
        if confidence is None:
            self.confidence = finetuning_batch_size
        else:
            self.confidence = confidence
        self.finetuning_epochs = finetuning_epochs
        self.training_steps = training_steps
        self.exclude_last_layer = exclude_last_layer
        self.set_activation_str(activation)

        self.model_framework = model_framework
        self.set_input_type(model_framework)

        self.pruner_type = pruner_type
        self.set_pruner_class(pruner_type)

        self.config_list = config_list
        self.set_pruner_mode(mode)

        self.prepare_dataloader_train_valid()

        self.pruning_on_cuda = pruning_on_cuda
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        nni_logger.setLevel(self.log.level)
        nni_graph_logger.setLevel(self.log.level)

        replace_module.update({
            'Add': add_replacer,
            'Reshape': reshape_replacer,
            'Expand': expand_conversion,
        })

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        model = self.inputtypes[self.inputtype](input_model_path, self.device)
        params_before = self.get_number_of_parameters(model)

        if self.exclude_last_layer:
            self.add_exclude_to_config(model)

        if not io_spec:
            io_spec = self.load_io_specification(input_model_path)
        self.io_spec = io_spec
        self.log.info(f"Model before pruning\n{model}")

        dummy_input = self.generate_dummy_input(io_spec)
        criterion = load_class(self.criterion_modulepath)()
        optimizer_cls = load_class(self.optimizer_modulepath)
        evaluator = self.create_evaluator(
            model, criterion, optimizer_cls, dummy_input
        )

        pruner: ActivationPruner = self.pruner_cls(
            model=model,
            config_list=self.config_list,
            evaluator=evaluator,
            training_steps=self.training_steps,
            activation=self.activation_str,
            mode=self.mode,
            dummy_input=dummy_input,
        )

        self.log.info("Pruning model")
        _, mask = pruner.compress()
        pruner._unwrap_model()

        if torch.cuda.is_available() and not self.pruning_on_cuda:
            model.to('cpu')
            dummy_input = dummy_input.to('cpu')
            for m_name, m_masks in mask.items():
                for w_name, w_mask in m_masks.items():
                    mask[m_name][w_name] = w_mask.to('cpu')

        try:
            ModelSpeedup(
                model,
                dummy_input=dummy_input,
                masks_file=mask,
                confidence=self.confidence,
            ).speedup_model()
        except Exception as ex:
            raise CompilationError from ex

        self.log.info(f"Model after pruning\n{model}\n")
        self.log.info(
            f"Parameters: {params_before:,} -> "
            f"{self.get_number_of_parameters(model):,}\n"
        )

        model.to(self.device)
        optimizer = optimizer_cls(model.parameters(),
                                  lr=self.finetuning_learning_rate)
        if self.log.level == logging.INFO and self.finetuning_epochs > 0:
            mean_loss = self.evaluate_model(model)
            self.log.info("Fine-tuning model starting with mean loss "
                          f"{mean_loss if mean_loss else None}\n")
        for finetuning_epoch in range(self.finetuning_epochs):
            self.train_model(
                model,
                optimizer,
                criterion,
                max_epochs=1)
            if self.log.level == logging.INFO:
                mean_loss = self.evaluate_model(model)
                self.log.info(
                    f"Epoch {finetuning_epoch+1} from {self.finetuning_epochs}"
                    f", validation data mean loss: {mean_loss}\n"
                )

        try:
            torch.save(model, self.compiled_model_path, pickle_module=dill)
        except Exception:
            self.log.error(
                "torch.save can't pickle full model, model parameters will be"
                " saved and dill will try to save full model")
            try:
                with open(self.compiled_model_path, 'wb') as fd:
                    dill.dump(model, fd)
            except Exception:
                torch.save(model.state_dict(), self.compiled_model_path)
                self.log.info("Only model's state dict saved "
                              f"to {self.compiled_model_path}")
            else:
                torch.save(model.state_dict(),
                           str(self.compiled_model_path)+'.state_dict')
                self.log.info(
                    f"Full model was saved to {self.compiled_model_path} "
                    "by `dill` and state dict was saved to "
                    f"{self.compiled_model_path}.state_dict")
        self.save_io_specification(input_model_path, io_spec)

    def train_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        lr_scheduler=None,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        The method used for training for one epoch

        This method is used by TorchEvaluator

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to train
        optimizer : torch.optim.Optimizer
            The instance of the optimizer class
        criterion :
            The callable object used to calculate loss
        lr_scheduler :
            The scheduler for learning rate manipulation
        max_steps : int | None
            The number of maximum steps - one step is equal to processing
            one batch of data
        max_epochs : int | None
            The number of maximum epochs
        args : Any
            Additional arguments.
        kwargs : Any
            Additional keyword arguments.
        """
        if max_steps is None and max_epochs is None:
            max_epochs = 5
        model.train()
        for batch_begin in tqdm(
            range(0, len(self.train_data[0]), self.finetuning_batch_size),
            file=LoggerProgressBar(),
        ):
            data, label = self.prepare_input_output_data(batch_begin)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if max_steps is not None:
                max_steps -= 1
                if max_steps == 0:
                    return

        if max_steps:
            self.log.info(f"{max_steps} steps left")
        if max_epochs is None:
            self.train_model(
                model,
                optimizer,
                criterion,
                max_steps=max_steps
            )
        elif max_epochs > 1:
            self.log.info(f"{max_epochs} epochs left")
            self.train_model(
                model,
                optimizer,
                criterion,
                max_steps=max_steps,
                max_epochs=max_epochs-1
            )

    def evaluate_model(self, model: torch.nn.Module):
        """
        The method used to evaluate model, by calculating mean of losses
        on test set

        This method is used by TorchEvaluator

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to evaluate

        Returns
        -------
        The model evaluation - mean of losses on test set
        """
        # TODO: For now evaluate mean loss, possible use of additional
        # parameter with modulepath to some torchmetrics or custom function
        criterion = load_class(self.criterion_modulepath)()
        model.eval()
        data_len = len(self.valid_data[0])
        loss_sum = 0
        with torch.no_grad():
            for batch_begin in tqdm(
                range(0, data_len, self.finetuning_batch_size),
                file=LoggerProgressBar(),
            ):
                data, target = self.prepare_input_output_data(batch_begin)
                output = model(data)
                loss: torch.Tensor = criterion(output, target)
                loss_sum += loss.sum(-1)
        return loss_sum / data_len

    def prepare_input_output_data(self, batch_begin: int):
        """
        The method used to prepare data in batche

        Parameters
        ----------
        batch_begin : int
            The index of the batch beginning

        Returns
        -------
        Prepared batch of data and targets
        """
        batch_x = self.train_data[0][
            batch_begin:batch_begin + self.finetuning_batch_size
        ]
        data = np.asarray(self.dataset.prepare_input_samples(batch_x))
        batch_y = self.train_data[1][
            batch_begin:batch_begin + self.finetuning_batch_size
        ]
        label = self.dataset.prepare_output_samples(batch_y)

        assert list(data.shape[1:]) == \
            list(self.io_spec['input'][0]['shape'][1:]), \
            f"Input data in shape {data.shape[1:]}, but only " \
            f"{self.io_spec['input'][0]['shape'][1:]} shape supported"

        data = torch.from_numpy(data).to(self.device)
        try:
            label = torch.from_numpy(np.asarray(label)).to(self.device)
        except TypeError:
            pass

        return data, label

    def prepare_dataloader_train_valid(self):
        """
        The method used to split dataset to train and validation set
        """
        Xt, Xv, Yt, Yv = self.dataset.train_test_split_representations()

        self.train_data = (Xt, Yt)
        self.valid_data = (Xv, Yv)

    def create_evaluator(
        self,
        model: torch.nn.Module,
        criterion: Callable,
        optimizer_cls: Type,
        dummy_input: torch.Tensor,
    ):
        """
        The method creating evaluator used during pruning

        Parameters
        ----------
        model : torch.nn.Module
            The input model which will be pruned
        criterion : Callable
            The callable object calculating loss
        optimizer_cls : Type
            The class of the optimizer
        dummy_input : torch.Tensor
            The tensor with random data suitable for model's input

        Returns
        -------
        The instance of TorchEvaluator
        """
        traced_optimizer = nni.trace(optimizer_cls)(
            model.parameters(), lr=self.finetuning_learning_rate
        )

        return TorchEvaluator(
            self.train_model,
            traced_optimizer,
            criterion,
            evaluating_func=self.evaluate_model,
            dummy_input=dummy_input,
        )

    def get_number_of_parameters(self, model: torch.nn.Module) -> int:
        """
        Get number of parameters in model

        Parameters
        ----------
        model : torch.nn.Module
            Model to get number of parameters

        Returns
        -------
        int :
            Number of parameters of model
        """
        return sum(p.numel() for p in model.parameters())

    def generate_dummy_input(
        self, io_spec: Dict[str, List[Dict]]
    ) -> torch.Tensor:
        """
        The method to generate dummy input used by pruner

        Parameters
        ----------
        io_spec : Dict[str, List[Dict]]
            The specification of model's input and output shape and type

        Returns
        -------
        The tensor with random data of shape suitable for model's input
        """
        inputs = io_spec["input"]
        assert (
            len(inputs) <= 1
        ), "NNI pruners only support dummy_input in form of one Tensor"
        assert len(inputs) >= 1, (
            "At least one"
            " input shape have to be specified to provide dummy input"
        )

        shape = inputs[0]["shape"]
        return torch.rand(
            [self.confidence, *shape[1:]],
            dtype=self.str_to_dtype[inputs[0]["dtype"]],
            device=self.device,
        )

    def add_exclude_to_config(self, model: torch.nn.Module):
        """
        The method appending config list with name of excluded last
        linear layer

        Parameters
        ----------
        model : torch.nn.Module
            Model which will be pruned
        """
        added = False
        for name, node in reversed(list(model.named_modules())):
            if isinstance(node, torch.nn.Linear):
                added = True
                self.config_list.append({
                    'exclude': True,
                    'op_names': [name]
                })
                break
        if not added:
            self.log.warning("Last Linear layer was not found -"
                             " cannot be excluded")

    def set_pruner_class(self, pruner_type):
        """
        The method used for choosing pruner class based on input string

        Parameters
        ----------
        pruner_type : str
            String with pruner label
        """
        assert pruner_type in self.prunertypes.keys(), (
            f"Unsupported pruner type {pruner_type}, only"
            f" {', '.join(self.prunertypes.keys())} are supported"
        )
        self.pruner_cls = self.prunertypes[pruner_type]

    def set_activation_str(self, activation):
        """
        The method used for selecting pruner activation based on input string

        Parameters
        ----------
        activation : str
            String with symbolic activation type
        """
        assert activation in self.arguments_structure["activation"]["enum"], (
            f"Unsupported pruner type {activation}, only"
            f" {', '.join(self.arguments_structure['activation']['enum'],)}"
            " are supported"
        )
        self.activation_str = activation

    def set_pruner_mode(self, mode):
        """
        The method used for selecting pruner mode based on input string

        Parameters
        ----------
        mode : str
            String with pruner mode
        """
        assert mode in self.arguments_structure["mode"]["enum"], (
            f"Unsupported pruner type {mode}, only"
            f" {', '.join(self.arguments_structure['mode']['enum'],)}"
            " are supported"
        )
        self.mode = mode

    def get_framework_and_version(self):
        return ("torch", torch.__version__)
