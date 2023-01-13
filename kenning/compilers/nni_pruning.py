import torch
import nni
from nni.compression.pytorch.pruning import (
    ActivationAPoZRankPruner,
    ActivationMeanRankPruner,
)
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.algorithms.compression.v2.pytorch import TorchEvaluator
from nni.experiment import experiment_config
from pathlib import Path
import numpy as np
from typing import Callable, Dict, Optional, List, Type
from enum import Enum
import json
from tqdm import tqdm

from kenning.core.optimizer import Optimizer
from kenning.core.dataset import Dataset
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper
from kenning.utils.class_loader import load_class
from kenning.utils.logger import LoggerProgressBar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: add documentation


def torchconversion(
    modelpath: Path, model_wrapper_class: Optional[str] = None
):
    loaded = torch.load(modelpath, map_location=device)
    if not isinstance(loaded, torch.nn.Module):
        # Probably only temporary for testing - not user-friendly to
        # pass ModelWrapper more than one time
        assert model_wrapper_class, (
            "ModelWrapper class have to be known to convert"
            " OrderedDict to specific torch.nn.Module"
        )
        cls = load_class(model_wrapper_class)
        model_wrapper: PyTorchWrapper = cls(modelpath, None, False)
        model_wrapper.create_model_structure()
        model_wrapper.load_weights(loaded)
        model = model_wrapper.model
        # TODO: after test change to raise error
        # raise CompilationError(f'Expecting model of type:' \
        # f' torch.nn.Module, but got {type(model).__name__}')
    else:
        model = loaded
    return model


class Modes(str, Enum):
    """
    Enum containing possible pruners's modes.
    """

    NORMAL = "normal"
    DEPENDENCY = "dependency_aware"


class NNIPruningOptimizer(Optimizer):
    """
    The Neural Network Intelligence optimizer for activation base pruners,
    ActivationAPoZRankPruner and ActivationMeanRankPruner.
    """

    outputtypes = ["torch"]

    inputtypes = {"torch": torchconversion}

    prunertypes = {
        "apoz": ActivationAPoZRankPruner,
        "mean_rank": ActivationMeanRankPruner,
    }

    arguments_structure = {
        "modelframework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "torch",
            "enum": list(inputtypes.keys()),
        },
        "finetuning_epochs": {  # TODO: desc
            "argparse_name": "--finetuning_epochs",
            "description": "",
            "type": int,
            "default": 3,
        },
        "pruner_type": {
            "argparse_name": "--pruner-type",
            "description": "Pruning method",
            "type": str,
            "required": True,
            "enum": list(prunertypes.keys()),
        },
        "config_list": {
            "description": "Pruning specification, for more information please see NNI documentation - Compression Config Specification",  # noqa: E501
            "type": str,
            "required": True,
        },
        "mode": {  # TODO: description
            "description": "",
            "default": Modes.NORMAL.value,
            "enum": [mode.value for mode in Modes],
        },
        "criterion": {
            "description": "",
            "type": str,
            "default": "torch.nn.CrossEntropyLoss",
        },
        "optimizer": {
            "description": "",
            "type": str,
            "default": "torch.optim.SGD",
        },
        "learning_rate": {
            "argparse_name": "--learnign-rate",
            "description": "",
            "type": float,
            "default": 0.001,
        },
        "training_steps": {
            "argparse_name": "--training-steps",
            "description": "",
            "type": int,
            "required": True,
        },
        "activation": {
            "description": "",
            "type": str,
            "enum": ["relu", "gelu", "relu6"],
            "default": "relu",
        },
        "model_wrapper_type": {  # Probably temporary
            "argparse_name": "--model-wrapper-type",
            "description": "Copy of modelwrapper.type",
            "type": str,
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
        compiled_model_path: Path,
        pruner_type: str,
        config_list: str,
        training_steps: int,
        mode: Optional[str] = Modes.NORMAL.value,
        criterion: str = "torch.nn.CrossEntropyLoss",
        optimizer: str = "torch.optim.lr_scheduler.MultiStepLR",
        learning_rate: float = 0.001,
        activation: Optional[str] = None,
        modelframework: str = "torch",
        finetuning_epochs: int = 3,
        model_wrapper_type: Optional[str] = None,
    ):
        super().__init__(dataset, compiled_model_path)

        self.criterion_modulepath = criterion
        self.optimizer_modulepath = optimizer
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.activation_str = activation

        self.modelframework = modelframework
        self.finetuning_epochs = finetuning_epochs
        self.set_input_type(modelframework)

        self.pruner_type = pruner_type
        self.set_pruner_class(pruner_type)

        # TODO: for now pass List[Dict] as str in json, upgrade argparse
        self.config_list = json.loads(config_list)
        self.mode = mode

        self.prepare_dataloader_train_valid()

        self.model_wrapper_type = model_wrapper_type

        # TODO: set NNI logger to specified level
        experiment_config.logging.root = self.log

    def compile(
        self,
        inputmodelpath: Path,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        model = self.inputtypes[self.inputtype](
            inputmodelpath, self.model_wrapper_type
        )

        if not io_spec:
            io_spec = self.load_io_specification(inputmodelpath)

        dummy_input = self.generate_dummy_input(io_spec)
        criterion = load_class(self.criterion_modulepath)()
        optimizer_cls = load_class(self.optimizer_modulepath)
        evaluator = self.create_evaluator(
            model, criterion, optimizer_cls, dummy_input
        )
        pruner = self.pruner_cls(
            model,
            self.config_list,
            evaluator,
            self.training_steps,
            self.activation_str,
            self.mode,
            dummy_input,
        )

        self.log.info("Pruning model")
        _, mask = pruner.compress()
        pruner._unwrap_model()

        ModelSpeedup(
            model,
            dummy_input=dummy_input,
            masks_file=mask,
        ).speedup_model()

        optimizer = optimizer_cls(model.parameters(), lr=self.learning_rate)
        for finetuning_epoch in range(self.finetuning_epochs):
            self.log.info(
                f"Fine-tuning pruned model - epoch {finetuning_epoch+1}"
            )
            self.train_model(model, optimizer, criterion)

        torch.save(model, self.compiled_model_path)

    def train_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        *args,
        **kwargs,
    ):
        model.train()
        for batch_begin in tqdm(
            range(0, len(self.train_data[0]), self.dataset.batch_size),
            file=LoggerProgressBar(),
        ):
            data, label = self.prepare_input_output_data(batch_begin)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

    def evaluate_model(self, model: torch.nn.Module):
        # TODO: For now evaluate mean loss, possible use of additional
        # parameter with modulepath to some torchmetrics or custom function
        criterion = load_class(self.criterion_modulepath)()
        model.eval()
        data_len = len(self.valid_data[0])
        loss_sum = 0
        with torch.no_grad():
            for batch_begin in tqdm(
                range(0, data_len, self.dataset.batch_size),
                file=LoggerProgressBar(),
            ):
                data, target = self.prepare_input_output_data(batch_begin)
                output = model(data)
                loss: torch.Tensor = criterion(output, target)
                loss_sum += loss.sum(-1)
        return loss_sum / data_len

    def prepare_input_output_data(self, batch_begin):
        batch_x = self.train_data[0][
            batch_begin:batch_begin + self.dataset.batch_size
        ]
        data = np.asarray(self.dataset.prepare_input_samples(batch_x))
        batch_y = self.train_data[1][
            batch_begin:batch_begin + self.dataset.batch_size
        ]
        label = np.asarray(self.dataset.prepare_output_samples(batch_y))
        # TODO: I assume right NHWC/NCHW is choosen
        data = torch.from_numpy(data).to(device)
        label = torch.from_numpy(label).to(device)
        return data, label

    def prepare_dataloader_train_valid(self):
        # TODO: I assume the same split like when pretraining
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
        traced_optimizer = nni.trace(optimizer_cls)(
            model.parameters(), lr=self.learning_rate
        )

        return TorchEvaluator(
            self.train_model,
            traced_optimizer,
            criterion,
            evaluating_func=self.evaluate_model,
            dummy_input=dummy_input,
        )

    def generate_dummy_input(
        self, io_spec: Dict[str, List[Dict]]
    ) -> torch.Tensor:
        inputs = io_spec["input"]
        assert (
            len(inputs) <= 1
        ), "NNI pruners only support dummy_input in form of one Tensor"
        assert len(inputs) >= 1, (
            "At least one"
            " input shape have to be specified to provide dummy input"
        )
        return torch.rand(
            inputs[0]["shape"],
            dtype=self.str_to_dtype[inputs[0]["dtype"]],
            device=device,
        )

    def set_pruner_class(self, pruner_type):
        assert pruner_type in self.prunertypes.keys(), (
            f"Unsupported pruner type {pruner_type}, only"
            " {', '.join(self.prunertypes.keys())} are supported"
        )
        self.pruner_cls = self.prunertypes[pruner_type]

    def get_framework_and_version(self):
        return ("torch", torch.__version__)

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.pruner_type,
            args.config_list,
            args.training_steps,
            args.mode,
            args.criterion,
            args.optimizer,
            args.learning_rate,
            args.activation,
            args.model_framework,
            args.finetuning_epochs,
            args.model_wrapper_type,
        )
