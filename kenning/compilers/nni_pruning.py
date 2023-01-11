from nni.compression.pytorch.speedup import ModelSpeedup
import torch
from nni.compression.pytorch.pruning import (
        LevelPruner,
        L1NormPruner,
        L2NormPruner,
        FPGMPruner,
        SlimPruner,
        ActivationAPoZRankPruner,
        ActivationMeanRankPruner,
        TaylorFOWeightPruner,
        ADMMPruner,
        LinearPruner,
        AGPPruner,
        LotteryTicketPruner,
        SimulatedAnnealingPruner,
        AutoCompressPruner,
        AMCPruner,
        MovementPruner
)
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from enum import Enum

# from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer  # , CompilationError
from kenning.core.dataset import Dataset
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper
from kenning.utils.class_loader import load_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: add documentation


def torchconversion(
    modelpath: Path, model_wrapper_class: Optional[str] = None
):
    loaded = torch.load(modelpath, map_location=device)
    if not isinstance(loaded, torch.nn.Module):
        # Probably only temporary for testing - not user-friendly to
        # pass ModelWrapper more than one time
        assert model_wrapper_class, \
            "ModelWrapper class have to be known to convert" \
            " OrderedDict to specific torch.nn.Module"
        cls = load_class(model_wrapper_class)
        model_wrapper: PyTorchWrapper = cls(modelpath, None, False)
        model_wrapper.create_model_structure()
        model_wrapper.load_weights(loaded)
        model = model_wrapper.model
        # raise CompilationError(f'Expecting model of type:' \
        # f' torch.nn.Module, but got {type(model).__name__}')
    else:
        model = loaded
    return model


class Modes(str, Enum):
    NORMAL = "normal"
    DEPENDENCY = ("dependency_aware",)
    GLOBAL = "global"


class NNIPruningOptimizer(Optimizer):
    """
    The Neural Network Intelligence optimizer.
    """

    outputtypes = ["torch"]

    inputtypes = {"torch": torchconversion}

    basicprunertypes = {
        "level": LevelPruner,
        "l1": L1NormPruner,
        "l2": L2NormPruner,
        "fpgm": FPGMPruner,
        "slim": SlimPruner,
        "apoz": ActivationAPoZRankPruner,
        "mean_rank": ActivationMeanRankPruner,
        "taylorfo": TaylorFOWeightPruner,
        "admm": ADMMPruner,
    }
    evaluationprunertypes = {}
    scheduledprunertypes = {
        "linear": LinearPruner,
        "agp": AGPPruner,
        "lottery_ticket": LotteryTicketPruner,
        "simulated_annealing": SimulatedAnnealingPruner,
        "auto_compress": AutoCompressPruner,
        "amc": AMCPruner,
    }
    movementprunertypes = {"movement": MovementPruner}
    prunertypes = dict(
        **basicprunertypes,
        **evaluationprunertypes,
        **scheduledprunertypes,
        **movementprunertypes,
    )

    # modetypes = ['normal', 'dependency_aware']

    arguments_structure = {
        "modelframework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "torch",
            "enum": list(inputtypes.keys()),
        },
        "speedup": {  # TODO: implement + desc
            "description": "",
            "type": bool,
            "default": True,
        },
        "finetuning": {  # TODO: implement + desc
            "description": "",
            "type": bool,
            "default": True,
        },
        "pruner_type": {
            "description": "Pruning method",
            "required": True,
            "enum": list(prunertypes.keys()),
        },
        "config_list": {  # TODO: try if dicts parameters is ok
            "description": "Pruning specification, for more information please see NNI documentation - Compression Config Specification",  # noqa E501
            "type": list,
            "required": True,
        },
        "mode": {  # TODO: description
            "description": "",
            "default": Modes.NORMAL.value,
            "enum": [mode.value for mode in Modes],
        },
        "model_wrapper_type": {  # Probably temporary
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
        config_list: List[Dict],
        mode: Optional[str] = Modes.NORMAL.value,
        modelframework: str = "torch",
        speedup: bool = True,
        finetuning: bool = True,
        model_wrapper_type: Optional[str] = None,
    ):
        self.modelframework = modelframework
        self.speedup = speedup
        self.finetuning = finetuning
        self.set_input_type(modelframework)
        self.pruner_type = pruner_type
        self.set_pruner_class(pruner_type)
        self.set_pruner_possible_args(pruner_type)
        self.config_list = config_list
        self.mode = mode

        self.run_pruner = self.chose_run_pruner(pruner_type)

        self.model_wrapper_type = model_wrapper_type

        # TODO: remove
        self.config_list = [{"sparsity": 0.8, "op_types": ["Conv2d"]}]
        print(self.config_list)
        super().__init__(dataset, compiled_model_path)

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

        argumetns = self.create_arguments_dict()
        argumetns["model"] = model
        mask = self.run_pruner(model, io_spec, argumetns)

        if self.speedup:
            ModelSpeedup(
                model,
                dummy_input=self.generate_dummy_input(io_spec),
                masks_file=mask,
            ).speedup_model()
            model.eval()

        if self.finetuning:
            pass

        torch.save(model.state_dict(), self.compiled_model_path)

    def _run_basic_pruner(
        self,
        model: torch.nn.Module,
        io_spec: Dict[str, List[Dict]],
        arguments: Dict,
    ):
        """
        Returns
        -------
        mask
            Mask aquired in pruning proces
        """
        self.add_dummy_input_argument(arguments, io_spec)
        print("@@@@@@@@@@@@@", arguments)
        pruner = self.pruner_class(**arguments)
        _, mask = pruner.compress()
        pruner._unwrap_model()
        return mask

    def _run_evaluation_pruner(
        self,
        model: torch.nn.Module,
        io_spec: Dict[str, List[Dict]],
        arguments: Dict,
    ):
        raise NotImplementedError

    def _run_scheduled_pruner(
        self,
        model: torch.nn.Module,
        io_spec: Dict[str, List[Dict]],
        arguments: Dict,
    ):
        raise NotImplementedError

    def _run_movement_pruner(
        self,
        model: torch.nn.Module,
        io_spec: Dict[str, List[Dict]],
        arguments: Dict,
    ):
        raise NotImplementedError

    def generate_dummy_input(
        self, io_spec: Dict[str, List[Dict]]
    ) -> torch.Tensor:
        inputs = io_spec["input"]
        print(inputs, len(inputs))
        assert (
            len(inputs) <= 1
        ), "NNI pruners only support dummy_input in form of one Tensor"
        assert len(inputs) >= 1, "At least one input " \
            "shape have to be specified to provide dummy input"
        return torch.rand(
            inputs[0]["shape"],
            dtype=self.str_to_dtype[inputs[0]["dtype"]],
            device=device,
        )

    def add_dummy_input_argument(
        self, arguments: Dict, io_spec: Dict[str, List[Dict]]
    ):
        if (
            self.mode == Modes.DEPENDENCY.value
            and "dummy_input" in self.pruner_possible_args
        ):
            arguments["dummy_input"] = self.generate_dummy_input(io_spec)

    def chose_run_pruner(self, pruner_type):
        types = (
            self.basicprunertypes,
            self.evaluationprunertypes,
            self.scheduledprunertypes,
            self.movementprunertypes,
        )
        functions = (
            self._run_basic_pruner,
            self._run_evaluation_pruner,
            self._run_scheduled_pruner,
            self._run_movement_pruner,
        )
        for _type, _function in zip(types, functions):
            if pruner_type in _type.keys():
                return _function
        raise KeyError(f"Wrong pruner type: {pruner_type}")

    def create_arguments_dict(self) -> Dict:
        possible_arguments_names: Tuple = (
            self.pruner_class.__init__.__code__.co_varnames
        )
        return {
            argument_name: argument_value
            for (argument_name, argument_value) in self.__dict__.items()
            if argument_name in possible_arguments_names
        }

    def set_pruner_class(self, pruner_type):
        assert pruner_type in self.prunertypes.keys(), (
            f"Unsupported pruner type {pruner_type}, only"
            " {', '.join(self.prunertypes.keys())} are supported"
        )
        self.pruner_class = self.prunertypes[self.pruner_type]

    def set_pruner_possible_args(self, pruner_type):
        self.pruner_possible_args = (
            self.pruner_class.__init__.__code__.co_varnames
        )

    def get_framework_and_version(self):
        return ("torch", torch.__version__)

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.pruner_type,
            args.config_list,
            args.mode,
            args.model_framework.args.speedup,
            args.finetuning,
            args.model_wrapper_type,
        )
