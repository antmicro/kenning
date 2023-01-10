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
)
from pathlib import Path
from typing import Dict, Optional, List

from kenning.core.optimizer import Optimizer
from kenning.core.dataset import Dataset
from kenning.modelwrappers.classification.pytorch_pet_dataset import (
    PyTorchPetDatasetMobileNetV2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: add documentation


def torchconversion(modelpath: Path):
    # TODO: generalize
    wrapper = PyTorchPetDatasetMobileNetV2(modelpath, None)
    wrapper.load_model(modelpath)
    # model = torch.load(modelpath, map_location=device)
    return wrapper.model


class NNIPruningOptimizer(Optimizer):
    """
    The Neural Network Intelligence optimizer.
    """

    outputtypes = ["torch"]

    inputtypes = {"torch": torchconversion}

    prunertypes = {
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

    modetypes = ["normal", "dependency_aware"]

    arguments_structure = {
        "modelframework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "torch",
            "enum": list(inputtypes.keys()),
        },
        "speedup": {  # TODO: implement
            "description": "",
            "type": bool,
            "default": True,
        },
        "finetuning": {  # TODO: implement
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
            "default": modetypes[0],
            "enum": modetypes,
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: Path,
        pruner_type: str,
        config_list: List[Dict],
        mode: Optional[str] = "normal",
        modelframework: str = "torch",
        speedup: bool = True,
        finetuning: bool = True,
    ):
        self.modelframework = modelframework
        self.speedup = speedup
        self.finetuning = finetuning
        self.set_input_type(modelframework)
        self.pruner_type = pruner_type
        self.set_pruner_class(pruner_type)
        self.config_list = config_list
        self.mode = mode

        # TODO: remove
        self.config_list = [{"sparsity": 0.8, "op_types": ["Conv2d"]}]
        print(self.config_list)
        print("@@@@@@@@@@@@@@")
        super().__init__(dataset, compiled_model_path)

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.pruner_type,
            args.config_list,
            args.mode,
            args.model_framework,
        )

    def compile(
        self,
        inputmodelpath: Path,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        model = self.inputtypes[self.inputtype](inputmodelpath)
        # print('!!!', type(model))

        # TODO: if 'dependency_aware' must have a dummy_input
        pruner = self.pruner_class(model, self.config_list)
        _, mask = pruner.compress()
        pruner._unwrap_model()

        if self.speedup:
            pass

        if self.finetuning:
            pass

        # print('!!!!!!', type(model))
        torch.save(model.state_dict(), self.compiled_model_path)

    def set_pruner_class(self, pruner_type):
        assert pruner_type in self.prunertypes.keys(), \
            f"Unsupported pruner type {pruner_type}, only" \
            " {', '.join(self.prunertypes.keys())} are supported"
        self.pruner_class = self.prunertypes[self.pruner_type]

    def get_framework_and_version(self):
        return ("torch", torch.__version__)


class NNIScheduledPrunerOptimizer(Optimizer):

    """,
        'linear': LinearPruner,
        'agp': AGPPruner,
        'lottery_ticket': LotteryTicketPruner,
        'simulated_annealing': SimulatedAnnealingPruner,
        'auto_compress': AutoCompressPruner,
        'amc': AMCPruner,
        'movement': MovementPruner
    }"""

    pass
