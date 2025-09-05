# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains Tinygrad models for the classification problem.

Pretrained on ImageNet dataset.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError, ParametersMismatchError
from kenning.datasets.imagenet_dataset import ImageNetDataset
from kenning.modelwrappers.frameworks.tinygrad import TinygradWrapper
from kenning.utils.resource_manager import PathOrURI


class TinygradImageNet(TinygradWrapper):
    """
    General-purpose model wrapper for ImageNet models in tinygrad.
    """

    default_dataset = ImageNetDataset
    pretrained_model_uri = "kenning:///models/classification/tinygrad_imagenet_resnet50.safetensors"
    arguments_structure = {
        "inputshape": {
            "argparse_name": "--input-shape",
            "description": "Input shape",
            "type": list[int],
            "default": [1, 224, 224, 3],
        },
        "numclasses": {
            "argparse_name": "--num-classes",
            "description": "Output shape",
            "type": int,
            "default": 1000,
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        model_name: Optional[str] = None,
        from_file: bool = True,
        model_file: str = "https://raw.githubusercontent.com/tinygrad/tinygrad/bcc7623025d39f4994eab0394beb83662d879ec8/extra/models/resnet.py",
        modelcls: str = "ResNet50",
        inputshape: List[int] = [1, 224, 224, 3],
        numclasses: int = 1000,
    ):
        """
        Creates model wrapper for Tinygrad classification model pretrained on
        ImageNet dataset.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        dataset : Dataset
            The dataset to verify the inference.
        model_name : Optional[str]
            Name of the model used for the report.
        from_file : bool
            True if model should be loaded from file.
        model_file : str
            File containing model implementation.
        modelcls : str
            The model class.
        inputshape : List[int]
            The shape of the input.
        numclasses : int
            Number of classes in the model.

        Raises
        ------
        ParametersMismatchError
            Raised when dataset batch size and ModelWrapper supported
            sizes do not match
        """
        super().__init__(
            model_path, dataset, model_file, modelcls, from_file, model_name
        )
        self.modelcls = modelcls
        self.modelinputname = "input"
        self.modeloutputname = "output"
        self.inputshape = inputshape
        self.numclasses = numclasses
        self.outputshape = [inputshape[0], numclasses]

        if dataset and dataset.batch_size != self.inputshape[0]:
            raise ParametersMismatchError(
                ["dataset.batch_size", "modelwrapper.inputshape[0]"]
            )

    @classmethod
    def _get_io_specification(
        cls,
        modelinputname: str,
        inputshape: Tuple[int],
        modeloutputname: str,
        outputshape: Tuple[int],
    ) -> Dict[str, List[Dict]]:
        return {
            "input": [
                {
                    "name": modelinputname,
                    "shape": inputshape,
                    "dtype": "float",
                }
            ],
            "processed_input": [
                {
                    "name": "input_1",
                    "shape": (inputshape[0], 3, 224, 224),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": modeloutputname,
                    "shape": outputshape,
                    "dtype": "float",
                }
            ],
        }

    @classmethod
    def derive_io_spec_from_json_params(
        cls, json_dict: Dict
    ) -> Dict[str, List[Dict]]:
        input_shape = json_dict["inputshape"]
        output_shape = [input_shape[0], json_dict["numclasses"]]
        return cls._get_io_specification(
            json_dict["modelinputname"],
            input_shape,
            json_dict["modeloutputname"],
            output_shape,
        )

    def get_io_specification_from_model(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification(
            self.modelinputname,
            self.inputshape,
            self.modeloutputname,
            self.outputshape,
        )

    def preprocess_input(self, X: List[np.ndarray]) -> List[Any]:
        if self.dataset.image_memory_layout == "NHWC":
            X = [x.transpose(0, 3, 1, 2) for x in X]
        return X

    def prepare_model(self):
        if self.model_prepared:
            return None
        if self.from_file:
            self.load_model(self.model_path)
            self.model_prepared = True
        else:
            raise NotSupportedError(
                "TinygradImageNet only supports loading model from a file."
            )

    def train_model(self):
        raise NotSupportedError("This model does not support training.")
