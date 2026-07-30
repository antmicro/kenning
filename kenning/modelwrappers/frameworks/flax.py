# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides base methods for using Flax models in Kenning.
"""

import tarfile
from abc import ABC
from typing import Any, List, Optional

import numpy as np

from kenning.cli.command_template import TRAIN
from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError
from kenning.core.model import ModelWrapper
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


class FlaxWrapper(ModelWrapper, ABC):
    """
    Base model wrapper for Flax models.
    """

    arguments_structure = {
        "batch_size": {
            "argparse_name": "--batch-size",
            "description": "Batch size for training. If not assigned, dataset batch size will be used.",  # noqa: E501
            "type": int,
            "default": 64,
            "subcommands": [TRAIN],
        },
        "learning_rate": {
            "description": "Learning rate for training",
            "type": float,
            "default": 1e-3,
            "subcommands": [TRAIN],
        },
        "num_epochs": {
            "argparse_name": "--num-epochs",
            "description": "Number of epochs to train for",
            "type": int,
            "default": 50,
            "subcommands": [TRAIN],
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool = True,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = 64,
        learning_rate: Optional[float] = 1e-3,
        num_epochs: Optional[int] = 50,
    ):
        self.model = None
        super().__init__(model_path, dataset, from_file, model_name)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def create_model_structure(self):
        """
        Recreates the model structure.
        Every FlaxWrapper subclass has to implement its own architecture.
        """
        raise NotSupportedError

    def load_model(self, model_path: PathOrURI):
        if self.model is None:
            self.create_model_structure()

        if model_path.suffix == ".tar":
            model_tar = tarfile.TarFile(model_path.absolute())

            # strip .tar suffix
            model_path = model_path.with_suffix("")
            KLogger.debug(f"Extracting model tar to {model_path}")
            model_tar.extractall(model_path)

        self.load_weights(model_path)

    def load_weights(self, model_path: PathOrURI):
        import orbax.checkpoint as ocp
        from flax import nnx

        params = nnx.state(self.model, nnx.Param)
        params_dict = nnx.to_pure_dict(params)

        ckptr = ocp.StandardCheckpointer()
        restored = ckptr.restore(model_path.resolve(), params_dict)

        nnx.replace_by_pure_dict(params, restored)
        nnx.update(self.model, params)

        ckptr.close()

    def save_model(self, model_path: PathOrURI):
        import orbax.checkpoint as ocp
        from flax import nnx

        params = nnx.state(self.model, nnx.Param)
        params_dict = nnx.to_pure_dict(params)

        ckptr = ocp.StandardCheckpointer()
        ckptr.save(model_path.resolve(), params_dict, force=True)

        ckptr.close()

    def save_to_onnx(self, model_path: PathOrURI):
        raise NotSupportedError

    @classmethod
    def get_framework(cls) -> str:
        return "flax"

    @classmethod
    def get_framework_version(cls) -> str:
        import flax

        return flax.__version__

    @classmethod
    def get_output_formats(cls):
        return ["flax"]

    def run_inference(self, X: List[Any]) -> List[Any]:
        self.prepare_model()

        y = self.model(*X)

        return [np.asarray(y)]

    def convert_input_to_bytes(self, inputdata: List[Any]) -> bytes:
        data = bytes()
        for inp in inputdata:
            for x in inp:
                data += x.tobytes()
        return data

    def convert_output_from_bytes(self, outputdata: bytes) -> List[Any]:
        out_spec = self.get_io_specification()["output"]

        result = []
        data_idx = 0
        for spec in out_spec:
            dtype = np.dtype(spec["dtype"])
            shape = spec["shape"]

            out_size = np.prod(shape) * np.dtype(dtype).itemsize
            arr = np.frombuffer(
                outputdata[data_idx : data_idx + out_size], dtype=dtype
            )
            data_idx += out_size
            result.append(arr.reshape(shape))

        return result
