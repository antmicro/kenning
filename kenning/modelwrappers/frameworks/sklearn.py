# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Generic ModelWrapper for models from scikit-learn framework.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from kenning.cli.command_template import TRAIN
from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError
from kenning.core.model import ModelWrapper
from kenning.datasets.tabular_dataset import TabularDataset
from kenning.utils.resource_manager import PathOrURI

# We use this data type as default, because of good support for this type in
# ONNX.
DEFAULT_SKLEARN_DTYPE = "int64"


class SKLearnModelWrapper(ModelWrapper, ABC):
    """
    Generic ModelWrapper for models from scikit-learn framework.
    """

    arguments_structure = {
        "dtype": {
            "argparse_name": "--dtype",
            "description": "Data type used in input/output of the model (eg. float32).",  # noqa: E501
            "type": str,
            "default": DEFAULT_SKLEARN_DTYPE,
            "subcommands": [TRAIN],
        },
    }

    default_dataset = TabularDataset

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool = False,
        model_name: Optional[str] = None,
        dtype: str = DEFAULT_SKLEARN_DTYPE,
    ):
        super().__init__(model_path, dataset, from_file, model_name)
        self.dtype = dtype

    @abstractmethod
    def get_model(self):
        ...

    def prepare_model(self):
        if self.model_prepared:
            return None
        if self.from_file:
            self.load_model(self.model_path)
            self.model_prepared = True
        else:
            self.model = self.get_model()
            self.model_prepared = True

    def load_model(self, model_path: PathOrURI):
        import joblib

        self.model = joblib.load(model_path)

    def save_model(self, model_path: PathOrURI):
        import joblib

        joblib.dump(self.model, model_path)

    def save_to_onnx(self, model_path: PathOrURI):
        raise NotSupportedError()

    @classmethod
    def get_output_formats(cls) -> List[str]:
        return ["sklearn"]

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        raise NotSupportedError()

    @classmethod
    def get_framework(cls) -> str:
        return "sklearn"

    @classmethod
    def get_framework_version(cls) -> str:
        import sklearn

        return sklearn.__version__

    def get_framework_and_version(self):
        import sklearn

        return ("sklearn", sklearn.__version__)
