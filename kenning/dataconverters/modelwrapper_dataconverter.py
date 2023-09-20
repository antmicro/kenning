# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
A DataConverter-derived class used to manipulate the data using the
ModelWrapper object for compatibility between RuntimeProtocol and Dataset
during runtime.
"""

from kenning.core.dataconverter import DataConverter
from kenning.utils.class_loader import load_class
from typing import Any, Dict


class ModelWrapperDataConverter(DataConverter):

    arguments_structure: Dict[str, str] = {
        'model_wrapper': {
            'argparse_name': '--model-wrapper',
            'description': 'Import path to the model wrapper class',
            'type': str,
            'required': True,
        },
    }

    def __init__(self, model_wrapper: str):
        """
        Initializes the ModelWrapperDataConverter object.

        Parameters
        ----------
        model_wrapper : str
            The import path to the ModelWrapper class.
        """
        self.model_wrapper_cls = load_class(model_wrapper)
        super().__init__()

    def to_message(self, data: Any) -> bytes:
        """
        Converts the data to bytes using the ModelWrapper.

        Parameters
        ----------
        data : Any
            The data to be converted.

        Returns
        -------
        bytes :
            The converted data.
        """
        prepX = self.model_wrapper_cls._preprocess_input(data)
        return self.model_wrapper_cls.convert_input_to_bytes(prepX)

    def from_message(self, data: bytes) -> Any:
        """
        Converts the data from bytes using the ModelWrapper.

        Parameters
        ----------
        data : bytes
            The data to be converted.

        Returns
        -------
        Any :
            The converted data.
        """
        preds = self.model_wrapper_cls.convert_output_from_bytes(data)
        return self.model_wrapper_cls._postprocess_outputs(preds)
