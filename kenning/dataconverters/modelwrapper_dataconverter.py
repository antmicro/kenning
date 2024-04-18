# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A DataConverter-derived class used to manipulate the data using the
ModelWrapper object for compatibility with surrounding block
during runtime.
"""

from typing import Any, Dict, List

from kenning.core.dataconverter import DataConverter
from kenning.core.model import ModelWrapper
from kenning.utils.args_manager import get_parsed_json_dict
from kenning.utils.class_loader import load_class


class ModelWrapperDataConverter(DataConverter):
    """
    A DataConverter based on the ModelWrapper object.
    """

    def __init__(self, model_wrapper: ModelWrapper):
        """
        Initializes the ModelWrapperDataConverter object.

        Parameters
        ----------
        model_wrapper : ModelWrapper
            The ModelWrapper object used to convert the data.
        """
        self.model_wrapper = model_wrapper
        super().__init__()

    def to_next_block(self, data: List[Any]) -> bytes:
        """
        Converts the data to bytes using the ModelWrapper.

        Parameters
        ----------
        data : List[Any]
            The data to be converted.

        Returns
        -------
        bytes
            The converted data.
        """
        return self.model_wrapper._preprocess_input(data)

    def to_previous_block(self, data: bytes) -> List[Any]:
        """
        Converts the data from bytes using the ModelWrapper.

        Parameters
        ----------
        data : bytes
            The data to be converted.

        Returns
        -------
        List[Any]
            The converted data.
        """
        return self.model_wrapper._postprocess_outputs(data)

    @classmethod
    def from_json(
        cls, json_dict: Dict[str, str]
    ) -> "ModelWrapperDataConverter":
        """
        Creates the ModelWrapperDataConverter object from the JSON
        configuration.

        Parameters
        ----------
        json_dict : Dict[str, str]
            The JSON dictionary containing the configuration.

        Returns
        -------
        ModelWrapperDataConverter
            The created ModelWrapperDataConverter object.
        """
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        modelwrapper_cfg = parsed_json_dict["model_wrapper"]
        modelwrapper_cls = load_class(modelwrapper_cfg["type"])
        modelwrapper = modelwrapper_cls.from_json(
            None, modelwrapper_cfg["parameters"]
        )
        return cls(modelwrapper)
