# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from kenning.dataconverters.modelwrapper_dataconverter import (
    ModelWrapperDataConverter,
)
from kenning.core.model import ModelWrapper
from unittest.mock import patch


class TestModelWRapperDataConverter:
    @patch("kenning.core.model.ModelWrapper.__abstractmethods__", set())
    def test_core_modelwrapper_to_next_block(self):
        """
        Test if the ModelWrapperDataConverter properly calls the
        ModelWrapper's methods to preprocess the input and convert it to bytes.
        """
        model = ModelWrapper(None, None)
        with (
            patch.object(model, "_preprocess_input", return_value=2),
            patch.object(
                model, "convert_input_to_bytes", return_value=b"0x02"
            ),
        ):
            converter = ModelWrapperDataConverter(model)
            assert converter.to_next_block(1) == b"0x02"
            assert model._preprocess_input.called_once_with(1)
            assert model.convert_input_to_bytes.called_once_with(2)

    @patch("kenning.core.model.ModelWrapper.__abstractmethods__", set())
    def test_core_modelwrapper_to_previous_block(self):
        """
        Test if the ModelWrapperDataConverter properly calls the
        ModelWrapper's methods to postprocess the output and convert it from
        bytes.
        """
        model = ModelWrapper(None, None)
        with (
            patch.object(model, "_postprocess_outputs", return_value=2),
            patch.object(model, "convert_output_from_bytes", return_value=1),
        ):
            converter = ModelWrapperDataConverter(model)
            assert converter.to_previous_block(b"0x01") == 2
            assert model.convert_output_from_bytes.called_once_with(b"0x01")
            assert model._postprocess_outputs.called_once_with(1)
