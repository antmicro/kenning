# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from kenning.core.model import ModelWrapper
from kenning.dataconverters.modelwrapper_dataconverter import (
    ModelWrapperDataConverter,
)


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
        ):
            converter = ModelWrapperDataConverter(model)
            assert converter.to_next_block(1) == 2
            assert model._preprocess_input.called_once_with(1)

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
        ):
            converter = ModelWrapperDataConverter(model)
            assert converter.to_previous_block(1) == 2
            assert model._postprocess_outputs.called_once_with(1)
