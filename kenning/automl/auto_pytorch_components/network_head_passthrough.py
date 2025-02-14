# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing Passthrough head network compatible with AutoPyTorch.
"""

from typing import Dict, Optional, Tuple, Union

import ConfigSpace as CS
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_head import add_head
from autoPyTorch.pipeline.components.setup.network_head.base_network_head import (  # noqa: E501
    NetworkHeadComponent,
)
from torch import nn

_REGISTERED = False


class Passthrough(nn.Module):
    """
    Module representing identity function.
    """

    def forward(self, x):
        return x


class PassthroughHead(NetworkHeadComponent):
    """
    Head returning data without a change.
    """

    def build_head(
        self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]
    ) -> nn.Module:
        """
        Initializes the module representing the head component.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            The shape of input data.
        output_shape : Tuple[int, ...]
            The shape of output data.

        Returns
        -------
        nn.Module
            Created PyTorch model.
        """
        return Passthrough()

    @staticmethod
    def get_properties(
        dataset_properties: Optional[
            Dict[str, BaseDatasetPropertiesType]
        ] = None,
    ) -> Dict[str, Union[str, bool]]:
        """
        Returns properties of head component.

        Parameters
        ----------
        dataset_properties : Optional[Dict[str, BaseDatasetPropertiesType]]
            Properties of used dataset, provided by AutoPyTorch.

        Returns
        -------
        Dict[str, Union[str, bool]]
            Component properties
        """
        return {
            "shortname": "PassthroughHead",
            "name": "PassthroughHead",
            "handles_tabular": True,
            "handles_image": True,
            "handles_time_series": True,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[
            Dict[str, BaseDatasetPropertiesType]
        ] = None,
    ) -> CS.ConfigurationSpace:
        """
        Generates Configuration Space for Passthrough head.

        Parameters
        ----------
        dataset_properties : Optional[Dict[str, BaseDatasetPropertiesType]]
            Properties of used dataset, provided by AutoPyTorch.

        Returns
        -------
        CS.ConfigurationSpace
            Configuration Space for head.
        """
        return CS.ConfigurationSpace()


def register_passthrough():
    """
    Registers PassthroughHead, only if it is not already registered.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    add_head(PassthroughHead)
    _REGISTERED = True
