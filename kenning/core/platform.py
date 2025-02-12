# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a wrapper for platforms.
"""

import sys
from abc import ABC
from time import perf_counter
from typing import Optional

import yaml

from kenning.core.measurements import Measurements
from kenning.core.protocol import Protocol
from kenning.resources import platforms
from kenning.utils.args_manager import ArgumentsHandler
from kenning.utils.logger import KLogger

if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path


class Platform(ArgumentsHandler, ABC):
    """
    Wraps the platform that is being evaluated. This class provides methods to
    handle tested platform. The platform can be the device that Kenning is run
    on, a board running Kenning Zephyr Runtime or bare-metal IREE runtime or
    any remote or device running Kenning inference server.
    """

    needs_protocol = False

    arguments_structure = {
        "name": {
            "description": "Name of the platform",
            "type": str,
            "nullable": True,
            "default": None,
        },
    }

    # default values for data that is read from platforms.yml
    platform_defaults = {}

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        """
        Constructs platform and reads its data from platforms.yaml.

        Parameters
        ----------
        name : Optional[str]
            Name of the platform.
        """
        self.name = name
        self.read_data_from_platforms_yaml()

    def init(self):
        """
        Initializes the platform.
        """
        ...

    def read_data_from_platforms_yaml(self):
        """
        Retrieves platform data from platforms.yml.
        """
        with path(platforms, "platforms.yml") as platforms_path:
            with open(platforms_path, "r") as platforms_yaml:
                platforms_dict = yaml.safe_load(platforms_yaml)

            if self.name in platforms_dict:
                platform_data = platforms_dict[self.name]
            else:
                platform_data = {}
                KLogger.warning(
                    f"Platform {self.name} not found in {platforms_path}"
                )

            default_attrs = self.__class__.platform_defaults
            all_attrs = set(platform_data.keys()).union(default_attrs.keys())

            for attr_name in all_attrs:
                if getattr(self, attr_name, None) is not None:
                    continue

                attr_srcs = (
                    (platform_data, str(platforms_path)),
                    (default_attrs, "defaults"),
                )
                param_schema = self.form_parameterschema()
                for attr_src, attr_src_name in attr_srcs:
                    if attr_src.get(attr_name, None) is None:
                        continue
                    attr_value = attr_src[attr_name]
                    arg_struct = param_schema["properties"].get(
                        attr_name, None
                    )
                    if arg_struct is not None:
                        attr_type = arg_struct["convert-type"]
                        if "array" in arg_struct.get("type", []):
                            attr_value = [attr_type(v) for v in attr_value]
                        else:
                            attr_value = attr_type(attr_value)
                    setattr(self, attr_name, attr_value)
                    KLogger.debug(
                        f"Setting {attr_name}={attr_value} from {attr_src_name}"  # noqa: E501
                    )
                    break

    def deinit(self, measurements: Optional[Measurements] = None):
        """
        Deinitializes platform.

        Parameters
        ----------
        measurements : Optional[Measurements]
            Measurements to which platform metrics can be added to.
        """
        ...

    def get_time(self) -> float:
        """
        Retrieves elapsed time from platform.

        Returns
        -------
        float
            Elapsed time.
        """
        return perf_counter()

    def get_default_protocol(self) -> Protocol:
        """
        Returns default protocol for given platform.

        Returns
        -------
        Protocol
            Default protocol for given platform.
        """
        return None

    def inference_step_callback(self):
        """
        Callback that is run every inference step.
        """
        ...
