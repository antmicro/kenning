# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a wrapper for platforms.
"""

import sys
from abc import ABC
from time import perf_counter
from typing import List, Optional

import yaml

from kenning.core.measurements import Measurements
from kenning.core.protocol import Protocol
from kenning.resources import platforms
from kenning.utils.args_manager import ArgumentsHandler, convert
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import ResourceURI

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
        "platforms_definitions": {
            "description": "Files with platform definitions from the least to the most significant",  # noqa: E501
            "type": list[ResourceURI],
            "nullable": True,
            "default": None,
            "overridable": True,
        },
    }

    # default values for data that is read from platforms.yml
    platform_defaults = {}

    def __init__(
        self,
        name: Optional[str] = None,
        platforms_definitions: Optional[List[ResourceURI]] = None,
    ):
        """
        Constructs platform and reads its data from platforms.yaml.

        Parameters
        ----------
        name : Optional[str]
            Name of the platform.
        platforms_definitions : Optional[List[ResourceURI]]
            Files with platform definitions
            from the least to the most significant.
        """
        self.name = name
        self.platforms_definitions = platforms_definitions
        if self.platforms_definitions is None:
            with path(platforms, "platforms.yml") as platforms_path:
                self.platforms_definitions = [platforms_path]
        self.read_data_from_platforms_yaml()

    def init(self):
        """
        Initializes the platform.
        """
        ...

    def read_data_from_platforms_yaml(self):
        """
        Retrieves platform data from specified platform definition files.
        """
        data, sources = [], []
        for platform_def in self.platforms_definitions:
            with platform_def.open("r") as platforms_yaml:
                data.append(yaml.safe_load(platforms_yaml).get(self.name, {}))
                sources.append(platform_def)

        if any(not platform for platform in data):
            KLogger.warning(
                f"Platform {self.name} not found in defined platforms"
            )

        default_attrs = self.__class__.platform_defaults
        all_attrs = set()
        for platform in data + [default_attrs]:
            all_attrs = all_attrs.union(platform.keys())

        attr_srcs = tuple(reversed(list(zip(data, map(str, sources))))) + (
            (default_attrs, "defaults"),
        )
        param_schema = self.form_parameterschema()
        for attr_name in all_attrs:
            if getattr(self, attr_name, None) is not None:
                KLogger.debug(f"Skipping {attr_name} attribute")
                continue

            for attr_src, attr_src_name in attr_srcs:
                if attr_src.get(attr_name, None) is None:
                    continue
                attr_value = attr_src[attr_name]
                arg_struct = param_schema["properties"].get(attr_name, None)
                if arg_struct is not None:
                    attr_type = arg_struct["convert-type"]
                    if not isinstance(attr_type, (list, tuple)):
                        attr_type = [attr_type]
                    if "array" in arg_struct.get("type", []):
                        attr_value = [
                            convert(attr_type, v) for v in attr_value
                        ]
                    else:
                        attr_value = convert(attr_type, attr_value)
                setattr(self, attr_name, attr_value)
                KLogger.debug(
                    f"Setting {attr_name}={attr_value} from {attr_src_name}"
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
