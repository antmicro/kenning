# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a wrapper for NVIDIA CUDA platform.
"""

from typing import List, Optional

from kenning.core.platform import Platform
from kenning.utils.resource_manager import ResourceURI


class CUDAPlatform(Platform):
    """
    Platform wrapper for CUDA devices.
    """

    needs_protocol = False

    arguments_structure = {
        "target": {
            "description": "Target architecture of the device",
            "type": str,
            "nullable": False,
            "default": "native",
        },
        "target_host": {
            "description": "Target architecture of the host",
            "type": str,
            "nullable": False,
            "default": "native",
        },
        "compilation_flags": {
            "description": "List of compilation flags",
            "type": List[str],
            "nullable": True,
            "default": None,
        },
    }

    platform_defaults = dict(
        Platform.platform_defaults, compute_capability="sm_60"
    )

    def __init__(
        self,
        name: Optional[str] = None,
        platforms_definitions: Optional[List[ResourceURI]] = None,
        compilation_flags: Optional[List[str]] = None,
        target: str = "",
        target_host: str = "",
    ):
        self.compilation_flags = (
            compilation_flags if compilation_flags is not None else []
        )
        self.target = target
        self.target_host = target_host

        super().__init__(name, platforms_definitions)
