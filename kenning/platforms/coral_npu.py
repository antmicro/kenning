# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a wrapper for CoralNPU platform.
"""

from typing import List, Optional

from kenning.core.platform import Platform
from kenning.utils.resource_manager import ResourceURI


class CoralNPUPlatform(Platform):
    """
    Platform wrapper for CoralNPU devices.
    """

    needs_protocol = False

    arguments_structure = {
        "compilation_flags": {
            "description": "List of compilation flags",
            "type": List[str],
            "nullable": True,
            "default": None,
        },
    }

    def __init__(
        self,
        name: Optional[str] = None,
        platforms_definitions: Optional[List[ResourceURI]] = None,
        compilation_flags: Optional[List[str]] = None,
    ):
        self.compilation_flags = compilation_flags

        super().__init__(name, platforms_definitions)
