# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing converters for model formats and optimizers.
"""

import os

from kenning.core.converter import ModelConverter
from kenning.utils.class_loader import get_all_subclasses

from .converter_registry import ConverterRegistry

# needed for python 3.12 compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"

converters = get_all_subclasses("kenning.converters", ModelConverter)
converter_registry = ConverterRegistry()

for converter in converters:
    converter_registry.register(converter)

converter_registry.add_alias("executorch", "torch")
converter_registry.add_alias("tinygrad", "onnx")
converter_registry.add_alias("iree", "onnx")
