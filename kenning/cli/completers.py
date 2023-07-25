# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing completer for class paths.
"""

import argparse
from typing import Dict, Optional
from argcomplete.completers import BaseCompleter

from kenning.scenarios.list_classes import list_classes
from kenning.utils.class_loader import (
    OPTIMIZERS,
    RUNNERS,
    DATA_PROVIDERS,
    DATASETS,
    MODEL_WRAPPERS,
    ONNX_CONVERSIONS,
    OUTPUT_COLLECTORS,
    RUNTIME_PROTOCOLS,
    RUNTIMES,
)

ALL_TYPES = (
    OPTIMIZERS,
    RUNNERS,
    DATA_PROVIDERS,
    DATASETS,
    MODEL_WRAPPERS,
    ONNX_CONVERSIONS,
    OUTPUT_COLLECTORS,
    RUNTIME_PROTOCOLS,
    RUNTIMES,
)


class ClassPathCompleter(BaseCompleter):
    def __init__(self, class_type: Optional[str] = None):
        assert class_type is None or class_type in ALL_TYPES
        self.class_type = class_type

    def __call__(
        self, *,
        prefix: str,
        action: argparse.Action,
        parser: argparse.ArgumentParser,
        parsed_args: argparse.Namespace
    ) -> Dict[str, str]:
        self.class_type
        paths = list_classes(
            [self.class_type] if self.class_type else ALL_TYPES,
            'autocomplete', prefix
        )
        return dict(paths)
