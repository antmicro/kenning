# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Implements the ModuleRunner CLI for dynamically executing Kenning blocks.
"""

import argparse
import ast
import inspect
import json
import threading
import typing
from pathlib import Path
from typing import Any, List, Optional, Tuple

from kenning.cli.command_template import (
    GROUP_SCHEMA,
    RUN_MODULE,
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)
from kenning.core.exceptions import (
    KenningError,
)
from kenning.utils.class_loader import (
    load_class,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import ResourceURI


def inject_uninitialized_attrs(instance: Any) -> None:
    """
    Parses class source code and find all self attributes and sets
    those that are not initialized by the constructor to None.

    Parameters
    ----------
    instance : Any
        The instance to which uninitialized attributes will be added.
    """
    cls = instance.__class__
    attributes = set()
    source = inspect.getsource(cls)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            attributes.add(node.attr)

    for attr in attributes:
        # Initialized by init
        if attr in instance.__dict__:
            continue
        if hasattr(cls, attr):
            continue
        setattr(instance, attr, None)


class ModuleRunner(CommandTemplate):
    """
    Command-line interface command for instantiating
    and isolating a Kenning module.
    """

    parse_all = False
    description = "Run a standalone kenning module"
    ID = generate_command_type()

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            ModuleRunner,
            ModuleRunner,
        ).configure_parser(parser, command, types, groups)

        command_group = parser.add_argument_group(
            GROUP_SCHEMA.format(RUN_MODULE)
        )

        command_group.add_argument(
            "--module",
            help="Kenning module to run. Example: kenning.optimizers.onnx.OnnxOptimizer",  # noqa: E501
            type=str,
            required=True,
        )

        command_group.add_argument(
            "--cfg",
            help="JSON serialized config dictionary",
            type=Path,
        )

        command_group.add_argument(
            "--no-wait",
            help="Don't wait for any events. Immediately exit the program on object instantiation",  # noqa: E501
            action="store_true",
        )

        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        module_cls = load_class(args.module)
        if not getattr(args, "cfg", None):
            raise KenningError(
                "Missing --cfg argument with JSON serialized config dict"
            )

        config = json.loads(str(args.cfg))

        init_sig = inspect.signature(module_cls.__init__)

        exclusion_list = (
            int,
            float,
            str,
            bool,
            list,
            dict,
            set,
            tuple,
            Path,
            type(None),
            ResourceURI,
        )
        injected_kwargs = {}

        for name, param in init_sig.parameters.items():
            if name in ["self", "args", "kwargs"]:
                continue
            annotation = param.annotation
            requires_injection = False

            if annotation is inspect.Parameter.empty:
                if name in {
                    "dataset",
                    "modelwrapper",
                    "model_wrapper",
                    "optimizer",
                    "platform",
                    "runtime",
                }:
                    requires_injection = True
            else:
                origin = typing.get_origin(annotation)
                types_to_check = (
                    typing.get_args(annotation) if origin else [annotation]
                )
                for t in types_to_check:
                    if t is inspect.Parameter.empty:
                        continue

                    if inspect.isclass(t) and not issubclass(
                        t, exclusion_list
                    ):
                        requires_injection = True
                        break

            if requires_injection:
                KLogger.debug(f"Adding injected positional argument {name}")
                injected_kwargs[name] = None

        instantiated_obj = module_cls.build_from_config(  # noqa: F841 # TODO: serve the kenning block.
            config, **injected_kwargs, **kwargs
        )

        if not args.no_wait:
            try:
                KLogger.info(
                    "Waiting for inter-module communication... "
                    "Press Ctrl+C to quit"
                )
                threading.Event().wait()
            except KeyboardInterrupt:
                KLogger.info("Ctrl+C detected. Shutting down")
