#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that runs inference server.

It requires implementations of several classes as input:

* RuntimeProtocol - provides routines for communicating with the client
* Runtime - provides implementation of inference runtime

Each of those classes require specific set or arguments to configure the
compilation and benchmark process.
"""

import sys
import argparse
import signal
import json
from typing import Optional, List, Tuple
from argcomplete.completers import FilesCompleter

from kenning.cli.command_template import (
    ArgumentsGroups, CommandTemplate, ParserHelpException
)
from kenning.cli.completers import (
    ClassPathCompleter, RUNTIMES, RUNTIME_PROTOCOLS
)
from kenning.utils.class_loader import load_class, get_command
import kenning.utils.logger as logger


JSON_CONFIG = "Server configuration with JSON"
FLAG_CONFIG = "Server configuration with flags"
ARGS_GROUPS = {
    JSON_CONFIG: f"Configuration with data defined in JSON file. This section is not compatible with '{FLAG_CONFIG}'. Arguments with '*' are required.",  # noqa: E501
    FLAG_CONFIG: f"Configuration with flags. This section is not compatible with '{JSON_CONFIG}'. Arguments with '*' are required.",  # noqa: E501
}


class InferenceServer(CommandTemplate):
    parse_all = False
    description = __doc__.split('\n\n')[0]

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            InferenceServer, InferenceServer
        ).configure_parser(parser, command, types, groups)

        groups = CommandTemplate.add_groups(parser, groups, ARGS_GROUPS)

        groups[JSON_CONFIG].add_argument(
            '--json-cfg',
            help='* The path to the input JSON file with configuration'
        ).completer = FilesCompleter("*.json")
        groups[FLAG_CONFIG].add_argument(
            '--protocol-cls',
            help='* RuntimeProtocol-based class with the implementation of communication between inference tester and inference runner',  # noqa: E501
        ).completer = ClassPathCompleter(RUNTIME_PROTOCOLS)
        groups[FLAG_CONFIG].add_argument(
            '--runtime-cls',
            help='* Runtime-based class with the implementation of model runtime'  # noqa: E501
        ).completer = ClassPathCompleter(RUNTIMES)

        return parser, groups

    @staticmethod
    def run(
        args: argparse.Namespace,
        not_parsed: List[str] = [],
        **kwargs
    ):
        logger.set_verbosity(args.verbosity)
        log = logger.get_logger()

        flag_config_names = ('runtime_cls', 'protocol_cls')
        flag_config_not_none = [getattr(args, name, None) is not None
                                for name in flag_config_names]
        if args.json_cfg is None and not any(flag_config_not_none):
            raise argparse.ArgumentError(
                None, "JSON or flag configuration is required."
            )
        if args.json_cfg is not None and any(flag_config_not_none):
            raise argparse.ArgumentError(
                None, "JSON and flag configurations are mutually exclusive. "
                "Please use only one method of configuration.")

        if args.json_cfg is not None:
            InferenceServer._run_from_json(args, log, not_parsed)

        else:
            missing_args = [
                f"'{n}'" for i, n in enumerate(flag_config_names)
                if not flag_config_not_none[i]
            ]
            if missing_args and not args.help:
                raise argparse.ArgumentError(
                    None,
                    'the following arguments are required: '
                    f'{", ".join(missing_args)}'
                )

            InferenceServer._run_from_flags(args, log, not_parsed)

    @staticmethod
    def _run_from_flags(
        args: argparse.Namespace,
        log,
        not_parsed: List[str] = [],
        **kwargs
    ):
        protocolcls = load_class(args.protocol_cls) \
            if args.protocol_cls else None
        runtimecls = load_class(args.runtime_cls) \
            if args.runtime_cls else None

        parser = argparse.ArgumentParser(
            ' '.join(map(lambda x: x.strip(), get_command(with_slash=False))),
            parents=[]
            + ([protocolcls.form_argparse()[0]] if protocolcls else [])
            + ([runtimecls.form_argparse()[0]] if runtimecls else []),
            add_help=False,
        )

        if args.help:
            raise ParserHelpException(parser)

        args = parser.parse_args(not_parsed)

        protocol = protocolcls.from_argparse(args)
        runtime = runtimecls.from_argparse(protocol, args)

        InferenceServer._run_server(runtime)

    @staticmethod
    def _run_from_json(
        args: argparse.Namespace,
        log,
        not_parsed: List[str] = [],
        **kwargs
    ):
        if not_parsed:
            raise argparse.ArgumentError(
                None,
                f"unrecognized arguments: {' '.join(not_parsed)}"
            )

        with open(args.json_cfg, 'r') as f:
            json_cfg = json.load(f)

        protocolcfg = json_cfg['runtime_protocol']
        runtimecfg = json_cfg['runtime']

        protocolcls = load_class(protocolcfg['type'])
        runtimecls = load_class(runtimecfg['type'])

        protocol = protocolcls.from_json(protocolcfg['parameters'])
        runtime = runtimecls.from_json(protocol, runtimecfg['parameters'])

        InferenceServer._run_server(runtime)

    @staticmethod
    def _run_server(runtime):
        formersighandler = signal.getsignal(signal.SIGINT)

        def sigint_handler(sig, frame):
            runtime.close_server()
            runtime.protocol.log.info('Closing application (press Ctrl-C again for force closing)...')  # noqa: E501
            signal.signal(signal.SIGINT, formersighandler)

        signal.signal(signal.SIGINT, sigint_handler)

        runtime.protocol.log.info('Starting server...')
        runtime.run_server()


if __name__ == '__main__':
    sys.exit(InferenceServer.scenario_run())
