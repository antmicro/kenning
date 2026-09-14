#!/usr/bin/env python

# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that runs probe server.
The server is supposed to monitor machine resources and send them when
requested.

It requires implementation of Protocol of several classes as input.
This Protocol will be used for sending back resource usage statistics.
"""

import argparse
import json
import signal
import sys
from threading import Event
from typing import Any, Dict, List, Optional, Tuple

from kenning.cli.command_template import (
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)
from kenning.cli.completers import (
    RUNTIME_PROTOCOLS,
    ClassPathCompleter,
)
from kenning.core.exceptions import (
    ConfigurationError,
)
from kenning.core.measurements import Measurements, SystemStatsCollector
from kenning.core.protocol import Protocol, ServerAction, ServerStatus
from kenning.utils.class_loader import (
    ConfigKey,
)
from kenning.utils.logger import KLogger

FLAG_CONFIG = "Probe configuration with flags"
ARGS_GROUPS = {
    FLAG_CONFIG: (
        "Configuration with flags. Arguments with '*' are required."
    ),
}


class ProbeServer(object):
    """
    A class for running probe that monitors resource usage.

    It is used to delegate evaluation of the model to a different
    device if the model is optimized for a different device than
    host.
    """

    def __init__(
        self, protocol: Protocol, prefix: str, pid: int, step: float = 0.1
    ):
        self.protocol = protocol
        self.close_server_event = Event()
        self.status = ServerStatus(ServerAction.IDLE)
        self.collector_args = {"prefix": prefix, "step": step, "pid": pid}
        self.collector = SystemStatsCollector(**self.collector_args)
        self.num_clients = 0

    def close(self):
        """
        Stops the probe.
        """
        self.close_server_event.set()

    def run(self):
        """
        Main runtime server program.

        It waits for requests from a single client. When client connects,
        starts collecting statistics.
        """

        def client_disconnected_callback():
            KLogger.info("Client disconnected.")
            self.status = ServerStatus(ServerAction.WAITING_FOR_CLIENT)
            self.num_clients -= 1
            if self.num_clients == 0:
                self.collector.stop()
                self.collector.join()
                self.collector = SystemStatsCollector(**self.collector_args)

        def client_connected_callback():
            KLogger.info("Client connected.")
            self.num_clients += 1
            if self.num_clients == 1:
                self.status = ServerStatus(ServerAction.CLIENT_CONNECTED)
                self.collector.start()

        self.status = ServerStatus(
            ServerAction.WAITING_FOR_CLIENT,
            self.protocol.initialize_server(
                client_connected_callback, client_disconnected_callback
            ),
        )

        if not self.status.success:
            KLogger.error("Probe prepare failed")
            return

        KLogger.info("Probe started")

        self.protocol.serve(
            download_stats_callback=self._stats_callback,
        )
        self.close_server_event.wait()
        self.status = ServerStatus(ServerAction.IDLE)
        self.protocol.disconnect()

    def _stats_callback(self) -> Tuple[ServerStatus, Optional[bytes]]:
        """
        Server callback for stopping measurements and retrieving stats.

        Returns
        -------
        Tuple[ServerStatus, Optional[bytes]]
            Object containing information about the last task executed by the
            server and whether that task succeeded or not.
            Inference statistics, or None in case of failure.
        """
        self.status.start_action(ServerAction.COMPUTING_STATISTICS)
        KLogger.info("Sending stats.")

        stats = json.dumps(self.collector.measurements.data)
        self.collector.measurements = Measurements()
        out = stats.encode("utf-8")

        self.status.finish_action(True)
        return self.status, out


class ProbeServerRunner(CommandTemplate):
    """
    Command template for running the probe server.
    """

    parse_all = False
    description = __doc__.split("\n\n")[0]
    ID = generate_command_type()

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            ProbeServerRunner, ProbeServerRunner
        ).configure_parser(parser, command, types, groups)

        groups = CommandTemplate.add_groups(parser, groups, ARGS_GROUPS)

        groups[FLAG_CONFIG].add_argument(
            "--protocol-cls",
            required=True,
            help=(
                "* Protocol-based class with the implementation of "
                "communication between server and host"
            ),
        ).completer = ClassPathCompleter(RUNTIME_PROTOCOLS)
        groups[FLAG_CONFIG].add_argument(
            "--pid",
            required=True,
            type=int,
            help=("* PID of the monitored process"),
        )
        groups[FLAG_CONFIG].add_argument(
            "--prefix",
            type=str,
            default="session_utilization",
            help=("* Prefix added to measurements"),
        )
        groups[FLAG_CONFIG].add_argument(
            "--step",
            required=False,
            type=float,
            default=0.1,
            help=("* Interval of sampling measurements"),
        )

        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        keys = [ConfigKey.protocol]

        args.json_cfg = None
        objs = ProbeServerRunner.initialize_blocks(args, not_parsed, keys)
        config = ProbeServerRunner.parse_configuration(args, not_parsed, keys)
        return ProbeServerRunner._run_server(
            objs,
            config.get("prefix"),
            config.get("pid"),
            config.get("step"),
        )

    @staticmethod
    def _run_server(
        objs: Dict[ConfigKey, Any], prefix: str, pid: int, step: float = 0.1
    ):
        protocol = objs.get(ConfigKey.protocol)

        if not protocol:
            raise ConfigurationError("No protocol set for the probe")

        formersighandler = signal.getsignal(signal.SIGINT)

        server = ProbeServer(protocol, prefix, pid, step)

        def sigint_handler(sig, frame):
            server.close()
            KLogger.info(
                "Closing application (press Ctrl-C again for force closing)..."
            )
            signal.signal(signal.SIGINT, formersighandler)

        signal.signal(signal.SIGINT, sigint_handler)

        KLogger.info("Starting probe...")
        server.run()

    @staticmethod
    def get_overridable(subcommands: List[str]) -> List[ConfigKey]:
        return [
            ConfigKey.protocol,
        ]


if __name__ == "__main__":
    sys.exit(ProbeServerRunner.scenario_run())
