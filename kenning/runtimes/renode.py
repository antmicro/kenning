# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for Renode
"""

from typing import Dict
from pathlib import Path
import re
from pyrenode import Pyrenode

from kenning.core.runtime import Runtime
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.measurements import MeasurementsCollector


class RenodeRuntime(Runtime):
    """
    Runtime subclass that provides and API for testing inference on bare-metal
    runtimes executed on Renode simulated platform.
    """

    arguments_structure = {
        'runtime_binary_path': {
            'argparse_name': '--runtime-binary-path',
            'description': 'Path to bare-metal runtime binary',
            'type': Path
        },
        'platform_resc_path': {
            'argparse_name': '--platform-resc-path',
            'description': 'Path to platform script',
            'type': Path
        }
    }

    def __init__(
            self,
            protocol: RuntimeProtocol,
            runtime_binary_path: Path,
            platform_resc_path: Path,
            collect_performance_data: bool = True):
        """
        Constructs Renode runtime.

        Parameters
        ----------
        protocols : RuntimeProtocol
            The implementation of the host-target communication protocol used
            to communicate with simulated platform.
        runtime_binary_path : Path
            Path to the runtime binary
        platform_resc_path : Path
            Path to the Renode script
        collect_performance_data : bool
            Disable collection and processing of performance metrics
        """
        self.runtime_binary_path = runtime_binary_path
        self.platform_resc_path = platform_resc_path
        self.renode_handler = None
        self.virtual_time_regex = re.compile(
            r'Elapsed Virtual Time: (\d{2}):(\d{2}):(\d{2}\.\d*)'
        )
        super().__init__(
            protocol,
            collect_performance_data
        )

    @classmethod
    def from_argparse(cls, protocol, args):
        return cls(
            protocol=protocol,
            runtime_binary_path=args.runtime_binary_path,
            platform_resc_path=args.platform_resc_path,
            collect_performance_data=args.disable_performance_measurements
        )

    def run_client(
            self,
            dataset: Dataset,
            modelwrapper: ModelWrapper,
            compiledmodelpath: Path):
        with Pyrenode() as renode_handler:
            self.renode_handler = renode_handler
            self.init_renode()

            pre_opcode_stats = self.get_opcode_stats()

            ret = super().run_client(dataset, modelwrapper, compiledmodelpath)

            post_opcode_stats = self.get_opcode_stats()

            MeasurementsCollector.measurements += {
                'opcode_counters': self._opcode_stats_diff(
                    pre_opcode_stats, post_opcode_stats
                )
            }

            self.renode_handler = None

        return ret

    def get_time(self):
        if self.renode_handler is None:
            return 0

        elapsed_time_str = self.renode_handler.run_robot_keyword(
            'ExecuteCommand', 'machine ElapsedVirtualTime'
        )
        match = self.virtual_time_regex.search(elapsed_time_str)
        if match is None:
            return 0

        groups = match.groups()

        h = int(groups[0])
        m = int(groups[1])
        s = float(groups[2])

        return 3600*h + 60*m + s

    def init_renode(self):
        """
        Initializes Renode process and starts runtime.
        """
        self.renode_handler.initialize()
        self.renode_handler.run_robot_keyword(
            'CreateLogTester', timeout=5.0
        )
        self.renode_handler.run_robot_keyword(
            'ExecuteCommand', f'$bin=@{self.runtime_binary_path}'
        )
        self.renode_handler.run_robot_keyword(
            'ExecuteCommand', f'i @{self.platform_resc_path}'
        )
        self.renode_handler.run_robot_keyword(
            'ExecuteCommand', 'start'
        )
        self.renode_handler.run_robot_keyword(
            'ExecuteCommand', 'sysbus.vec_controlblock WriteDoubleWord 0xc 0'
        )
        self.renode_handler.run_robot_keyword(
            'WaitForLogEntry', r'.*Runtime started.*', treatAsRegex=True
        )

    def get_opcode_stats(self) -> Dict[str, int]:
        """
        Retrieves opcode counters from Renode.

        Returns
        -------
        Dict[str, int] :
            Dict where the keys are opcodes and the values are counters
        """
        # retrieve opcode counters
        stats_raw = self.renode_handler.run_robot_keyword(
            'ExecuteCommand', 'sysbus.cpu GetAllOpcodesCounters'
        )
        # remove double newlines
        stats_raw = stats_raw.replace('\n\n', '\n')
        lines = stats_raw.split('\n')
        # skip the header
        lines = lines[3:]
        stats = {}
        for line in lines:
            line_split = line.split('|')
            if len(line_split) != 4:
                continue
            _, opcode, counter, _ = line.split('|')
            stats[opcode.strip()] = int(counter.strip())

        return stats

    @staticmethod
    def _opcode_stats_diff(
            opcode_stats_a: Dict[str, int],
            opcode_stats_b: Dict[str, int]) -> Dict[str, int]:
        """
        Computes difference of opcode counters. It is assumed that counters
        from second dict are greater.

        Parameters
        ----------
        opcode_stats_a : Dict[str, int]
            First opcode stats
        opcode_stats_b : Dict[str, int]
            Seconds opcode stats

        Returns
        -------
        Dict[str, int] :
            Difference between two opcode stats
        """
        ret = {}
        for opcode in opcode_stats_b.keys():
            ret[opcode] = (opcode_stats_b[opcode] -
                           opcode_stats_a.get(opcode, 0))
        return ret
