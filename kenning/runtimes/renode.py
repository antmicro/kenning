# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for Renode
"""

from typing import Dict, Any, Tuple, BinaryIO
from pathlib import Path
from collections import defaultdict
import tempfile
import struct
import re
from pyrenode import Pyrenode

from kenning.core.runtime import Runtime
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.measurements import MeasurementsCollector
from kenning.utils.logger import get_logger


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
        self.profiler_dump_path = None
        self.log = get_logger()
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

        MeasurementsCollector.measurements += self.get_profiler_stats()

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
        self.profiler_dump_path = Path(tempfile.mktemp(
            prefix='renode_profiler_', suffix='.dump'
        ))
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
        if self.collect_performance_data:
            self.renode_handler.run_robot_keyword(
                'ExecuteCommand',
                f'machine EnableProfiler @{self.profiler_dump_path}'
            )
        self.renode_handler.run_robot_keyword(
            'ExecuteCommand', 'sysbus.vec_controlblock WriteDoubleWord 0xc 0'
        )
        self.renode_handler.run_robot_keyword(
            'WaitForLogEntry', r'.*Runtime started.*', treatAsRegex=True
        )
        self.log.info(f'Profiler dump path: {self.profiler_dump_path}')

    def get_opcode_stats(self) -> Dict[str, int]:
        """
        Retrieves opcode counters from Renode.

        Returns
        -------
        Dict[str, int] :
            Dict where the keys are opcodes and the values are counters
        """
        self.log.info('Retrieving opcode counters')

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

    def get_profiler_stats(self) -> Dict[str, Any]:
        """
        Parses Renode profiler dump.

        Returns
        -------
        Dict[str, List[float]] :
            Stats retrieved from Renode profiler dump
        """
        self.log.info('Parsing Renode profiler dump')
        if self.profiler_dump_path is None:
            self.log.error('Missing profiler dump file')
            raise FileNotFoundError

        parser = _ProfilerDumpParser(self.profiler_dump_path)

        return parser.parse()

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


class _ProfilerDumpParser(object):
    ENTRY_TYPE_INSTRUCTIONS = b'\x00'
    ENTRY_TYPE_MEM0RY = b'\x01'
    ENTRY_TYPE_PERIPHERALS = b'\x02'
    ENTRY_TYPE_EXCEPTIONS = b'\x03'

    ENTRY_HEADER_FORMAT = '<qdc'
    ENTRY_FORMAT_INSTRUCTIONS = '<cQ'
    ENTRY_FORMAT_MEM0RY = 'c'
    ENTRY_FORMAT_PERIPHERALS = '<cQ'
    ENTRY_FORMAT_EXCEPTIONS = 'Q'

    MEMORY_OPERATION_READ = b'\x02'
    MEMORY_OPERATION_WRITE = b'\x03'

    PERIPHERAL_OPERATION_READ = b'\x00'
    PERIPHERAL_OPERATION_WRITE = b'\x01'

    def __init__(self, dump_path: Path):
        self.dump_path = dump_path

    def parse(self) -> Dict[str, Any]:
        """
        Parses Renode profiler dump

        Returns
        -------
        Dict[str, Any] :
            Dict containing statistics retrieved from the dump file
        """
        profiler_timestamps = []
        stats = {
            'executed_instructions': {},
            'memory_accesses': {'read': [], 'write': []},
            'peripheral_accesses': {},
            'exceptions': []
        }

        with open('/tmp/profiler.dump', 'rb') as f:
            # parse header
            cpus, peripherals = self._parse_header(f)

            for cpu in cpus.values():
                stats['executed_instructions'][cpu] = []

            for peripheral in peripherals.keys():
                stats['peripheral_accesses'][peripheral] = {
                    'read': [], 'write': []
                }

            startTime = 0
            entry = struct.Struct(self.ENTRY_HEADER_FORMAT)
            interval_step = 10  # [ms]
            prev_instr_counter = defaultdict(lambda: 0)

            # parse entries
            while True:
                entry_header = f.read(entry.size)
                if not entry_header:
                    break
                real_time, virtual_time, entry_type = \
                    entry.unpack(entry_header)
                if startTime == 0:
                    startTime = real_time
                real_time = (real_time - startTime) / 10000

                interval_start = virtual_time - virtual_time % interval_step
                interval_start /= 1000.
                if (len(profiler_timestamps) == 0 or
                        profiler_timestamps[-1] != interval_start):
                    # new interval - need to add its start timestamp and append
                    # new counters to each stats list
                    profiler_timestamps.append(interval_start)
                    stats_to_update = [stats]
                    while len(stats_to_update):
                        s = stats_to_update.pop(0)
                        if isinstance(s, list):
                            s.append(0)
                        elif isinstance(s, dict):
                            stats_to_update.extend(s.values())

                if entry_type == self.ENTRY_TYPE_INSTRUCTIONS:
                    # parse executed instruction entry
                    output_list = stats['executed_instructions']
                    cpu_id, instr_counter = self._read(
                        self.ENTRY_FORMAT_INSTRUCTIONS,
                        f
                    )

                    cpu = cpus[cpu_id[0]]
                    output_list = output_list[cpu]

                    output_list[-1] += instr_counter - prev_instr_counter[cpu]
                    prev_instr_counter[cpu] = instr_counter

                elif entry_type == self.ENTRY_TYPE_MEM0RY:
                    # parse memory access entry
                    output_list = stats['memory_accesses']
                    operation = self._read(self.ENTRY_FORMAT_MEM0RY, f)[0]

                    if operation == self.MEMORY_OPERATION_READ:
                        # read
                        output_list = output_list['read']
                    elif operation == self.MEMORY_OPERATION_WRITE:
                        # write
                        output_list = output_list['write']
                    else:
                        # invalid operation
                        continue

                    output_list[-1] += 1

                elif entry_type == self.ENTRY_TYPE_PERIPHERALS:
                    # parse peripheral access entry
                    output_list = stats['peripheral_accesses']
                    operation, address = self._read(
                        self.ENTRY_FORMAT_PERIPHERALS,
                        f
                    )

                    peripheral_found = False
                    for peripheral, address_range in peripherals.items():
                        if address_range[0] <= address <= address_range[1]:
                            output_list = output_list[peripheral]
                            peripheral_found = True
                            break

                    if not peripheral_found:
                        continue

                    if operation == self.PERIPHERAL_OPERATION_READ:
                        # read
                        output_list = output_list['read']
                    elif operation == self.PERIPHERAL_OPERATION_WRITE:
                        # write
                        output_list = output_list['write']
                    else:
                        # invalid operation
                        continue

                    output_list[-1] += 1

                elif entry_type == self.ENTRY_TYPE_EXCEPTIONS:
                    # parse exception entry
                    output_list = stats['exceptions']
                    _ = self._read(self.ENTRY_FORMAT_EXCEPTIONS, f)

                    output_list[-1] += 1

                else:
                    raise Exception(
                        f'Invalid entry in profiler dump: {entry_type}'
                    )

        return dict(stats, profiler_timestamps=profiler_timestamps)

    def _parse_header(
            self,
            file: BinaryIO
            ) -> Tuple[Dict[int, str], Dict[str, Tuple[int, int]]]:
        """
        Parses header of Renode profiler dump

        Parameters
        ----------
        file : BinaryIO
            File-like object

        Returns
        -------
        Tuple[Dict[int, str], Dict[str, List[int]]] :
            Tuples of dicts containing cpus and peripherals data
        """
        cpus = {}
        peripherals = {}
        cpus_count = self._read('i', file)[0]
        for _ in range(cpus_count):
            cpu_id = self._read('i', file)[0]
            cpu_name_len = self._read('i', file)[0]
            cpus[cpu_id] = self._read(f'{cpu_name_len}s', file)[0].decode()

        peripherals_count = self._read('i', file)[0]
        for _ in range(peripherals_count):
            peripheral_name_len = self._read('i', file)[0]
            peripheral_name = self._read(
                f'{peripheral_name_len}s', file
            )[0].decode()
            peripheral_start_address, peripheral_end_address = self._read(
                '2Q', file
            )
            peripherals[peripheral_name] = (
                peripheral_start_address,
                peripheral_end_address
            )

        return cpus, peripherals

    @staticmethod
    def _read(format_str: str, file: BinaryIO) -> Tuple[Any, ...]:
        """
        Reads struct of given format from file

        Parameters
        ----------
        format_str : str
            Format of the struct
        file : BinaryIO
            File-like object

        Returns
        -------
        Tuple[Any, ...] :
            Struct read from file
        """
        return struct.unpack(
            format_str,
            file.read(struct.calcsize(format_str))
        )
