# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict, List

import pytest

import kenning.utils.coralnpu_trace_parser as trace_parser
import kenning.utils.coralnpu_trace_parser.coralnpu_trace_pb2 as trace_pb2


class TestCoralNPUTraceParser:
    def make_trace_file(
        self,
        tmp_path: Path,
        entries: list[trace_pb2.TraceEntry],
    ) -> Path:
        """
        Creates a temporary serialized trace file.

        Parameters
        ----------
        tmp_path : Path
            Directory in which the trace file is created.
        entries : list[TraceEntry]
            Trace entries to serialize.

        Returns
        -------
        Path
            Path to the serialized trace file.
        """
        trace = trace_pb2.TraceData()

        for cycle, address, opcode, disasm in entries:
            entry = trace.entry.add()
            entry.cycle = cycle
            entry.address = address
            entry.opcode = opcode
            entry.disasm = disasm

        path = tmp_path / "trace.pb"
        path.write_bytes(trace.SerializeToString())

        return path

    def parse(
        self,
        tmp_path: Path,
        entries: list[trace_pb2.TraceEntry],
    ) -> Dict[str, List]:
        """
        Creates and parses a temporary trace file.

        Parameters
        ----------
        tmp_path : Path
            Directory in which the trace file is created.
        entries : list[TraceEntry]
            Trace entries to serialize and parse.

        Returns
        -------
        ParsedTrace
            Parsed trace data.
        """
        path = self.make_trace_file(tmp_path, entries)
        return trace_parser.parse(str(path))

    def assert_entries(
        self,
        result: Dict[str, list],
        expected: list[trace_pb2.TraceEntry],
    ):
        """
        Verifies parsed trace entries.

        Parameters
        ----------
        result : ParsedTrace
            Parsed trace data.
        expected : list[TraceEntry]
            Expected trace entries.
        """
        assert len(result["cycle"]) == len(expected)
        assert len(result["address"]) == len(expected)
        assert len(result["opcode"]) == len(expected)
        assert len(result["disasm"]) == len(expected)

        for i, (cycle, address, opcode, disasm) in enumerate(expected):
            assert result["cycle"][i] == cycle
            assert result["address"][i] == address
            assert result["opcode"][i] == opcode
            assert result["disasm"][i] == disasm

    def test_empty_trace(self, tmp_path):
        """
        Test parsing an empty trace.
        """
        result = self.parse(tmp_path, [])

        self.assert_entries(result, [])

    def test_single_entry(self, tmp_path):
        """
        Test parsing a trace with a single entry.
        """
        entries = [
            (0, 0x1000, 0x00000013, "nop"),
        ]

        result = self.parse(tmp_path, entries)

        self.assert_entries(result, entries)

    def test_multiple_entries(self, tmp_path):
        """
        Test parsing a trace with multiple entries.
        """
        entries = [
            (0, 0x1000, 0x00000013, "nop"),
            (1, 0x1004, 0x00100093, "li ra, 1"),
            (2, 0x1008, 0x00008067, "ret"),
        ]

        result = self.parse(tmp_path, entries)

        self.assert_entries(result, entries)

    @pytest.mark.parametrize(
        "cycle,address,opcode,disasm",
        [
            (0, 0, 0, ""),
            (1, 1, 1, "x"),
            (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, "max"),
            (0x8000000000000000, 0x80000000, 0x80000000, "high bit set"),
        ],
    )
    def test_values(self, tmp_path, cycle, address, opcode, disasm):
        """
        Test parsing boundary and representative field values.
        """
        entries = [
            (cycle, address, opcode, disasm),
        ]

        result = self.parse(tmp_path, entries)

        self.assert_entries(result, entries)

    def test_preserves_entry_order(self, tmp_path):
        """
        Test that trace entry order is preserved.
        """
        entries = [
            (300, 0x3000, 3, "third"),
            (100, 0x1000, 1, "first"),
            (200, 0x2000, 2, "second"),
        ]

        result = self.parse(tmp_path, entries)

        self.assert_entries(result, entries)

    def test_many_entries(self, tmp_path):
        """
        Test parsing a trace with many entries.
        """
        entries = [
            (
                i,
                0x1000 + i * 4,
                i,
                f"instruction_{i}",
            )
            for i in range(1000)
        ]

        result = self.parse(tmp_path, entries)

        self.assert_entries(result, entries)
