# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "parser.hpp":
    int parse_dump(
        string filename,
        double start_timestamp,
        double end_timestamp,
        double interval_step,
        vector[double] profiler_timestamps,
        map[string, vector[int]] executed_instructions,
        map[char, vector[int]] memory_accesses,
        map[string, map[char, vector[int]]] peripheral_accesses,
        vector[int] exceptions
    )

ctypedef enum MemoryOperationType:
    MEM_OP_READ = 0x2,
    MEM_OP_WRITE = 0x3,

ctypedef enum PeripheralOperationType:
    PERIPH_OP_READ = 0x0,
    PERIPH_OP_WRITE = 0x1,

def parse(filename, start_timestamp, end_timestamp, interval_step):
    cdef vector[double] profiler_timestamps
    cdef map[string, vector[int]] executed_instructions
    cdef map[char, vector[int]] memory_accesses
    cdef map[string, map[char, vector[int]]] peripheral_accesses
    cdef vector[int] exceptions

    parse_dump(
        filename,
        start_timestamp,
        end_timestamp,
        interval_step,
        profiler_timestamps,
        executed_instructions,
        memory_accesses,
        peripheral_accesses,
        exceptions
    )

    return {
        "profiler_timestamps": profiler_timestamps,
        "executed_instructions": {
            cpu.decode(): instr for cpu, instr in executed_instructions
        },
        "memory_accesses": {
            "read": memory_accesses[MEM_OP_READ],
            "write": memory_accesses[MEM_OP_WRITE],
        },
        "peripheral_accesses": {
            periph.decode(): {
                "read": accesses[PERIPH_OP_READ],
                "write": accesses[PERIPH_OP_WRITE],
            } for periph, accesses in peripheral_accesses
        },
        "exceptions": exceptions,
    }
