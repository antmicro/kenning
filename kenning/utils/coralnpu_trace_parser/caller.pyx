# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
cimport numpy as cnp

from libc.stddef cimport size_t
from libc.stdint cimport uint8_t
from libc.stdint cimport uint32_t
from libc.stdint cimport uint64_t
from libcpp.string cimport string


cnp.import_array()


cdef extern from "parser.hpp":
    int get_dump_sizes(
        const string& filename,
        size_t& entry_count,
        size_t& disasm_size
    ) nogil

    int parse_dump(
        const string& filename,
        uint64_t* cycle,
        uint32_t* address,
        uint32_t* opcode,
        uint64_t* disasm_offsets,
        uint8_t* disasm_data,
        size_t entry_count,
        size_t disasm_size
    ) nogil


cdef class PackedStrings:
    """
    Sequence of strings stored in a packed byte buffer.

    Strings are decoded lazily when accessed, avoiding allocation of
    Python string objects for all trace entries at once.
    """

    cdef object data
    cdef object offsets

    def __cinit__(self, data, offsets):
        """
        Initializes packed string storage.

        Parameters
        ----------
        data : numpy.ndarray
            Packed data.
        offsets : numpy.ndarray
            Offsets of individual strings in data.
        """
        self.data = data
        self.offsets = offsets

    def __len__(self):
        """
        Retrieves number of stored strings.

        Returns
        -------
        int
            Number of stored strings.
        """
        return len(self.offsets) - 1

    def __getitem__(self, index):
        """
        Retrieves and decodes a string.

        Parameters
        ----------
        index : int or slice
            Index or slice of strings to retrieve.

        Returns
        -------
        str or list[str]
            Decoded string or list of decoded strings.
        """
        cdef Py_ssize_t length = len(self)
        cdef Py_ssize_t start
        cdef Py_ssize_t end

        if isinstance(index, slice):
            slice_start, slice_stop, slice_step = index.indices(length)
            return [
                self[i]
                for i in range(slice_start, slice_stop, slice_step)
            ]

        if index < 0:
            index += length

        if index < 0 or index >= length:
            raise IndexError("disasm index out of range")

        start = <Py_ssize_t>self.offsets[index]
        end = <Py_ssize_t>self.offsets[index + 1]

        return self.data[start:end].tobytes().decode()

    def __iter__(self):
        """
        Iterates over decoded strings.

        Yields
        ------
        str
            Decoded string.
        """
        cdef Py_ssize_t i

        for i in range(len(self)):
            yield self[i]


def parse(filename):
    """
    Parses CoralNPU trace dump.

    Numeric trace fields are written directly into NumPy-owned buffers.
    Disassembled instructions are stored in a packed byte buffer and
    decoded lazily.

    Parameters
    ----------
    filename : str
        Path to the trace dump file.

    Returns
    -------
    dict
        Parsed trace containing cycle counters, addresses, opcodes and
        lazily decoded disassembled instructions.

    Raises
    ------
    RuntimeError
        Raised when the trace cannot be parsed.
    """
    cdef string cpp_filename = filename.encode()

    cdef size_t entry_count = 0
    cdef size_t disasm_size = 0
    cdef int ret

    with nogil:
        ret = get_dump_sizes(
            cpp_filename,
            entry_count,
            disasm_size
        )

    if ret != 0:
        raise RuntimeError(
            f"Failed to determine trace size: {ret}"
        )

    cdef cnp.ndarray[cnp.uint64_t, ndim=1] cycle = np.empty(
        entry_count,
        dtype=np.uint64
    )

    cdef cnp.ndarray[cnp.uint32_t, ndim=1] address = np.empty(
        entry_count,
        dtype=np.uint32
    )

    cdef cnp.ndarray[cnp.uint32_t, ndim=1] opcode = np.empty(
        entry_count,
        dtype=np.uint32
    )

    cdef cnp.ndarray[cnp.uint64_t, ndim=1] disasm_offsets = np.empty(
        entry_count + 1,
        dtype=np.uint64
    )

    cdef cnp.ndarray[cnp.uint8_t, ndim=1] disasm_data = np.empty(
        disasm_size,
        dtype=np.uint8
    )

    cdef uint64_t[:] cycle_view = cycle
    cdef uint32_t[:] address_view = address
    cdef uint32_t[:] opcode_view = opcode
    cdef uint64_t[:] disasm_offsets_view = disasm_offsets
    cdef uint8_t[:] disasm_data_view = disasm_data

    cdef uint64_t* cycle_ptr = NULL
    cdef uint32_t* address_ptr = NULL
    cdef uint32_t* opcode_ptr = NULL
    cdef uint64_t* disasm_offsets_ptr = NULL
    cdef uint8_t* disasm_data_ptr = NULL

    if entry_count != 0:
        cycle_ptr = &cycle_view[0]
        address_ptr = &address_view[0]
        opcode_ptr = &opcode_view[0]

    # disasm_offsets always contains entry_count + 1 elements, so it is never
    # empty
    disasm_offsets_ptr = &disasm_offsets_view[0]

    if disasm_size != 0:
        disasm_data_ptr = &disasm_data_view[0]

    with nogil:
        ret = parse_dump(
            cpp_filename,
            cycle_ptr,
            address_ptr,
            opcode_ptr,
            disasm_offsets_ptr,
            disasm_data_ptr,
            entry_count,
            disasm_size
        )

    if ret != 0:
        raise RuntimeError(f"Failed to parse trace: {ret}")

    return {
        "cycle": cycle,
        "address": address,
        "opcode": opcode,
        "disasm": PackedStrings(disasm_data, disasm_offsets),
    }
