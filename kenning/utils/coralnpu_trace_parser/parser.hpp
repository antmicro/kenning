/*
 * Copyright (c) 2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <string>

/**
 * Retrieves the number of entries and total size of disassembly strings stored in a CoralNPU trace dump.
 *
 * @param filename    Path to the dump file.
 * @param entry_count Number of trace entries in the dump.
 * @param disasm_size Total size, in bytes, of all disassembly strings.
 *
 * @returns parsing error code.
 */
int get_dump_sizes(const std::string &filename, std::size_t &entry_count, std::size_t &disasm_size);

/**
 * Parses CoralNPU trace dump directly into provided output buffers.
 *
 * @param filename       Path to the dump file.
 * @param cycle          Cycle counters of the instructions.
 * @param address        Addresses of the instructions.
 * @param opcode         Opcodes of the instructions.
 * @param disasm_offsets Offsets of individual disassembly strings in @p disasm_data. The buffer must contain @p
 *                       entry_count + 1 elements.
 * @param disasm_data    Buffer containing concatenated disassembly strings.
 * @param entry_count    Number of entries expected in the dump and number of elements available in the cycle, address
 *                       and opcode buffers.
 * @param disasm_size    Size, in bytes, of the disasm_data buffer.
 *
 * @returns parsing error code.
 */
int parse_dump(
    const std::string &filename,
    uint64_t *cycle,
    uint32_t *address,
    uint32_t *opcode,
    uint64_t *disasm_offsets,
    uint8_t *disasm_data,
    std::size_t entry_count,
    std::size_t disasm_size);
