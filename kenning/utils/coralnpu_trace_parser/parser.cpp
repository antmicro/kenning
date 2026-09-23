/*
 * Copyright (c) 2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parser.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/wire_format_lite.h>

#include "coralnpu_trace.pb.h"

namespace
{

using TraceEntryCallback = std::function< int(const coralnpu::sim::proto::TraceEntry &) >;

/**
 * Iterates over entries stored in a CoralNPU trace dump.
 *
 * Each entry is parsed independently and passed to the provided callback.
 * This avoids loading the entire trace into memory.
 *
 * @param filename Path to the dump file.
 * @param callback Function invoked for every parsed trace entry.
 *
 * @returns parsing error code or an error code returned by the callback.
 */
int for_each_entry(const std::string &filename, const TraceEntryCallback &callback)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input)
    {
        return -1;
    }

    google::protobuf::io::IstreamInputStream zero_copy_input(&input);
    google::protobuf::io::CodedInputStream coded_input(&zero_copy_input);

    coded_input.SetTotalBytesLimit(std::numeric_limits< int >::max());

    using WireFormatLite = google::protobuf::internal::WireFormatLite;

    while (true)
    {
        const uint32_t tag = coded_input.ReadTag();

        if (tag == 0)
        {
            break;
        }

        const int field_number = WireFormatLite::GetTagFieldNumber(tag);

        const auto wire_type = WireFormatLite::GetTagWireType(tag);

        if (field_number == coralnpu::sim::proto::TraceData::kEntryFieldNumber &&
            wire_type == WireFormatLite::WIRETYPE_LENGTH_DELIMITED)
        {
            uint32_t size;

            if (!coded_input.ReadVarint32(&size))
            {
                return -2;
            }

            const auto limit = coded_input.PushLimit(size);

            coralnpu::sim::proto::TraceEntry entry;

            if (!entry.ParseFromCodedStream(&coded_input))
            {
                coded_input.PopLimit(limit);
                return -2;
            }

            if (!coded_input.ConsumedEntireMessage())
            {
                coded_input.PopLimit(limit);
                return -2;
            }

            coded_input.PopLimit(limit);

            const int ret = callback(entry);
            if (ret != 0)
            {
                return ret;
            }
        }
        else
        {
            if (!WireFormatLite::SkipField(&coded_input, tag))
            {
                return -2;
            }
        }
    }

    return 0;
}

} // namespace

int get_dump_sizes(const std::string &filename, std::size_t &entry_count, std::size_t &disasm_size)
{
    entry_count = 0;
    disasm_size = 0;

    return for_each_entry(
        filename,
        [&](const coralnpu::sim::proto::TraceEntry &entry)
        {
            if (entry_count == std::numeric_limits< std::size_t >::max())
            {
                return -3;
            }

            const std::size_t entry_disasm_size = entry.disasm().size();

            if (entry_disasm_size > std::numeric_limits< std::size_t >::max() - disasm_size)
            {
                return -3;
            }

            ++entry_count;
            disasm_size += entry_disasm_size;

            return 0;
        });
}

int parse_dump(
    const std::string &filename,
    uint64_t *cycle,
    uint32_t *address,
    uint32_t *opcode,
    uint64_t *disasm_offsets,
    uint8_t *disasm_data,
    std::size_t entry_count,
    std::size_t disasm_size)
{
    std::size_t entry_index = 0;
    std::size_t disasm_offset = 0;

    if (disasm_offsets == nullptr)
    {
        return -3;
    }

    disasm_offsets[0] = 0;

    const int ret = for_each_entry(
        filename,
        [&](const coralnpu::sim::proto::TraceEntry &entry)
        {
            if (entry_index >= entry_count)
            {
                return -3;
            }

            const std::size_t entry_disasm_size = entry.disasm().size();

            if (entry_disasm_size > disasm_size - disasm_offset)
            {
                return -3;
            }

            cycle[entry_index] = entry.cycle();
            address[entry_index] = entry.address();
            opcode[entry_index] = entry.opcode();

            if (entry_disasm_size != 0)
            {
                if (disasm_data == nullptr)
                {
                    return -3;
                }

                std::memcpy(disasm_data + disasm_offset, entry.disasm().data(), entry_disasm_size);
            }

            disasm_offset += entry_disasm_size;

            disasm_offsets[entry_index + 1] = static_cast< uint64_t >(disasm_offset);

            ++entry_index;

            return 0;
        });

    if (ret != 0)
    {
        return ret;
    }

    if (entry_index != entry_count || disasm_offset != disasm_size)
    {
        return -3;
    }

    return 0;
}
