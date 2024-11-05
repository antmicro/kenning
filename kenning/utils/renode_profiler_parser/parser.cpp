/*
 * Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#include "parser.hpp"

/**
 * Enum with dump parser entries types.
 */
enum EntryType : char
{
    ENTRY_TYPE_INSTRUCTION = 0x0,
    ENTRY_TYPE_MEMORY = 0x1,
    ENTRY_TYPE_PERIPHERALS = 0x2,
    ENTRY_TYPE_EXCEPTIONS = 0x3,
};

/**
 * Enum with memory operations types.
 */
enum MemoryOperationType : char
{
    MEM_OP_READ = 0x2,
    MEM_OP_WRITE = 0x3,
};

/**
 * Enum with peripheral operations types.
 */
enum PeripheralOperationType : char
{
    PERIPH_OP_READ = 0x0,
    PERIPH_OP_WRITE = 0x1,
};

/**
 * Profiler dump entry header.
 */
struct __attribute__((packed)) EntryHeader
{
    long long real_time;
    double virt_time;
    EntryType entry_type;
};

/**
 * Profiler dump entry body for instruction type entry.
 */
struct __attribute__((packed)) EntryInstruction
{
    int cpu_id;
    unsigned long long instr_counter;
};

/**
 * Profiler dump entry body for memory type entry.
 */
struct __attribute__((packed)) EntryMemory
{
    MemoryOperationType op;
};

/**
 * Profiler dump entry body for peripheral type entry.
 */
struct __attribute__((packed)) EntryPeripherals
{
    PeripheralOperationType op;
    unsigned long long addr;
};

/**
 * Profiler dump entry body for exception type entry.
 */
struct __attribute__((packed)) EntryExceptions
{
    unsigned long long index;
};

/**
 * Prints progress in percentage and parsed vs total bytes.
 *
 * @param done Number of parsed bytes
 * @param total Number of total bytes
 */
void print_progress(std::uintmax_t done, std::uintmax_t total)
{
    static std::uintmax_t perc = 0;

    std::uintmax_t new_perc = (100 * done) / total;

    if (new_perc != perc)
    {
        std::uintmax_t done_mb = done >> 20;
        std::uintmax_t total_mb = total >> 20;
        std::cout << "\r" << new_perc << "% [" << done_mb << " MB / " << total_mb << " MB] " << std::flush;
        if (new_perc == 100)
        {
            std::cout << std::endl;
        }
        perc = new_perc;
    }
}

/**
 * Reads bytes from stream into given variable.
 *
 * @param stream Input stream
 * @param var Variable to read bytes into.
 */
template <typename T>
void read_into(std::ifstream &stream, T &var)
{
    stream.read((char *)&var, sizeof(T));
}

/**
 * Reads string of given length from input stream.
 *
 * @param stream Input stream
 * @param len Length of the string to be read
 *
 * @returns String value read from the stream
 */
std::string read_string(std::ifstream &stream, size_t len)
{
    std::vector<char> buf(len);
    stream.read(&buf.front(), len);
    return std::string(buf.data(), len);
}

int parse_dump(
    std::string filename,
    double start_timestamp,
    double end_timestamp,
    double interval_step,
    std::vector<double> &profiler_timestamps,
    std::map<std::string, std::vector<int>> &executed_instructions,
    std::map<char, std::vector<int>> &memory_accesses,
    std::map<std::string, std::map<char, std::vector<int>>> &peripheral_accesses,
    std::vector<int> &exceptions)
{
    std::ifstream dump_file(filename, std::ios_base::in | std::ios_base::binary);

    dump_file.seekg(0, dump_file.end);
    std::uintmax_t dump_file_size = dump_file.tellg();
    dump_file.seekg(0, dump_file.beg);

    if (!dump_file.is_open())
    {
        std::cout << "Error opening file " << filename << std::endl;
        return 1;
    }

    int cpu_count;
    read_into(dump_file, cpu_count);

    // parse cpus
    std::map<int, std::string> cpus;
    for (int i = 0; i < cpu_count; ++i)
    {
        int cpu_id, cpu_name_len;
        read_into(dump_file, cpu_id);
        read_into(dump_file, cpu_name_len);
        auto cpu_name = read_string(dump_file, cpu_name_len);
        cpus[cpu_id] = std::move(cpu_name);
    }

    int peripherals_count;
    read_into(dump_file, peripherals_count);

    // parse peripherals
    std::map<std::string, std::pair<unsigned long long, unsigned long long>> peripherals;
    for (int i = 0; i < peripherals_count; ++i)
    {
        int peripheral_name_len;
        unsigned long long start_addr, end_addr;
        read_into(dump_file, peripheral_name_len);
        auto peripheral_name = read_string(dump_file, peripheral_name_len);
        read_into(dump_file, start_addr);
        read_into(dump_file, end_addr);
        peripherals[peripheral_name] = {start_addr, end_addr};
    }

    // prepare containers for stats
    profiler_timestamps.clear();

    executed_instructions.clear();
    std::map<std::string, int> prev_instr_counter;

    memory_accesses.clear();

    peripheral_accesses.clear();

    exceptions.clear();

    int entries = 0;
    int invalid_entries = 0;

    while (dump_file.peek() != EOF)
    {
        if (0 == (entries % 1000))
        {
            print_progress(dump_file.tellg(), dump_file_size);
        }
        struct EntryHeader entry_header;
        read_into(dump_file, entry_header);

        ++entries;

        // check entry type
        if (entry_header.entry_type != ENTRY_TYPE_INSTRUCTION && entry_header.entry_type != ENTRY_TYPE_MEMORY &&
            entry_header.entry_type != ENTRY_TYPE_PERIPHERALS && entry_header.entry_type != ENTRY_TYPE_EXCEPTIONS)
        {
            std::cout << "Invalid entry type " << std::hex << (unsigned int)entry_header.entry_type << std::endl;
            return 1;
        }

        double virt_time = entry_header.virt_time / 1000;

        // ignore entry
        if (virt_time < start_timestamp || virt_time > end_timestamp)
        {
            switch (entry_header.entry_type)
            {
            case ENTRY_TYPE_INSTRUCTION:
            {
                struct EntryInstruction entry;
                read_into(dump_file, entry);
            }
            break;
            case ENTRY_TYPE_MEMORY:
            {
                struct EntryMemory entry;
                read_into(dump_file, entry);
            }
            break;
            case ENTRY_TYPE_PERIPHERALS:
            {
                struct EntryPeripherals entry;
                read_into(dump_file, entry);
            }
            break;
            case ENTRY_TYPE_EXCEPTIONS:
            {
                struct EntryExceptions entry;
                read_into(dump_file, entry);
            }
            break;
            default:
            {
                throw std::invalid_argument("Unhandled entry type");
            }
            }
            continue;
        }

        // align virtual time to interval step
        double interval_start = virt_time - std::fmod(virt_time, (interval_step / 1000));

        // check if new interval
        if (profiler_timestamps.empty() || profiler_timestamps.back() != interval_start)
        {
            profiler_timestamps.push_back(interval_start);

            for (const auto &cpu : cpus)
            {
                executed_instructions[cpus[cpu.first]].push_back(0);
            }

            memory_accesses[MEM_OP_READ].push_back(0);
            memory_accesses[MEM_OP_WRITE].push_back(0);

            for (const auto &peripheral : peripherals)
            {
                peripheral_accesses[peripheral.first][PERIPH_OP_READ].push_back(0);
                peripheral_accesses[peripheral.first][PERIPH_OP_WRITE].push_back(0);
            }

            exceptions.push_back(0);
        }

        switch (entry_header.entry_type)
        {
        case ENTRY_TYPE_INSTRUCTION:
        {
            struct EntryInstruction entry;
            read_into(dump_file, entry);

            if (cpus.count(entry.cpu_id))
            {
                int value = entry.instr_counter - prev_instr_counter[cpus[entry.cpu_id]];
                executed_instructions[cpus[entry.cpu_id]].back() += value;
                prev_instr_counter[cpus[entry.cpu_id]] = entry.instr_counter;
            }
            else
            {
                ++invalid_entries;
            }
        }
        break;
        case ENTRY_TYPE_MEMORY:
        {
            struct EntryMemory entry;
            read_into(dump_file, entry);

            switch (entry.op)
            {
            case MEM_OP_READ:
            case MEM_OP_WRITE:
                ++memory_accesses[entry.op].back();
                break;
            default:
                ++invalid_entries;
                break;
            }
        }
        break;
        case ENTRY_TYPE_PERIPHERALS:
        {
            struct EntryPeripherals entry;
            read_into(dump_file, entry);

            std::string found_peripheral;

            for (const auto &peripheral : peripherals)
            {
                const auto &addr = peripheral.second;
                if (addr.first <= entry.addr && entry.addr <= addr.second)
                {
                    found_peripheral = peripheral.first;
                    break;
                }
            }

            if (!found_peripheral.empty())
            {
                switch (entry.op)
                {
                case PERIPH_OP_READ:
                case PERIPH_OP_WRITE:
                    ++peripheral_accesses[found_peripheral][entry.op].back();
                    break;
                default:
                    ++invalid_entries;
                    break;
                }
            }
            else
            {
                ++invalid_entries;
            }
        }
        break;
        case ENTRY_TYPE_EXCEPTIONS:
        {
            struct EntryExceptions entry;
            read_into(dump_file, entry);

            ++exceptions.back();
        }
        break;
        default:
        {
            throw std::invalid_argument("Unhandled entry type");
        }
        }
    }

    print_progress(dump_file_size, dump_file_size);

    dump_file.close();

    return 0;
}
