/*
 * Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <map>
#include <string>
#include <vector>

/**
 * Parses Renode profiler dump. The parsed entries are returned via passed container references.
 *
 * @param filename Path to the dump file
 * @param start_timestamp Timestamp from which the parsing should start
 * @param end_timestamp Timestamp to which parsing should be done
 * @param interval_step Interval on which time should be divided
 * @param profiler_timestamps Container for parsed profiler timestamps
 * @param executed_instructions Container for parsed executed instructions
 * @param memory_accesses Container for parsed memory accesses
 * @param peripheral_accesses Container for parsed peripheral accesses
 * @param exceptions Container for parsed exceptions
 *
 * @returns parsing error code.
 */
int parse_dump(
    std::string filename,
    double start_timestamp,
    double end_timestamp,
    double interval_step,
    std::vector<double> &profiler_timestamps,
    std::map<std::string, std::vector<int>> &executed_instructions,
    std::map<char, std::vector<int>> &memory_accesses,
    std::map<std::string, std::map<char, std::vector<int>>> &peripheral_accesses,
    std::vector<int> &exceptions);
