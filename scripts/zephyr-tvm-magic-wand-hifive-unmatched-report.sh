#!/bin/bash

# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

set -e

SCENARIO_PATH_VAL=${SCENARIO_PATH:-./scripts/configs/zephyr-tvm-magic-wand-hifive-unmatched-report.yml}
REPORT_NAME_VAL=${REPORT_NAME:-sample-riscv-zephyr-tracing-report.md}
OUTPUT_DIR_VAL=${OUTPUT_DIR:-./docs}

pushd $ZEPHYR_WORKSPACE/kenning-zephyr-runtime
mkdir -p build/
python -m kenning report \
    --cfg $SCENARIO_PATH_VAL \
    --measurements ./results.json \
    --report-path $OUTPUT_DIR_VAL/source/generated/$REPORT_NAME_VAL \
    --report-types \
        classification \
        performance \
        renode_stats \
        zephyr_traces \
        model \
    --root-dir $OUTPUT_DIR_VAL/source/ \
    --img-dir $OUTPUT_DIR_VAL/source/generated/img/ \
    --report-name "Sample RISC-V Zephyr Tracing Report" \
    --verbosity INFO \
    --to-html

popd
