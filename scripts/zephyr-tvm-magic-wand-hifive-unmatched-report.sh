#!/bin/bash

# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

set -e

pushd $ZEPHYR_WORKSPACE/kenning-zephyr-runtime
mkdir -p build/
python -m kenning optimize test report \
    --cfg $SCENARIO_PATH \
    --measurements ./results.json \
    --report-path $DOCS_DIR/source/generated/$REPORT_NAME \
    --root-dir $DOCS_DIR/source/ \
    --img-dir $DOCS_DIR/source/generated/img/ \
    --report-name "Sample RISC-V Zephyr Tracing Report" \
    --verbosity INFO \
    --to-html

popd
