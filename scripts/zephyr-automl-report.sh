#!/bin/bash

# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

set -e

cd $ZEPHYR_WORKSPACE/kenning-zephyr-runtime

mkdir -p workspace
python3 -m kenning automl optimize test report \
  --cfg $AUTOML_SCENARIO_CONFIG \
  --report-path $DOCS_DIR/source/generated/$REPORT_NAME \
  --allow-failures --to-html --ver DEBUG \
  --root-dir $DOCS_DIR/source \
  --img-dir $DOCS_DIR/source/generated/img \
  --comparison-only \
  --smaller-header
