#!/bin/bash

# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

set -e

mkdir -p $ZEPHYR_WORKSPACE && pushd $ZEPHYR_WORKSPACE
git clone https://github.com/antmicro/kenning-zephyr-runtime.git
cd kenning-zephyr-runtime

./scripts/prepare_zephyr_env.sh
./scripts/prepare_modules.sh
source .venv/bin/activate
source ./scripts/prepare_renode.sh

popd
