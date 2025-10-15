#!/bin/bash

# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

set -e

mkdir -p $ZEPHYR_WORKSPACE && pushd $ZEPHYR_WORKSPACE
git clone https://github.com/antmicro/kenning-zephyr-runtime.git
cd kenning-zephyr-runtime

python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install pip setuptools --upgrade
pip install -r requirements.txt
west init -l .
west update
west zephyr-export
west packages pip --install
west sdk install --toolchains x86_64-zephyr-elf arm-zephyr-eabi riscv64-zephyr-elf
./scripts/prepare_modules.sh
source ./scripts/prepare_renode.sh

popd
