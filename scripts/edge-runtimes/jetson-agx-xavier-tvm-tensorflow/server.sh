#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning.scenarios.inference_server \
    --protocol-cls kenning.protocols.network.NetworkProtocol \
    --runtime-cls kenning.runtimes.tvm.TVMRuntime \
    --host 0.0.0.0 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/nvidia/compiled-model.tar \
    --target-device-context cuda \
    --input-dtype float32 \
    --verbosity INFO
