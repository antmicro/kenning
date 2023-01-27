#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python3 -m kenning.scenarios.inference_tester \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.datasets.pet_dataset.PetDataset \
    ./build/jetson-agx-xavier-tvm-tensorflow.json \
    --compiler-cls kenning.compilers.tvm.TVMCompiler \
    --runtime-cls kenning.runtimes.tvm.TVMRuntime \
    --protocol-cls kenning.runtimeprotocols.network.NetworkProtocol \
    --model-path ./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --target "nvidia/jetson-agx-xavier" \
    --target-host "llvm -mtriple=aarch64-linux-gnu" \
    --compiled-model-path ./build/compiled-model.tar \
    --opt-level 3 \
    --host $1 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/nvidia/compiled-model.tar \
    --target-device-context cuda \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO
