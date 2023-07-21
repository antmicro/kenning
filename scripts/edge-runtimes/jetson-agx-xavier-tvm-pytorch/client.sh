#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning.scenarios.inference_tester \
    --modelwrapper-cls kenning.modelwrappers.classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2 \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --measurements ./build/jetson-agx-xavier-tvm-pytorch.json \
    --compiler-cls kenning.compilers.tvm.TVMCompiler \
    --runtime-cls kenning.runtimes.tvm.TVMRuntime \
    --protocol-cls kenning.runtimeprotocols.network.NetworkProtocol \
    --model-path kenning:///models/classification/pytorch_pet_dataset_mobilenetv2.pth \
    --convert-to-onnx ./build/pytorch_pet_dataset_mobilenetv2.onnx \
    --target "cuda -keys=cuda,gpu -libs=cudnn,cublas -arch=sm_72 -max_num_threads=1024 -max_threads_per_block=1024 -registers_per_block=65536 -shared_memory_per_block=49152 -thread_warp_size=32" \
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
