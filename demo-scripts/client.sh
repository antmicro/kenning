#!/bin/bash

python3 -m dl_framework_analyzer.scenarios.inference_client \
    dl_framework_analyzer.modelwrappers.classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2 \
    dl_framework_analyzer.compilers.tvm.TVMCompiler \
    dl_framework_analyzer.runtimeprotocols.network.NetworkProtocol \
    dl_framework_analyzer.runtimes.tvm.TVMRuntime \
    dl_framework_analyzer.datasets.pet_dataset.PetDataset \
    ./build/before-compile.onnx \
    ./build/xavier-output \
    jetson-agx-xavier-tvm \
    --model-path build/pytorch-net-2.pth \
    --target "nvidia/jetson-agx-xavier" \
    --target-host "llvm -mtriple=aarch64-linux-gnu" \
    --compiled-model-path ./build/compiled-model.tar \
    --opt-level 3 \
    --host 192.168.188.100 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/nvidia/compiled-model.tar \
    --target-device-context cuda \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO
