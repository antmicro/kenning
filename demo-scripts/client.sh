#!/bin/bash

python3 -m edge_ai_tester.scenarios.inference_client \
    edge_ai_tester.modelwrappers.classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2 \
    edge_ai_tester.compilers.tvm.TVMCompiler \
    edge_ai_tester.runtimeprotocols.network.NetworkProtocol \
    edge_ai_tester.runtimes.tvm.TVMRuntime \
    edge_ai_tester.datasets.pet_dataset.PetDataset \
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
