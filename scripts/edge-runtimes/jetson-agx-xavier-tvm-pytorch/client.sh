#!/bin/bash

python3 -m edge_ai_tester.scenarios.inference_tester \
    edge_ai_tester.modelwrappers.classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2 \
    edge_ai_tester.compilers.tvm.TVMCompiler \
    edge_ai_tester.runtimes.tvm.TVMRuntime \
    edge_ai_tester.datasets.pet_dataset.PetDataset \
    ./build/jetson-agx-xavier-tvm-pytorch.json \
    --protocol-cls edge_ai_tester.runtimeprotocols.network.NetworkProtocol \
    --model-path ./edge_ai_tester/resources/models/classification/pytorch_pet_dataset_mobilenetv2.pth \
    --convert-to-onnx ./build/pytorch_pet_dataset_mobilenetv2.onnx \
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
