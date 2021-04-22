#!/bin/bash

python3 -m edge_ai_tester.scenarios.inference_client \
    edge_ai_tester.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    edge_ai_tester.compilers.tvm.TVMCompiler \
    edge_ai_tester.runtimeprotocols.network.NetworkProtocol \
    edge_ai_tester.runtimes.tvm.TVMRuntime \
    edge_ai_tester.datasets.pet_dataset.PetDataset \
    ./build/jetson-agx-xavier-tvm-tensorflow.json \
    --model-path ./edge_ai_tester/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
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
