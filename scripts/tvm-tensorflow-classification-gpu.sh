#!/bin/bash

python3 -m edge_ai_tester.scenarios.inference_tester \
    edge_ai_tester.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    edge_ai_tester.compilers.tvm.TVMCompiler \
    edge_ai_tester.runtimes.tvm.TVMRuntime \
    edge_ai_tester.datasets.pet_dataset.PetDataset \
    ./build/local-gpu-tvm-tensorflow-classification.json \
    --model-path ./edge_ai_tester/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --target "cuda" \
    --compiled-model-path ./build/compiled-model.tar \
    --opt-level 3 \
    --save-model-path ./build/compiled-model.tar \
    --target-device-context cuda \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO
