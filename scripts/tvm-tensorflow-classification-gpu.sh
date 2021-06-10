#!/bin/bash

python3 -m kenning.scenarios.inference_tester \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.compilers.tvm.TVMCompiler \
    kenning.runtimes.tvm.TVMRuntime \
    kenning.datasets.pet_dataset.PetDataset \
    ./build/local-gpu-tvm-tensorflow-classification.json \
    --model-path ./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --target "cuda" \
    --compiled-model-path ./build/compiled-model.tar \
    --opt-level 3 \
    --save-model-path ./build/compiled-model.tar \
    --target-device-context cuda \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO
