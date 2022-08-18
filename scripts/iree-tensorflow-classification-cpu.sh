#!/bin/bash

python3 -m kenning.scenarios.inference_tester \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.datasets.pet_dataset.PetDataset \
    ./build/local-cpu-iree-tensorflow-classification.json \
    --compiler-cls kenning.compilers.iree.IREECompiler \
    --runtime-cls kenning.runtimes.iree.IREERuntime \
    --model-path ./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --backend dylib \
    --compiled-model-path ./build/compiled-model.vmbf \
    --save-model-path ./build/compiled-model.vmbf \
    --driver dylib \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO