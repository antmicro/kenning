#!/bin/bash

python3 -m kenning.scenarios.inference_tester \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.runtimes.iree.IREERuntime \
    kenning.datasets.pet_dataset.PetDataset \
    ./build/local-cpu-iree-tensorflow-classification.json \
    --modelcompiler-cls kenning.compilers.iree.IREECompiler \
    --model-path ./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --backend vulkan \
    --compiled-model-path ./build/compiled-model.vmbf \
    --save-model-path ./build/compiled-model.vmbf \
    --driver vulkan \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO