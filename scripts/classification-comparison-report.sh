#!/bin/bash

python3 -m kenning.scenarios.inference_tester \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.datasets.pet_dataset.PetDataset \
    ./build/pet-dataset-tvm.json \
    --compiler-cls kenning.compilers.tvm.TVMCompiler \
    --runtime-cls kenning.runtimes.tvm.TVMRuntime \
    --model-path ./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --target cuda \
    --compiled-model-path ./build/compiled-model.tar \
    --opt-level 3 \
    --save-model-path ./build/compiled-model.tar \
    --target-device-context cuda \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --download-dataset \
    --verbosity INFO

python -m kenning.scenarios.inference_tester \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.datasets.pet_dataset.PetDataset \
    ./build/pet-dataset-tflite.json \
    --compiler-cls kenning.compilers.tflite.TFLiteCompiler \
    --runtime-cls kenning.runtimes.tflite.TFLiteRuntime \
    --model-path ./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --compiled-model-path ./build/compiled-model.tflite \
    --save-model-path ./build/compiled-model.tflite \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO

python -m kenning.scenarios.inference_tester \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.datasets.pet_dataset.PetDataset \
    ./build/pet-dataset-iree.json \
    --compiler-cls kenning.compilers.iree.IREECompiler \
    --runtime-cls kenning.runtimes.iree.IREERuntime \
    --model-path ./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --backend vulkan \
    --compiled-model-path ./build/compiled-model.vmbf \
    --save-model-path ./build/compiled-model.vmbf \
    --driver vulkan \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO

python -m kenning.scenarios.render_report \
    "Classification comparison on Pet Dataset" \
    build/pet-report \
    --report-types performance classification \
    --measurements build/pet-dataset-tvm.json build/pet-dataset-tflite.json build/pet-dataset-iree.json