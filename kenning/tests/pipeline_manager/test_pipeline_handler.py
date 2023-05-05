# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from kenning.tests.pipeline_manager.handler_tests import (
    HandlerTests,
    factory_test_create_dataflow,
    factory_test_equivalence,
)  # noqa: E501
from kenning.pipeline_manager.pipeline_handler import PipelineHandler

PET_DATASET_DATAFLOW_NODE = {
    "type": "PetDataset",
    "id": "0",
    "title": "PetDataset",
    "properties": {
        "dataset_root": {"value": "./build/PetDataset"},
        "inference_batch_size": {"value": 1},
        "download_dataset": {"value": False},
        "external_calibration_dataset": {"value": None},
        "classify_by": {"value": "breeds"},
        "image_memory_layout": {"value": "NHWC"},
    },
    "inputs": {},
    "outputs": {"Dataset": {"id": "1"}},
    "position": {"x": 46, "y": 173},
    "width": 300,
    "twoColumn": False,
}

TENSORFLOW_MOBILE_NET_DATAFLOW_NODE = {
    "type": "TensorFlowPetDatasetMobileNetV2",
    "id": "2",
    "title": "TensorFlowPetDatasetMobileNetV2",
    "properties": {
        "model_path": {
            "value": "./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5"  # noqa: E501
        }
    },
    "inputs": {"Dataset": {"id": "3"}},
    "outputs": {"ModelWrapper": {"id": "4"}, "Model": {"id": "5"}},
    "position": {"x": 400, "y": 50},
    "width": 300,
    "twoColumn": False,
}

TFLITE_RUNTIME_DATAFLOW_NODE = {
    "type": "TFLiteRuntime",
    "id": "6",
    "title": "TFLiteRuntime",
    "properties": {
        "disable_performance_measurements": {"value": True},
        "save_model_path": {"value": "./build/fp32.tflite"},
        "delegates_list": {"value": None},
        "num_threads": {"value": 4},
    },
    "inputs": {
        "ModelWrapper": {"id": "7"},
        "Model": {"id": "8"},
        "RuntimeProtocol": {"id": "9"},
    },
    "outputs": {},
    "position": {"x": 750, "y": 50},
    "width": 300,
    "twoColumn": False,
}


TFLITE_COMPILER_DATAFLOW_NODE = {
    "type": "TFLiteCompiler",
    "id": "10",
    "title": "TFLiteCompiler",
    "properties": {
        "compiled_model_path": {"value": "./build/fp32.tflite"},
        "epochs": {"value": 3},
        "batch_size": {"value": 32},
        "optimizer": {"value": "adam"},
        "disable_from_logits": {"value": False},
        "model_framework": {"value": "onnx"},
        "target": {"value": "default"},
        "inference_input_type": {"value": "float32"},
        "inference_output_type": {"value": "float32"},
        "dataset_percentage": {"value": 0.25},
        "quantization_aware_training": {"value": False},
        "use_tf_select_ops": {"value": False},
    },
    "inputs": {"Input model": {"id": "11"}},
    "outputs": {"Compiled model": {"id": "12"}},
    "position": {"x": 1100, "y": 50},
    "width": 300,
    "twoColumn": False,
}


class TestPipelineHandler(HandlerTests):
    dataflow_nodes = [
        PET_DATASET_DATAFLOW_NODE,
        TENSORFLOW_MOBILE_NET_DATAFLOW_NODE,
        TFLITE_RUNTIME_DATAFLOW_NODE,
        TFLITE_COMPILER_DATAFLOW_NODE,
    ]
    dataflow_connections = [
        {"id": "13", "from": "1", "to": "3"},
        {"id": "14", "from": "4", "to": "7"},
        {"id": "15", "from": "5", "to": "11"},
        {"id": "16", "from": "12", "to": "8"},
    ]

    @pytest.fixture(scope="class")
    def handler(self):
        return PipelineHandler()

    def equivalence_check(self, dataflow1, dataflow2):
        return dataflow1 == dataflow2

    PATH_TO_JSON_SCRIPTS = "./scripts/jsonconfigs"

    test_create_dataflow = factory_test_create_dataflow(PATH_TO_JSON_SCRIPTS)

    test_equivalence = factory_test_equivalence(PATH_TO_JSON_SCRIPTS)
