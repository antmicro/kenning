# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from kenning.tests.pipeline_manager.handlertests import HandlerTests
from kenning.pipeline_manager.pipeline_handler import PipelineHandler
from kenning.datasets.pet_dataset import PetDataset
from kenning.modelwrappers.classification.tensorflow_pet_dataset import TensorFlowPetDatasetMobileNetV2  # noqa: E501
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.compilers.tflite import TFLiteCompiler
from kenning.utils.args_manager import get_parsed_json_dict

PET_DATASET_DATAFLOW_NODE = {
    "type": "PetDataset",
    "id": "0",
    "name": "PetDataset",
    "options": [
        ["dataset_root", "./build/PetDataset"],
        ["inference_batch_size", 1],
        ["download_dataset", False],
        ["external_calibration_dataset", None],
        ["classify_by", "breeds"],
        ["image_memory_layout", "NHWC"]
    ],
    "state": {},
    "interfaces": [
        ["Dataset", {
            "id": "1",
            "value": None,
            "isInput": False,
            "type": "dataset"
        }]
    ],
    "position": {
        "x": 46,
        "y": 173
    },
    "width": 300,
    "twoColumn": False,
    "customClasses": ""
}

TENSORFLOW_MOBILE_NET_DATAFLOW_NODE = {
    "type": "TensorFlowPetDatasetMobileNetV2",
    "id": "2",
    "name": "TensorFlowPetDatasetMobileNetV2",
    "options": [
        ["model_path", "./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5"]  # noqa: E501
    ],
    "state": {},
    "interfaces": [
        ["Dataset", {
            "id": "3",
            "value": None,
            "isInput": True,
            "type": "dataset"
        }],
        ["ModelWrapper", {
            "id": "4",
            "value": None,
            "isInput": False,
            "type": "model_wrapper"
        }],
        ["Model", {
            "id": "5",
            "value": None,
            "isInput": False,
            "type": "model"
        }]
    ],
    "position": {
        "x": 400,
        "y": 50
    },
    "width": 300,
    "twoColumn": False,
    "customClasses": ""
}

TFLITE_RUNTIME_DATAFLOW_NODE = {
    "type": "TFLiteRuntime",
    "id": "6",
    "name": "TFLiteRuntime",
    "options": [
        ["disable_performance_measurements", True],
        ["save_model_path", "./build/fp32.tflite"],
        ["delegates_list", None],
        ["num_threads", 4]
    ],
    "state": {},
    "interfaces": [
        ["ModelWrapper", {
            "id": "7",
            "value": None,
            "isInput": True,
            "type": "model_wrapper"
        }],
        ["Model", {
            "id": "8",
            "value": None,
            "isInput": True,
            "type": "model"
        }],
        ["RuntimeProtocol", {
            "id": "9",
            "value": None,
            "isInput": True,
            "type": "runtime_protocol"
        }]
    ],
    "position": {
        "x": 750,
        "y": 50
    },
    "width": 300,
    "twoColumn": False,
    "customClasses": ""
}


TFLITE_COMPILER_DATAFLOW_NODE = {
    "type": "TFLiteCompiler",
    "id": "10",
    "name": "TFLiteCompiler",
    "options": [
        ["compiled_model_path", "./build/fp32.tflite"],
        ["epochs", 3],
        ["batch_size", 32],
        ["optimizer", "adam"],
        ["disable_from_logits", False],
        ["model_framework", "onnx"],
        ["target", "default"],
        ["inference_input_type", "float32"],
        ["inference_output_type", "float32"],
        ["dataset_percentage", 0.25],
        ["quantization_aware_training", False],
        ["use_tf_select_ops", False]
    ],
    "state": {},
    "interfaces": [
        ["Input model", {
            "id": "11",
            "value": None,
            "isInput": True,
            "type": "model"
        }],
        ["Compiled model", {
            "id": "12",
            "value": None,
            "isInput": False,
            "type": "model"
        }]
    ],
    "position": {
        "x": 1100,
        "y": 50
    },
    "width": 300,
    "twoColumn": False,
    "customClasses": ""
}


class TestPipelineHandler(HandlerTests):
    dataflow_nodes = [
        PET_DATASET_DATAFLOW_NODE,
        TENSORFLOW_MOBILE_NET_DATAFLOW_NODE,
        TFLITE_RUNTIME_DATAFLOW_NODE,
        TFLITE_COMPILER_DATAFLOW_NODE
    ]
    dataflow_connections = [
        {
            "id": "13",
            "from": "1",
            "to": "3"
        },
        {
            "id": "14",
            "from": "4",
            "to": "7"
        },
        {
            "id": "15",
            "from": "5",
            "to": "11"
        },
        {
            "id": "16",
            "from": "12",
            "to": "8"
        }
    ]

    @pytest.fixture(scope="class")
    def handler(self):
        return PipelineHandler()

    @pytest.fixture(scope="class")
    def pipeline_json(self):
        clss = [
            ('dataset', PetDataset, {'dataset_root': './build/PetDataset'}),
            ('model_wrapper', TensorFlowPetDatasetMobileNetV2, {'model_path': './kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5'}),  # noqa: E501
            ('optimizers', TFLiteCompiler, {'compiled_model_path': './build/model.tflite'}),  # noqa: E501
            ('runtime', TFLiteRuntime, {}),
        ]
        json = {}
        for type_, cls, arg_json in clss:
            parameterschema = cls.form_parameterschema()
            kenning_parameters = get_parsed_json_dict(
                parameterschema, arg_json
            )
            kenning_module = {
                'type': f'{cls.__module__}.{cls.__name__}',
                'parameters': kenning_parameters
            }
            if type_ == 'optimizers':
                kenning_module = [kenning_module]
            json[type_] = kenning_module
        return json
