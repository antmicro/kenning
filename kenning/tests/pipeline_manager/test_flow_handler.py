# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from kenning.tests.pipeline_manager.handlertests import HandlerTests, create_dataflow_test_factory  # noqa: E501
from kenning.pipeline_manager.flow_handler import KenningFlowHandler


CAMERA_DATAPROVIDER_DATAFLOW_NODE = {
    "type": "CameraDataProvider",
    "id": "0",
    "name": "CameraDataProvider",
    "options": [
        ["video_file_path", "/dev/video0"],
        ["input_memory_layout", "NCHW"],
        ["input_color_format", "BGR"],
        ["input_width", 608],
        ["input_height", 608]
    ],
    "state": {},
    "interfaces": [
        ["Data", {
            "id": "1",
            "value": None,
            "isInput": False,
            "type": "data_runner"
        }]
    ],
    "position": {
        "x": 50,
        "y": 50
    },
    "width": 300,
    "twoColumn": False,
    "customClasses": ""
}

ONNXYOLO_DATAFLOW_NODE = {
    "type": "ONNXYOLOV4",
    "id": "2",
    "name": "ONNXYOLOV4",
    "options": [
        ["model_path", "./kenning/resources/models/detection/yolov4.onnx"]
    ],
    "state": {},
    "interfaces": [
        ["Model wrapper", {
            "id": "3",
            "value": None,
            "isInput": False,
            "type": "model_wrapper"
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

ONNXRUNTIME_DATAFLOW_NODE = {
    "type": "ONNXRuntime",
    "id": "4",
    "name": "ONNXRuntime",
    "options": [
        ["disable_performance_measurements", True],
        ["save_model_path", "./kenning/resources/models/detection/yolov4.onnx"],  # noqa: E501
        ["execution_providers", ["CUDAExecutionProvider"]]
    ],
    "state": {},
    "interfaces": [
        ["Runtime", {
            "id": "5",
            "value": None,
            "isInput": False,
            "type": "runtime"
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

NODELRUNTIMERUNNER_DATAFLOW_NODE = {
    "type": "ModelRuntimeRunner",
    "id": "6",
    "name": "ModelRuntimeRunner",
    "options": [],
    "state": {},
    "interfaces": [
        ["Input data", {
            "id": "7",
            "value": None,
            "isInput": True,
            "type": "data_runner"
        }],
        ["Model Wrapper", {
            "id": "8",
            "value": None,
            "isInput": True,
            "type": "model_wrapper"
        }],
        ["Runtime", {
            "id": "9",
            "value": None,
            "isInput": True,
            "type": "runtime"
        }],
        ["Calibration dataset", {
            "id": "10",
            "value": None,
            "isInput": True,
            "type": "dataset"
        }],
        ["Model output", {
            "id": "11",
            "value": None,
            "isInput": False,
            "type": "model_output"
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

DETECTIONVISUALIZER_DATAFLOW_NODE = {
    "type": "RealTimeDetectionVisualizer",
    "id": "14",
    "name": "RealTimeDetectionVisualizer",
    "options": [
        ["viewer_width",  512],
        ["viewer_height", 512],
        ["input_color_format", "BGR"],
        ["input_memory_layout", "NCHW"]
    ],
    "state": {},
    "interfaces": [
        ["Model output", {
            "id": "15",
            "value": None,
            "isInput": True,
            "type": "model_output"
        }],
        ["Input frames", {
            "id": "16",
            "value": None,
            "isInput": True,
            "type": "data_runner"
        }]
    ],
    "position": {
        "x": 1450,
        "y": 50
    },
    "width": 300,
    "twoColumn": False,
    "customClasses": ""
}


class TestFlowHandler(HandlerTests):
    dataflow_nodes = [
        CAMERA_DATAPROVIDER_DATAFLOW_NODE,
        ONNXYOLO_DATAFLOW_NODE,
        ONNXRUNTIME_DATAFLOW_NODE,
        NODELRUNTIMERUNNER_DATAFLOW_NODE,
        DETECTIONVISUALIZER_DATAFLOW_NODE
    ]
    dataflow_connections = [
        {
            "id": "12",
            "from": "3",
            "to": "8"
        },
        {
            "id": "13",
            "from": "5",
            "to": "9"
        },
        {
            "id": "17",
            "from": "11",
            "to": "15"
        },
        {
            "id": "18",
            "from": "1",
            "to": "7"
        },
        {
            "id": "19",
            "from": "1",
            "to": "16"
        }
    ]
    test_create_dataflow = create_dataflow_test_factory(
        "./scripts/jsonflowconfigs"
    )

    @pytest.fixture(scope="class")
    def handler(self):
        return KenningFlowHandler()
