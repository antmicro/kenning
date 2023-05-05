# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from kenning.runners.modelruntime_runner import ModelRuntimeRunner
from kenning.tests.pipeline_manager.handler_tests import HandlerTests, factory_test_create_dataflow, factory_test_equivalence  # noqa: E501
from kenning.pipeline_manager.flow_handler import KenningFlowHandler
from kenning.utils.class_loader import load_class

CAMERA_DATAPROVIDER_DATAFLOW_NODE = {
    "type": "CameraDataProvider",
    "id": "0",
    "title": "CameraDataProvider",
    "properties": {
        "video_file_path": {"value": "/dev/video0"},
        "input_memory_layout": {"value": "NCHW"},
        "input_color_format": {"value": "BGR"},
        "input_width": {"value": 608},
        "input_height": {"value": 608},
    },
    "outputs": {"Data": {"id": "1"}},
    "inputs": {},
    "position": {"x": 50, "y": 50},
    "width": 300,
    "twoColumn": False,
}

ONNXYOLO_DATAFLOW_NODE = {
    "type": "ONNXYOLOV4",
    "id": "2",
    "title": "ONNXYOLOV4",
    "properties": {
        "model_path": {"value": "./kenning/resources/models/detection/yolov4.onnx"}  # noqa: E501
    },
    "outputs": {"Model wrapper": {"id": "3"}},
    "inputs": {},
    "position": {"x": 400, "y": 50},
    "width": 300,
    "twoColumn": False,
}

ONNXRUNTIME_DATAFLOW_NODE = {
    "type": "ONNXRuntime",
    "id": "4",
    "title": "ONNXRuntime",
    "properties": {
        "disable_performance_measurements": {"value": True},
        "save_model_path": {
            "value": "./kenning/resources/models/detection/yolov4.onnx"
        },
        "execution_providers": {"value": ["CUDAExecutionProvider"]},
    },
    "outputs": {"Runtime": {"id": "5"}},
    "inputs": {},
    "position": {"x": 750, "y": 50},
    "width": 300,
    "twoColumn": False,
}

MODELRUNTIMERUNNER_DATAFLOW_NODE = {
    "type": "ModelRuntimeRunner",
    "id": "6",
    "title": "ModelRuntimeRunner",
    "properties": {},
    "inputs": {
        "Input data": {"id": "7"},
        "Model Wrapper": {"id": "8"},
        "Runtime": {"id": "9"},
        "Calibration dataset": {"id": "10"},
    },
    "outputs": {"Model output": {"id": "11"}},
    "position": {"x": 1100, "y": 50},
    "width": 300,
    "twoColumn": False,
}

DETECTIONVISUALIZER_DATAFLOW_NODE = {
    "type": "RealTimeDetectionVisualizer",
    "id": "14",
    "title": "RealTimeDetectionVisualizer",
    "properties": {
        "viewer_width": {"value": 512},
        "viewer_height": {"value": 512},
        "input_color_format": {"value": "BGR"},
        "input_memory_layout": {"value": "NCHW"},
    },
    "inputs": {"Model output": {"id": "15"}, "Input framer": {"id": "16"}},
    "outputs": {},
    "position": {"x": 1450, "y": 50},
    "width": 300,
    "twoColumn": False,
}


@pytest.fixture(scope='function')
def use_static_io_spec_parser():
    """
    Changes the ModelRuntimeRunner `get_io_specification` implementation to use
    runtime's static IO spec parser.
    """
    def _create_model(dataset, json_dict):
        cls = load_class(json_dict['type'])
        model = cls.from_json(
            dataset=dataset,
            json_dict=json_dict['parameters'])
        model._json_dict = json_dict['parameters']
        return model

    def get_io_specification(self):
        return self._get_io_specification(
            self.model.parse_io_specification_from_json(
                self.model._json_dict
            )
        )

    ModelRuntimeRunner._create_model = _create_model
    ModelRuntimeRunner.get_io_specification = get_io_specification


@pytest.mark.usefixtures('use_static_io_spec_parser')
class TestFlowHandler(HandlerTests):
    dataflow_nodes = [
        CAMERA_DATAPROVIDER_DATAFLOW_NODE,
        ONNXYOLO_DATAFLOW_NODE,
        ONNXRUNTIME_DATAFLOW_NODE,
        MODELRUNTIMERUNNER_DATAFLOW_NODE,
        DETECTIONVISUALIZER_DATAFLOW_NODE
    ]
    dataflow_connections = [
        {"id": "12", "from": "3", "to": "8"},
        {"id": "13", "from": "5", "to": "9"},
        {"id": "17", "from": "11", "to": "15"},
        {"id": "18", "from": "1", "to": "7"},
        {"id": "19", "from": "1", "to": "16"},
    ]

    @pytest.fixture(scope="class")
    def handler(self):
        return KenningFlowHandler()

    def equivalence_check(self, dataflow1, dataflow2):
        # There is a degree of freedom when naming global connections when
        # defining KenningFlows. Two JSOns are equivalent when there is
        # 1-to-1 mapping between global connection names.
        conn_name_mapping = {}

        def connection_check(node1_io, node2_io, local_name):
            global_name1 = node1_io[local_name]
            global_name2 = node2_io[local_name]
            if global_name1 in conn_name_mapping:
                assert conn_name_mapping[global_name1] == global_name2
            else:
                assert global_name2 not in conn_name_mapping.values()
                conn_name_mapping[global_name1] = global_name2

        for node1, node2 in zip(dataflow1, dataflow2):
            for local_name in node1.get('inputs', {}):
                connection_check(node1['inputs'], node2['inputs'], local_name)

            for local_name in node1.get('outputs', {}):
                connection_check(
                    node1['outputs'],  node2['outputs'], local_name
                )
        return True

    PATH_TO_JSON_SCRIPTS = "./scripts/jsonflowconfigs"

    test_create_dataflow = factory_test_create_dataflow(
        PATH_TO_JSON_SCRIPTS
    )

    test_equivalence = factory_test_equivalence(
        PATH_TO_JSON_SCRIPTS
    )
