# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from kenning.pipeline_manager.core import VisualEditorGraphParserError
from kenning.pipeline_manager.flow_handler import KenningFlowHandler
from kenning.runners.modelruntime_runner import ModelRuntimeRunner
from kenning.tests.pipeline_manager.handler_tests import (
    HandlerTests,
    factory_test_create_dataflow,
    factory_test_equivalence,
)
from kenning.utils.class_loader import load_class

CAMERA_DATAPROVIDER_DATAFLOW_NODE = {
    "name": "CameraDataProvider",
    "id": "0",
    "position": {"x": 50, "y": 50},
    "width": 300,
    "twoColumn": False,
    "interfaces": [
        {"name": "Data", "id": "1", "direction": "output", "side": "right"}
    ],
    "properties": [
        {"name": "video_file_path", "id": "2", "value": "/dev/video0"},
        {"name": "input_memory_layout", "id": "3", "value": "NCHW"},
        {
            "name": "input_color_format",
            "id": "a9966a81-edbb-4b1c-b15a-a7f7c24823c9",
            "value": "BGR",
        },
        {"name": "input_width", "id": "4", "value": 608},
        {"name": "input_height", "id": "5", "value": 608},
    ],
}

ONNXYOLO_DATAFLOW_NODE = {
    "name": "ONNXYOLOV4",
    "id": "6",
    "position": {"x": 400, "y": 50},
    "width": 300,
    "twoColumn": False,
    "interfaces": [
        {
            "name": "Model wrapper",
            "id": "7",
            "direction": "output",
            "side": "right",
        }
    ],
    "properties": [
        {
            "name": "classes",
            "id": "66c5b122-711e-4e87-afd5-7dfda9b633b6",
            "value": "coco",
        },
        {
            "name": "model_path",
            "id": "8",
            "value": "kenning:///models/detection/yolov4.onnx",
        },
    ],
}

ONNXRUNTIME_DATAFLOW_NODE = {
    "name": "ONNXRuntime",
    "id": "9",
    "position": {"x": 750, "y": 50},
    "width": 300,
    "twoColumn": False,
    "interfaces": [
        {"name": "Runtime", "id": "10", "direction": "output", "side": "right"}
    ],
    "properties": [
        {
            "name": "save_model_path",
            "id": "11",
            "value": "kenning:///models/detection/yolov4.onnx",
        },
        {
            "name": "execution_providers",
            "id": "12",
            "value": ["CUDAExecutionProvider"],
        },
        {
            "name": "disable_performance_measurements",
            "id": "9fcaafdc-5f09-4204-a54e-fa0f66abd804",
            "value": False,
        },
    ],
}

MODELRUNTIMERUNNER_DATAFLOW_NODE = {
    "name": "ModelRuntimeRunner",
    "id": "13",
    "position": {"x": 1100, "y": 50},
    "width": 300,
    "twoColumn": False,
    "interfaces": [
        {
            "name": "Input data",
            "id": "14",
            "direction": "input",
            "side": "left",
        },
        {
            "name": "Model Wrapper",
            "id": "15",
            "direction": "input",
            "side": "left",
        },
        {"name": "Runtime", "id": "16", "direction": "input", "side": "left"},
        {
            "name": "Calibration dataset",
            "id": "17",
            "direction": "input",
            "side": "left",
        },
        {
            "name": "Model output",
            "id": "18",
            "direction": "output",
            "side": "right",
        },
    ],
    "properties": [],
}

DETECTIONVISUALIZER_DATAFLOW_NODE = {
    "name": "RealTimeDetectionVisualizer",
    "id": "21",
    "position": {"x": 1450, "y": 50},
    "width": 300,
    "twoColumn": False,
    "interfaces": [
        {
            "name": "Model output",
            "id": "22",
            "direction": "input",
            "side": "left",
        },
        {
            "name": "Input frames",
            "id": "23",
            "direction": "input",
            "side": "left",
        },
    ],
    "properties": [
        {"name": "viewer_width", "id": "24", "value": 512},
        {"name": "viewer_height", "id": "25", "value": 512},
        {"name": "input_color_format", "id": "27", "value": "BGR"},
        {"name": "input_memory_layout", "id": "26", "value": "NCHW"},
    ],
}


@pytest.fixture(scope="function")
def use_static_io_spec_parser():
    """
    Changes the ModelRuntimeRunner `get_io_specification` implementation to use
    runtime's static IO spec parser.
    """

    def _create_model(dataset, json_dict):
        cls = load_class(json_dict["type"])
        model = cls.from_json(
            dataset=dataset, json_dict=json_dict["parameters"]
        )
        model._json_dict = json_dict["parameters"]
        return model

    def get_io_specification(self):
        return self._get_io_specification(
            self.model.parse_io_specification_from_json(self.model._json_dict)
        )

    ModelRuntimeRunner._create_model = _create_model
    ModelRuntimeRunner.get_io_specification = get_io_specification


@pytest.mark.xdist_group(name="use_resources")
@pytest.mark.usefixtures("use_static_io_spec_parser")
class TestFlowHandler(HandlerTests):
    dataflow_nodes = [
        CAMERA_DATAPROVIDER_DATAFLOW_NODE,
        ONNXYOLO_DATAFLOW_NODE,
        ONNXRUNTIME_DATAFLOW_NODE,
        MODELRUNTIMERUNNER_DATAFLOW_NODE,
        DETECTIONVISUALIZER_DATAFLOW_NODE,
    ]
    dataflow_connections = [
        {"id": "19", "from": "7", "to": "15"},
        {"id": "20", "from": "10", "to": "16"},
        {"id": "28", "from": "1", "to": "14"},
        {"id": "29", "from": "1", "to": "23"},
        {"id": "30", "from": "18", "to": "22"},
    ]

    @pytest.fixture(scope="class")
    def handler(self):
        return KenningFlowHandler(layout_algorithm="NoLayout")

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
            for local_name in node1.get("inputs", {}):
                connection_check(node1["inputs"], node2["inputs"], local_name)

            for local_name in node1.get("outputs", {}):
                connection_check(
                    node1["outputs"], node2["outputs"], local_name
                )
        return True

    PATH_TO_JSON_SCRIPTS = "./scripts/jsonflowconfigs"

    test_create_dataflow = factory_test_create_dataflow(PATH_TO_JSON_SCRIPTS)

    test_equivalence = factory_test_equivalence(PATH_TO_JSON_SCRIPTS)

    def test_create_dataflow_fail(self, handler):
        """
        Test if the handler correctly fails when the JSON is invalid.
        """
        with pytest.raises(VisualEditorGraphParserError) as e:
            invalid_flow_json = [
                {
                    "type": "Unknown",
                }
            ]
            handler.create_dataflow(invalid_flow_json)
        assert "Invalid Kenningflow: Each node must" in str(e.value)

        with pytest.raises(VisualEditorGraphParserError) as e:
            invalid_flow_json = [
                {
                    "parameters": {},
                    "test": "test",
                }
            ]
            handler.create_dataflow(invalid_flow_json)
        assert "Invalid Kenningflow: Each node must" in str(e.value)
