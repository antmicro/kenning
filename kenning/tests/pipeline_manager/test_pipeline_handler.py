# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from kenning.core.exceptions import VisualEditorGraphParserError
from kenning.pipeline_manager.pipeline_handler import PipelineHandler
from kenning.tests.pipeline_manager.handler_tests import (
    HandlerTests,
    factory_test_create_dataflow,
    factory_test_equivalence,
)

PET_DATASET_DATAFLOW_NODE = {
    "name": "PetDataset",
    "id": "0",
    "position": {"x": 50, "y": 50},
    "width": 300,
    "twoColumn": False,
    "interfaces": [
        {"name": "Dataset", "id": "1", "direction": "output", "side": "right"}
    ],
    "properties": [
        {
            "name": "classify_by",
            "id": "4844ae38-2378-44f4-ad3c-62e1e1aded31",
            "value": "breeds",
        },
        {
            "name": "image_memory_layout",
            "id": "ade5a0dc-08fc-4004-b88e-dcc2f324ac23",
            "value": "NHWC",
        },
        {"name": "dataset_root", "id": "2", "value": "./build/pet-dataset"},
        {
            "name": "inference_batch_size",
            "id": "a1153125-58a6-43c9-a562-2aecc9d256fd",
            "value": 1,
        },
        {"name": "download_dataset", "id": "3", "value": False},
        {
            "name": "external_calibration_dataset",
            "id": "53187108-482d-4a7a-91a5-90c5f385cc8e",
            "value": "",
        },
        {
            "name": "split_fraction_test",
            "id": "30c0e40f-0989-4349-a48c-ff942e02b754",
            "value": 0.2,
        },
        {
            "name": "split_fraction_val",
            "id": "bacf9515-5b43-4b34-b2f5-4032f481979c",
            "value": 0,
        },
        {
            "name": "split_seed",
            "id": "2910e9c3-a385-4eb0-909a-ecf03e28b2d7",
            "value": 1234,
        },
    ],
}

TENSORFLOW_MOBILE_NET_DATAFLOW_NODE = {
    "name": "TensorFlowPetDatasetMobileNetV2",
    "id": "4",
    "position": {"x": 400, "y": 50},
    "width": 300,
    "twoColumn": False,
    "interfaces": [
        {"name": "Dataset", "id": "5", "direction": "input", "side": "left"},
        {
            "name": "ModelWrapper",
            "id": "6",
            "direction": "output",
            "side": "right",
        },
        {"name": "Model", "id": "7", "direction": "output", "side": "right"},
    ],
    "properties": [
        {
            "name": "model_path",
            "id": "8",
            "value": "kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5",
        }
    ],
}

TFLITE_RUNTIME_DATAFLOW_NODE = {
    "name": "TFLiteRuntime",
    "id": "9",
    "position": {"x": 750, "y": 50},
    "width": 300,
    "twoColumn": False,
    "interfaces": [
        {
            "name": "ModelWrapper",
            "id": "10",
            "direction": "input",
            "side": "left",
        },
        {"name": "Model", "id": "11", "direction": "input", "side": "left"},
        {"name": "Protocol", "id": "12", "direction": "input", "side": "left"},
    ],
    "properties": [
        {
            "name": "save_model_path",
            "id": "13",
            "value": "./build/fp32.tflite",
        },
        {
            "name": "delegates_list",
            "id": "db50a878-acfc-44f5-a537-1ebbd6bc49cd",
            "value": [],
        },
        {
            "name": "num_threads",
            "id": "2a17c576-0d4d-4787-b66e-7f82bcdcb2a7",
            "value": 4,
        },
        {
            "name": "disable_performance_measurements",
            "id": "37e8fa07-4142-414a-9796-20c523c81310",
            "value": False,
        },
    ],
}


TFLITE_COMPILER_DATAFLOW_NODE = {
    "name": "TFLiteCompiler",
    "id": "14",
    "position": {"x": 1100, "y": 50},
    "width": 300,
    "twoColumn": False,
    "interfaces": [
        {
            "name": "Input model",
            "id": "15",
            "direction": "input",
            "side": "left",
        },
        {
            "name": "Compiled model",
            "id": "16",
            "direction": "output",
            "side": "right",
        },
    ],
    "properties": [
        {
            "name": "model_framework",
            "id": "afc039dd-dc3f-4813-a2a7-08189e34021f",
            "value": "onnx",
        },
        {"name": "target", "id": "17", "value": "default"},
        {"name": "inference_input_type", "id": "19", "value": "float32"},
        {"name": "inference_output_type", "id": "20", "value": "float32"},
        {
            "name": "dataset_percentage",
            "id": "cb13e3df-9f8a-4a6d-a6f9-02c04113cec5",
            "value": 0.25,
        },
        {
            "name": "quantization_aware_training",
            "id": "ad34ad64-e853-44e2-bcce-341f66a1a4de",
            "value": False,
        },
        {
            "name": "use_tf_select_ops",
            "id": "4a49eed1-fe40-4825-a5c7-aadb614b5074",
            "value": False,
        },
        {
            "name": "epochs",
            "id": "fba2ecfa-d1aa-47aa-854e-f091019d7c94",
            "value": 3,
        },
        {
            "name": "batch_size",
            "id": "76ac7e46-fd10-48ef-bcb7-8e4249d0982a",
            "value": 32,
        },
        {
            "name": "optimizer",
            "id": "df71de67-0df0-46e0-abc0-91e7f5a82a10",
            "value": "adam",
        },
        {
            "name": "disable_from_logits",
            "id": "468d8edd-4e23-45d7-a50e-decae4de888c",
            "value": False,
        },
        {
            "name": "compiled_model_path",
            "id": "18",
            "value": "./build/fp32.tflite",
        },
    ],
}


class TestPipelineHandler(HandlerTests):
    dataflow_nodes = [
        PET_DATASET_DATAFLOW_NODE,
        TENSORFLOW_MOBILE_NET_DATAFLOW_NODE,
        TFLITE_RUNTIME_DATAFLOW_NODE,
        TFLITE_COMPILER_DATAFLOW_NODE,
    ]
    dataflow_connections = [
        {"id": "21", "from": "1", "to": "5"},
        {"id": "22", "from": "6", "to": "10"},
        {"id": "23", "from": "7", "to": "15"},
        {"id": "11", "from": "16", "to": "11"},
    ]

    @pytest.fixture(scope="class")
    def handler(self):
        from kenning.pipeline_manager.rpc_handler import FlowHandlerRPC

        handler = PipelineHandler(layout_algorithm="NoLayout")
        handler.get_specification(
            handler.spec_builder.assets_dir,
            actions=FlowHandlerRPC(
                None, None, None, None
            ).get_navbar_actions(),
        )
        return handler

    def equivalence_check(self, dataflow1, dataflow2):
        def compare_blocks(block1, block2):
            if (
                block1["type"] != block2["type"]
                and block1["type"].split(".")[-1]
                != block2["type"].split(".")[-1]
            ):
                return False

            return block1.get("parameters", dict()) == block2.get(
                "parameters", dict()
            )

        if "optimizers" not in dataflow1:
            dataflow1["optimizers"] = []
        if "optimizers" not in dataflow2:
            dataflow2["optimizers"] = []
        if set(dataflow1.keys()) != set(dataflow2.keys()):
            return False

        for key in dataflow1.keys():
            if type(dataflow1[key]) is not type(dataflow2[key]):
                return False
            if isinstance(dataflow1[key], list):
                if len(dataflow1[key]) != len(dataflow2[key]):
                    return False
                for block1, block2 in zip(dataflow1[key], dataflow2[key]):
                    if not compare_blocks(block1, block2):
                        return False
            else:
                if not compare_blocks(dataflow1[key], dataflow2[key]):
                    return False

        return True

    PATH_TO_JSON_SCRIPTS = "./scripts/jsonconfigs"

    test_create_dataflow = factory_test_create_dataflow(PATH_TO_JSON_SCRIPTS)

    test_equivalence = factory_test_equivalence(PATH_TO_JSON_SCRIPTS)

    def test_create_dataflow_fail(self, handler):
        """
        Test if the handler correctly fails when the JSON is invalid.
        """
        with pytest.raises(VisualEditorGraphParserError) as e:
            invalid_flow_json = {
                "test": {
                    "Unknown": "test_parameter",
                },
            }
            handler.create_dataflow(invalid_flow_json)
        assert "is not available in the Visual Editor." in str(e.value)
