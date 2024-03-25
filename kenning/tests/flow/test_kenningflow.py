# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext as does_not_raise
from copy import deepcopy
from typing import Dict, Final

import numpy as np
import pytest

from kenning.core.flow import KenningFlow
from kenning.dataproviders.camera_dataprovider import CameraDataProvider
from kenning.interfaces.io_interface import IOCompatibilityError
from kenning.outputcollectors.real_time_visualizers import (
    BaseRealTimeVisualizer,
)
from kenning.runners.modelruntime_runner import ModelRuntimeRunner

FLOW_STEPS: Final = 4

# base camera data provider runner
CAMERA_DATA_PROVIDER_NCHW_JSON = {
    "type": "kenning.dataproviders.camera_dataprovider.CameraDataProvider",
    "parameters": {
        "video_file_path": "/dev/video0",
        "input_memory_layout": "NCHW",
        "input_width": 608,
        "input_height": 608,
    },
    "outputs": {"frame": "cam_frame"},
}

# base YOLOv4 (detection model) runner
MDL_RT_RUNNER_YOLOV4_JSON = {
    "type": "kenning.runners.modelruntime_runner.ModelRuntimeRunner",
    "parameters": {
        "model_wrapper": {
            "type": "kenning.modelwrappers.object_detection.yolov4.ONNXYOLOV4",
            "parameters": {
                "model_path": "kenning:///models/detection/yolov4.onnx"
            },
        },
        "runtime": {
            "type": "kenning.runtimes.onnx.ONNXRuntime",
            "parameters": {
                "save_model_path": "kenning:///models/detection/yolov4.onnx",
                "execution_providers": ["CPUExecutionProvider"],
            },
        },
    },
    "inputs": {"input": "cam_frame"},
    "outputs": {"detection_output": "predictions"},
}

# base YOLACT (segmentation model) runner
MDL_RT_RUNNER_YOLACT_JSON = {
    "type": "kenning.runners.modelruntime_runner.ModelRuntimeRunner",
    "parameters": {
        "dataset": {
            "type": "kenning.datasets.coco_dataset.COCODataset2017",
            "parameters": {
                "dataset_root": "./build/COCODataset2017",
                "download_dataset": False,
            },
        },
        "model_wrapper": {
            "type": "kenning.modelwrappers.instance_segmentation.yolact.YOLACT",  # noqa: E501
            "parameters": {
                "model_path": "kenning:///models/instance_segmentation/yolact.onnx"  # noqa: E501
            },
        },
        "runtime": {
            "type": "kenning.runtimes.onnx.ONNXRuntime",
            "parameters": {
                "save_model_path": "kenning:///models/instance_segmentation/yolact.onnx",  # noqa: E501
                "execution_providers": ["CPUExecutionProvider"],
            },
        },
    },
    "inputs": {"input": "cam_frame"},
    "outputs": {"segmentation_output": "predictions"},
}

# base detection visualizer runner
DECT_VISUALIZER_JSON = {
    "type": "kenning.outputcollectors.detection_visualizer.DetectionVisualizer",  # noqa: E501
    "parameters": {
        "output_width": 608,
        "output_height": 608,
        "save_to_file": True,
        "save_path": str(pytest.test_directory / "out_1.mp4"),
    },
    "inputs": {"frame": "cam_frame", "detection_data": "predictions"},
}

# base real time detection visualizer runner
RT_DECT_VISUALIZER_JSON = {
    "type": "kenning.outputcollectors.real_time_visualizers.RealTimeDetectionVisualizer",  # noqa: E501
    "parameters": {
        "viewer_width": 512,
        "viewer_height": 512,
        "input_memory_layout": "NCHW",
        "input_color_format": "BGR",
    },
    "inputs": {"frame": "cam_frame", "detection_data": "predictions"},
}

# base real time segmentation visualizer runner
RT_SEGM_VISUALIZER_JSON = {
    "type": "kenning.outputcollectors.real_time_visualizers.RealTimeSegmentationVisualizer",  # noqa: E501
    "parameters": {
        "viewer_width": 512,
        "viewer_height": 512,
        "score_threshold": 0.4,
        "input_memory_layout": "NCHW",
        "input_color_format": "BGR",
    },
    "inputs": {"frame": "cam_frame", "segmentation_data": "predictions"},
}

# valid scenario with camera, detection model and detection visualizer
FLOW_SCENARIO_DETECTION = [
    CAMERA_DATA_PROVIDER_NCHW_JSON,
    MDL_RT_RUNNER_YOLOV4_JSON,
    DECT_VISUALIZER_JSON,
]
# valid scenario with camera, detection model and real time detection
# visualizer
FLOW_SCENARIO_RT_DETECTION = [
    CAMERA_DATA_PROVIDER_NCHW_JSON,
    MDL_RT_RUNNER_YOLOV4_JSON,
    RT_DECT_VISUALIZER_JSON,
]
# valid scenario with camera, segmentation model and real time segmentation
# visualizer. For this model we need to change camera output height and width
CAMERA_DATA_PROVIDER_NCHW_SEGM_JSON = deepcopy(CAMERA_DATA_PROVIDER_NCHW_JSON)
CAMERA_DATA_PROVIDER_NCHW_SEGM_JSON["parameters"]["input_width"] = 550
CAMERA_DATA_PROVIDER_NCHW_SEGM_JSON["parameters"]["input_height"] = 550

FLOW_SCENARIO_RT_SEGMENTATION = [
    CAMERA_DATA_PROVIDER_NCHW_SEGM_JSON,
    MDL_RT_RUNNER_YOLACT_JSON,
    RT_SEGM_VISUALIZER_JSON,
]
# a complex scenario with two detection models and three detection visualizers.
# To make it work, we need to change names of variables used by models to
# prevent redefinition - we change second model output to 'predictions_2' and
# names of visualizers outputs to 'out_2.mp4' and 'out_3.mp4' respectively. We
# also change one visualizer to use second model output
MDL_RT_RUNNER_YOLOV4_2_JSON = deepcopy(MDL_RT_RUNNER_YOLOV4_JSON)
MDL_RT_RUNNER_YOLOV4_2_JSON["outputs"]["detection_output"] = "predictions_2"

DECT_VISUALIZER_2_JSON = deepcopy(DECT_VISUALIZER_JSON)
DECT_VISUALIZER_2_JSON["parameters"]["save_path"] = str(
    pytest.test_directory / "out_2.mp4"
)

DECT_VISUALIZER_3_JSON = deepcopy(DECT_VISUALIZER_JSON)
DECT_VISUALIZER_3_JSON["inputs"]["detection_data"] = "predictions_2"
DECT_VISUALIZER_3_JSON["parameters"]["save_path"] = str(
    pytest.test_directory / "out_3.mp4"
)

FLOW_SCENARIO_COMPLEX = [
    CAMERA_DATA_PROVIDER_NCHW_JSON,
    MDL_RT_RUNNER_YOLOV4_JSON,
    MDL_RT_RUNNER_YOLOV4_2_JSON,
    DECT_VISUALIZER_JSON,
    DECT_VISUALIZER_2_JSON,
    DECT_VISUALIZER_3_JSON,
]

# detection scenario is indeed valid so we will use it as valid case
FLOW_SCENARIO_VALID = FLOW_SCENARIO_DETECTION

# to prepare scenario with redefined variable, we change model output name to
# same as camera provider output - 'cam_frame'
MDL_RT_RUNNER_YOLOV4_REDEF_VAR_JSON = deepcopy(MDL_RT_RUNNER_YOLOV4_JSON)
MDL_RT_RUNNER_YOLOV4_REDEF_VAR_JSON["outputs"][
    "detection_output"
] = "cam_frame"

FLOW_SCENARIO_REDEF_VARIABLE = [
    CAMERA_DATA_PROVIDER_NCHW_JSON,
    MDL_RT_RUNNER_YOLOV4_REDEF_VAR_JSON,
    DECT_VISUALIZER_JSON,
]

# to prepare scenario with undefined variable, we simply change model input
# name to 'undefined_cam_frame'
MDL_RT_RUNNER_YOLOV4_UNDEF_VAR_JSON = deepcopy(MDL_RT_RUNNER_YOLOV4_JSON)
MDL_RT_RUNNER_YOLOV4_UNDEF_VAR_JSON["inputs"]["input"] = "undefined_cam_frame"

FLOW_SCENARIO_UNDEF_VARIABLE = [
    CAMERA_DATA_PROVIDER_NCHW_JSON,
    MDL_RT_RUNNER_YOLOV4_UNDEF_VAR_JSON,
    DECT_VISUALIZER_JSON,
]

# to prepare scenario with incompatible IO, we change camera memory layout from
# 'NCHW' to 'NHWC' which changes its shape and makes incompatible with model
# input
CAMERA_DATA_PROVIDER_NHWC_JSON = deepcopy(CAMERA_DATA_PROVIDER_NCHW_JSON)
CAMERA_DATA_PROVIDER_NHWC_JSON["parameters"]["input_memory_layout"] = "NHWC"

FLOW_SCENARIO_INCOMPATIBLE_IO = [
    CAMERA_DATA_PROVIDER_NHWC_JSON,
    MDL_RT_RUNNER_YOLOV4_JSON,
    DECT_VISUALIZER_JSON,
]


@pytest.fixture(autouse=True)
def mock_camera_fetch_input():
    """
    Mocks camera input - instead of camera frame returns random noise.
    """

    def fetch_input(self):
        return np.random.randint(
            low=0, high=255, size=(256, 256, 3), dtype=np.uint8
        )

    CameraDataProvider.fetch_input = fetch_input
    CameraDataProvider.prepare = lambda self: None
    CameraDataProvider.detach_from_source = lambda self: None


@pytest.fixture
def set_should_close_after_3_calls():
    """
    Mocks should_close method so that after 3 calls it returns True.
    """

    def should_close(self):
        should_close.calls += 1
        return should_close.calls >= 3

    should_close.calls = 0

    ModelRuntimeRunner.should_close = should_close


@pytest.fixture
def mock_dear_py_gui():
    """
    Mocks DearPyGui so that there is no GUI being showed.
    """

    def _gui_thread(self):
        while not self.stop:
            _ = self.process_data.get()

    BaseRealTimeVisualizer._gui_thread = _gui_thread
    BaseRealTimeVisualizer.should_close = lambda self: False


class TestKenningFlowScenarios:
    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "json_scenario,expectation",
        [
            (FLOW_SCENARIO_VALID, does_not_raise()),
            (FLOW_SCENARIO_REDEF_VARIABLE, pytest.raises(Exception)),
            (FLOW_SCENARIO_UNDEF_VARIABLE, pytest.raises(Exception)),
            (
                FLOW_SCENARIO_INCOMPATIBLE_IO,
                pytest.raises(IOCompatibilityError),
            ),
        ],
        ids=[
            "valid_scenario",
            "redefined_variable",
            "undefined_variable",
            "incompatible_IO",
        ],
    )
    def test_load_kenning_flows(self, json_scenario: Dict, expectation):
        """
        Tests KenningFlow loading from JSON and runner's IO validation.
        """
        with expectation:
            flow = KenningFlow.from_json(json_scenario)

            flow.cleanup()

    @pytest.mark.usefixtures("mock_dear_py_gui")
    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "json_scenario",
        [
            FLOW_SCENARIO_RT_DETECTION,
            FLOW_SCENARIO_RT_SEGMENTATION,
            FLOW_SCENARIO_DETECTION,
            FLOW_SCENARIO_COMPLEX,
        ],
        ids=[
            "realtime_detection_scenario",
            "realtime_segmentation_scenario",
            "detection_scenario",
            "complex_scenario",
        ],
    )
    def test_run_kenning_flows(self, json_scenario: Dict):
        """
        Tests execution of example flows.
        """
        flow = KenningFlow.from_json(json_scenario)

        try:
            for _ in range(FLOW_STEPS):
                flow.init_state()
                flow.run_single_step()
        except Exception as e:
            pytest.fail(f"Error during flow run: {e}")
        finally:
            flow.cleanup()

    @pytest.mark.usefixtures("set_should_close_after_3_calls")
    @pytest.mark.xdist_group(name="use_resources")
    def test_kenning_flow_close_when_runner_should_close(self):
        """
        Tests closing flow when some runner got exit indicator.
        """
        flow = KenningFlow.from_json(FLOW_SCENARIO_VALID)
        flow.run()
