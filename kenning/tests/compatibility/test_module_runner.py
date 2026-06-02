# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from typing import Type

import pytest

from kenning.core.automl import AutoML
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.core.report import Report
from kenning.core.runner import Runner
from kenning.datasets.anomaly_detection_dataset import AnomalyDetectionDataset
from kenning.datasets.tabular_dataset import TabularDataset
from kenning.dispatcher.block_config import (
    BLOCK_CONFIGURATIONS_KEY,
    UNAFFILIATED_PARAMETERS_KEY,
)
from kenning.protocols.ros2 import ROS2Protocol
from kenning.protocols.uart import UARTProtocol
from kenning.scenarios.module_runner import ModuleRunner
from kenning.utils.class_loader import (
    get_all_subclasses,
    get_base_classes_dict,
    load_class,
)


def mark_xfail_if_abstract(cls_name: Type):
    module_cls = load_class(cls_name)
    if inspect.isabstract(module_cls):
        pytest.xfail(f"Abstract class: {cls_name}")


all_classes = sorted(
    [
        f"{module_path}.{module_cls}"
        for module_name, (path, cls) in get_base_classes_dict().items()
        for module_cls, module_path in get_all_subclasses(
            path, cls, import_classes=False
        )
    ]
)

skipped = []

xfails = [
    # Compiler
    "Ai8xCompiler",
    # DataConverter
    "ModelWrapperDataConverter",
    "ROS2DataConverter",
    # ModelWrapper
    "Ai8xAnomalyDetectionCNN",
    "ONNXYOLOV4",
    "TVMDarknetCOCOYOLOV3",
    "TensorFlowPetDatasetMobileNetV2",
    "YOLACTWrapper",
    # Optimizer
    "NNIPruningOptimizer",
    # RuntimeBuilder
    "ZephyrRuntimeBuilder",
    # InferenceLoop
    # there may be an error in the implementation
    # with referencing a `None` attribute
    # in `self._platform.sensors_frequency`
    "AnomalyDetectionInferenceLoop",
    "SensorRealtimeInferenceLoop",
    # Protocols
    "ROS2Protocol",
]

skipped = tuple(map(load_class, skipped))
xfails = tuple(map(load_class, xfails))


def mark_xfail_or_skip(module_cls: Type):
    """
    Mark the ``cls`` as xfail or skip if needed.
    Otherwise, nothing happens.
    """
    cls_name = module_cls.__class__.__name__
    if not (
        hasattr(module_cls, "form_argparse")
        and callable(getattr(module_cls, "form_argparse"))
    ):
        pytest.xfail(f"class {cls_name} has no form_argparse method")

    if inspect.isabstract(module_cls):
        pytest.xfail(f"Abstract class failed: {cls_name}")

    if issubclass(module_cls, (Runner)):
        pytest.xfail("Runner tests are unsupported")
    # TODO: Currently unsupported. Have
    # complex logic in __init__ that makes it hard to test.
    if issubclass(module_cls, xfails):
        pytest.xfail(f"Unsupported test: {cls_name}")
    if issubclass(module_cls, skipped):
        pytest.skip(f"class {cls_name} is explicitly skipped.")


@pytest.mark.slow
class TestModuleRunnerCompatibility:
    @pytest.mark.parametrize("module_cls_name", all_classes)
    def test_matrix(self, module_cls_name: str, tmp_path):
        module_cls = load_class(module_cls_name)
        mark_xfail_or_skip(module_cls)

        argv = [
            "kenning-pytest",
            "--module",
            module_cls_name,
            "--no-wait",
            "True",
        ]

        tmp_path_str = str(tmp_path)
        dict_extensions = {
            ModelWrapper: {"model_path": tmp_path_str},
            Dataset: {
                "dataset_root": tmp_path_str,
                # Save disk space
                "no_download_dataset": True,
                "no_prepare_dataset": True,
            },
            Optimizer: {"compiled_model_path": tmp_path_str},
            Report: {"measurements": tmp_path_str},
            UARTProtocol: {
                "port": "/dev/devnull",
            },
            ROS2Protocol: {
                "process_action_type_str": tmp_path_str,
                "process_action_name": "ModelRuntimeRunner",
            },
            AutoML: {
                "output_directory": tmp_path_str,
                "skip_model_size_check": True,
            },
            load_class("ModelInserter"): {
                "input_model_path": tmp_path_str,
                "model_framework": "dummy",
            },
            AnomalyDetectionDataset: {
                "csv_file": "kenning:///datasets/anomaly_detection/minispot.csv",
            },
            TabularDataset: {
                "dataset_path": tmp_path_str,
                "cols_x": "col1,col2,col3",
                "col_y": "colY",
            },
        }

        config_dict = {
            BLOCK_CONFIGURATIONS_KEY: {},
            UNAFFILIATED_PARAMETERS_KEY: {},
        }

        # Look for the base class
        for superclass, exts in dict_extensions.items():
            if issubclass(module_cls, superclass):
                config_dict["free_flags"].update(exts)

        argv.extend(["--config", json.dumps(config_dict)])

        ModuleRunner.scenario_run(argv)
