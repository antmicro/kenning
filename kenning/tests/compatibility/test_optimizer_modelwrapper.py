# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

import pytest
from schema import Type

from kenning.utils.logger import KLogger

# DO NOT REMOVE
# removing this import will result in SegFault when running all compatibility
# test, caused by the tests involving tinygrad and onnx changing some global
# state. It has the outcome of pyarrow segfaulting when pytest ends.
try:
    import tinygrad.frontend.onnx  # noqa: F401
except ImportError:
    KLogger.warning(
        "Could not import onnx frontend for tinygrad, this may influence the "
        "outcome of this test suite. Try installing the correct version of "
        "tinygrad with onnx frontend enabled."
    )

from itertools import product

from kenning.converters import converter_registry
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import EXT_TO_FRAMEWORK, Optimizer
from kenning.core.platform import Platform
from kenning.modelwrappers.llm.llm import LLM
from kenning.modelwrappers.object_detection.yolov4 import ONNXYOLOV4
from kenning.optimizers.gptq import GPTQOptimizer
from kenning.optimizers.gptq_sparsegpt import GPTQSparseGPTOptimizer
from kenning.optimizers.model_inserter import ModelInserter
from kenning.optimizers.nni_pruning import NNIPruningOptimizer
from kenning.tests.conftest import (
    Samples,
    get_tmp_path,
)
from kenning.tests.core.conftest import (
    get_dataset_random_mock,
    remove_file_or_dir,
)
from kenning.tests.core.test_model import create_model
from kenning.utils.class_loader import get_all_subclasses
from kenning.utils.pipeline_runner import PipelineRunner

MODELWRAPPER_SUBCLASSES = get_all_subclasses(
    "kenning.modelwrappers", ModelWrapper, raise_exception=True
)
OPTIMIZER_SUBCLASSES = get_all_subclasses(
    "kenning.optimizers", Optimizer, raise_exception=True
)

LLM_OPTIMIZERS = [
    optimizer
    for optimizer in OPTIMIZER_SUBCLASSES
    if getattr(optimizer, "inputtypes", []) == {"safetensors-native"}
]

LLM_MODELWRAPPERS = get_all_subclasses(
    "kenning.modelwrappers.llm", LLM, raise_exception=True
)

NON_LLM_MODELWRAPPERS = set(MODELWRAPPER_SUBCLASSES) - set(LLM_MODELWRAPPERS)
NON_LLM_OPTIMIZERS = set(OPTIMIZER_SUBCLASSES) - set(LLM_OPTIMIZERS)

EXPECTED_FAIL = [
    # Skipping because these tests will fail if there is no dataset provided
    ("Ai8xAnomalyDetectionCnn", "ExecuTorchOptimizer"),
    ("Ai8xAnomalyDetectionCnn", "TensorFlowClusteringOptimizer"),
    ("Ai8xAnomalyDetectionCnn", "TensorFlowPruningOptimizer"),
    ("Dinov2ONNX", "Ai8xCompiler"),
    ("Dinov2ONNX", "ExecuTorchOptimizer"),
    ("Dinov2ONNX", "NNIPruningOptimizer"),
    ("Dinov2ONNX", "TFLiteCompiler"),
    ("Dinov2ONNX", "TVMCompiler"),
    ("Dinov2ONNX", "TensorFlowClusteringOptimizer"),
    ("Dinov2ONNX", "TensorFlowPruningOptimizer"),
    ("Llama", "AWQOptimizer"),
    ("Llama", "GPTQOptimizer"),
    ("Llama", "GPTQSparseGPTOptimizer"),
    ("MMPoseDetectionInput", "Ai8xCompiler"),
    ("MMPoseDetectionInput", "ExecuTorchOptimizer"),
    ("MMPoseDetectionInput", "IREECompiler"),
    ("MMPoseDetectionInput", "NNIPruningOptimizer"),
    ("MMPoseDetectionInput", "TFLiteCompiler"),
    ("MMPoseDetectionInput", "TVMCompiler"),
    ("MMPoseDetectionInput", "TensorFlowClusteringOptimizer"),
    ("MMPoseDetectionInput", "TensorFlowPruningOptimizer"),
    ("MMPoseONNX", "Ai8xCompiler"),
    ("MMPoseONNX", "ExecuTorchOptimizer"),
    ("MMPoseONNX", "IREECompiler"),
    ("MMPoseONNX", "NNIPruningOptimizer"),
    ("MMPoseONNX", "TFLiteCompiler"),
    ("MMPoseONNX", "TVMCompiler"),
    ("MMPoseONNX", "TensorFlowClusteringOptimizer"),
    ("MMPoseONNX", "TensorFlowPruningOptimizer"),
    ("MMPoseModelWrapper", "Ai8xCompiler"),
    ("MMPoseModelWrapper", "ExecuTorchOptimizer"),
    ("MMPoseModelWrapper", "IREECompiler"),
    ("MMPoseModelWrapper", "NNIPruningOptimizer"),
    ("MMPoseModelWrapper", "TFLiteCompiler"),
    ("MMPoseModelWrapper", "TVMCompiler"),
    ("MMPoseModelWrapper", "TensorFlowClusteringOptimizer"),
    ("MMPoseModelWrapper", "TensorFlowPruningOptimizer"),
    ("MMPoseRTMOONNX", "Ai8xCompiler"),
    ("MMPoseRTMOONNX", "ExecuTorchOptimizer"),
    ("MMPoseRTMOONNX", "IREECompiler"),
    ("MMPoseRTMOONNX", "NNIPruningOptimizer"),
    ("MMPoseRTMOONNX", "TFLiteCompiler"),
    ("MMPoseRTMOONNX", "TVMCompiler"),
    ("MMPoseRTMOONNX", "TensorFlowClusteringOptimizer"),
    ("MMPoseRTMOONNX", "TensorFlowPruningOptimizer"),
    ("MagicWandModelWrapper", "NNIPruningOptimizer"),
    ("MagicWandModelWrapper", "TFLiteCompiler"),
    ("MagicWandModelWrapper", "TVMCompiler"),
    ("MagicWandModelWrapper", "Ai8xCompiler"),
    ("ONNXYOLOV4", "Ai8xCompiler"),
    ("ONNXYOLOV4", "ExecuTorchOptimizer"),
    ("ONNXYOLOV4", "TensorFlowClusteringOptimizer"),
    ("ONNXYOLOV4", "TensorFlowPruningOptimizer"),
    ("PHI2", "AWQOptimizer"),
    ("PHI2", "GPTQOptimizer"),
    ("PHI2", "GPTQSparseGPTOptimizer"),
    ("PersonDetectionModelWrapper", "IREECompiler"),
    ("PersonDetectionModelWrapper", "Ai8xCompiler"),
    ("PersonDetectionModelWrapper", "ExecuTorchOptimizer"),
    ("PersonDetectionModelWrapper", "TFLiteCompiler"),
    ("PersonDetectionModelWrapper", "NNIPruningOptimizer"),
    ("PersonDetectionModelWrapper", "TensorFlowClusteringOptimizer"),
    ("PersonDetectionModelWrapper", "TensorFlowPruningOptimizer"),
    ("PyTorchAnomalyDetectionVAE", "Ai8xCompiler"),
    ("PyTorchAnomalyDetectionVAE", "NNIPruningOptimizer"),
    ("PyTorchAnomalyDetectionVAE", "TensorFlowClusteringOptimizer"),
    ("PyTorchAnomalyDetectionVAE", "TensorFlowPruningOptimizer"),
    ("PyTorchCOCOMaskRCNN", "Ai8xCompiler"),
    ("PyTorchCOCOMaskRCNN", "ExecuTorchOptimizer"),
    ("PyTorchCOCOMaskRCNN", "NNIPruningOptimizer"),
    ("PyTorchCOCOMaskRCNN", "TFLiteCompiler"),
    ("PyTorchCOCOMaskRCNN", "TVMCompiler"),
    ("PyTorchCOCOMaskRCNN", "IREECompiler"),
    ("PyTorchCOCOMaskRCNN", "TensorFlowClusteringOptimizer"),
    ("PyTorchCOCOMaskRCNN", "TensorFlowPruningOptimizer"),
    ("PyTorchGenericAutoencoderClassification", "NNIPruningOptimizer"),
    (
        "PyTorchGenericAutoencoderClassification",
        "TensorFlowClusteringOptimizer",
    ),
    ("PyTorchGenericAutoencoderClassification", "TensorFlowPruningOptimizer"),
    ("PyTorchGenericClassification", "Ai8xCompiler"),
    ("PyTorchGenericClassification", "TensorFlowClusteringOptimizer"),
    ("PyTorchGenericClassification", "TensorFlowPruningOptimizer"),
    ("PyTorchMagicWandModelWrapper", "NNIPruningOptimizer"),
    ("PyTorchMagicWandModelWrapper", "Ai8xCompiler"),
    ("PyTorchPetDatasetMobileNetV2", "Ai8xCompiler"),
    ("PyTorchPetDatasetMobileNetV2", "TensorFlowClusteringOptimizer"),
    ("PyTorchPetDatasetMobileNetV2", "TensorFlowPruningOptimizer"),
    ("TensorFlowImageNet", "Ai8xCompiler"),
    ("TensorFlowImageNet", "ExecuTorchOptimizer"),
    ("TensorFlowImageNet", "NNIPruningOptimizer"),
    ("TensorFlowImageNet", "TensorFlowClusteringOptimizer"),
    ("TensorFlowImageNet", "TensorFlowPruningOptimizer"),
    ("TensorFlowImageNet", "TensorFlowClusteringOptimizer"),
    ("TensorFlowImageNet", "TensorFlowPruningOptimizer"),
    ("TensorFlowPetDatasetMobileNetV2", "Ai8xCompiler"),
    ("TensorFlowPetDatasetMobileNetV2", "ExecuTorchOptimizer"),
    ("TensorFlowPetDatasetMobileNetV2", "NNIPruningOptimizer"),
    ("TinygradImageNet", "Ai8xCompiler"),
    ("TinygradImageNet", "ExecuTorchOptimizer"),
    ("TinygradImageNet", "IREECompiler"),
    ("TinygradImageNet", "NNIPruningOptimizer"),
    ("TinygradImageNet", "ONNXCompiler"),
    ("TinygradImageNet", "TFLiteCompiler"),
    ("TinygradImageNet", "TVMCompiler"),
    ("TinygradImageNet", "TensorFlowClusteringOptimizer"),
    ("TinygradImageNet", "TensorFlowPruningOptimizer"),
    ("YOLACT", "Ai8xCompiler"),
    ("YOLACT", "ExecuTorchOptimizer"),
    ("YOLACT", "NNIPruningOptimizer"),
    ("YOLACT", "TensorFlowClusteringOptimizer"),
    ("YOLACT", "TensorFlowPruningOptimizer"),
    ("YOLACTWithPostprocessing", "Ai8xCompiler"),
    ("YOLACTWithPostprocessing", "ExecuTorchOptimizer"),
    ("YOLACTWithPostprocessing", "IREECompiler"),
    ("YOLACTWithPostprocessing", "NNIPruningOptimizer"),
    ("YOLACTWithPostprocessing", "TFLiteCompiler"),
    ("YOLACTWithPostprocessing", "TVMCompiler"),
    ("YOLACTWithPostprocessing", "TensorFlowClusteringOptimizer"),
    ("YOLACTWithPostprocessing", "TensorFlowPruningOptimizer"),
    ("TVMDarknetCOCOYOLOV3", "TVMCompiler"),
    ("TVMDarknetCOCOYOLOV3", "TensorFlowClusteringOptimizer"),
    ("TVMDarknetCOCOYOLOV3", "TensorFlowPruningOptimizer"),
    # doesn't work for unknown causes
    ("SKLearnGenericDecisionTreeClassifier", "Ai8xCompiler"),
    ("SKLearnGenericDecisionTreeClassifier", "ExecuTorchOptimizer"),
    ("SKLearnGenericDecisionTreeClassifier", "IREECompiler"),
    ("SKLearnGenericDecisionTreeClassifier", "NNIPruningOptimizer"),
    ("SKLearnGenericDecisionTreeClassifier", "TFLiteCompiler"),
    ("SKLearnGenericDecisionTreeClassifier", "TVMCompiler"),
    ("SKLearnGenericDecisionTreeClassifier", "TensorFlowClusteringOptimizer"),
    ("SKLearnGenericDecisionTreeClassifier", "TensorFlowPruningOptimizer"),
    # some strange error with executorch
]

SKIP = (
    [
        ("MistralInstruct", "AWQOptimizer"),
        ("MistralInstruct", "GPTQOptimizer"),
        ("MistralInstruct", "GPTQSparseGPTOptimizer"),
        # For now we don't have support for emlearn format
        ("StubEmlearnModel", "Ai8xCompiler"),
        ("StubEmlearnModel", "ExecuTorchOptimizer"),
        ("StubEmlearnModel", "IREECompiler"),
        ("StubEmlearnModel", "NNIPruningOptimizer"),
        ("StubEmlearnModel", "ONNXCompiler"),
        ("StubEmlearnModel", "TFLiteCompiler"),
        ("StubEmlearnModel", "TVMCompiler"),
        ("StubEmlearnModel", "TinygradOptimizer"),
        ("StubEmlearnModel", "TensorFlowClusteringOptimizer"),
        ("StubEmlearnModel", "TensorFlowPruningOptimizer"),
    ]
    # Skip LLM specific optimizers for non-LLMs.
    + [
        (model.__name__, optimizer.__name__)
        for model, optimizer in product(NON_LLM_MODELWRAPPERS, LLM_OPTIMIZERS)
    ]
    # Skip non-LLM specific optimizers for LLMs.
    + [
        (model.__name__, optimizer.__name__)
        for model, optimizer in product(LLM_MODELWRAPPERS, NON_LLM_OPTIMIZERS)
    ]
)

expect_fail = pytest.mark.xfail(reason="Expected incompatible")
skip = pytest.mark.skip(reason="Time or resource intensive")


def prepare_objects(
    model_cls: Type[ModelWrapper],
    optimizer_cls: Type[Optimizer],
) -> Tuple[ModelWrapper, Optimizer, Optional[Platform]]:
    if optimizer_cls is ModelInserter:
        pytest.skip("ModelInserter is not supported")

    model_type = model_cls.get_framework()
    optimizer_type = optimizer_cls.get_framework()
    if not converter_registry.find_all_paths(model_type, optimizer_type):
        pytest.xfail("No available conversion path")

    # by default, do not enforce platforms
    platform = None
    compiled_model_path_suffix = ""

    # AI8X is specific to a limited set of platforms, hence
    # platform definition below
    if "Ai8x" in model_cls.__name__ or "Ai8x" in optimizer_cls.__name__:
        platform = Platform("max78002evkit/max78002/m4")
        platform.read_data_from_platforms_yaml()
        compiled_model_path_suffix = ".bin"

    dataset_cls = model_cls.default_dataset
    try:
        dataset = get_dataset_random_mock(dataset_cls, model_cls)
    except NotImplementedError:
        dataset = None
    model = create_model(model_cls, dataset)
    if platform is not None:
        model.read_platform(platform)
    if model_cls.pretrained_model_uri is not None:
        model.prepare_model()
    else:
        model.model_path = model.model_path.with_suffix(
            next(  # Get suffix for chosen model_type
                filter(
                    lambda suf_type: suf_type[1] == model_type,
                    EXT_TO_FRAMEWORK.items(),
                ),
                ("", ""),
            )[0]
        )
        model.prepare_model()
        if model_type == "onnx":
            model.save_to_onnx(model.model_path)
        else:
            model.save_model(model.model_path)
    model.save_io_specification(model.model_path)

    kwargs = {}
    if optimizer_cls not in (GPTQOptimizer, GPTQSparseGPTOptimizer):
        kwargs["model_framework"] = model_type

    if optimizer_cls is NNIPruningOptimizer:
        kwargs["finetuning_epochs"] = 0
        if model_cls is ONNXYOLOV4:
            kwargs["criterion"] = "kenning.utils.yolov4_loss.YOLOv4Loss"
            kwargs["confidence"] = 2

    optimizer = optimizer_cls(
        model.dataset,
        get_tmp_path(compiled_model_path_suffix),
        model_wrapper=model,
        **kwargs,
    )
    return model, optimizer, platform


@pytest.mark.slow
class TestOptimizerModelWrapper:
    @pytest.mark.parametrize(
        "optimizername",
        [
            ("TFLiteCompiler_keras"),
            ("TVMCompiler_keras"),
            ("TVMCompiler_torch"),
        ],
    )
    def test_compile_existence_models(
        self,
        tmpfolder: Path,
        optimizername: str,
        modelsamples: Samples,
        optimizersamples: Samples,
        modelwrappersamples: Samples,
    ):
        """
        Tests compilation process for models presented in Kenning docs.

        List of methods that are being tested
        --------------------------------
        Optimizer.compile()

        Used fixtures
        -------------
        tmpfolder - to get a folder where compiled model will be placed.
        modelsamples - to get paths for models to compile.
        optimizersamples - to get optimizer instances.
        modelwrappersamples - to get inputshape and data type.
        """
        optimizer = optimizersamples.get(optimizername)
        model_path, wrapper_name = modelsamples.get(optimizer.inputtype)
        wrapper = modelwrappersamples.get(wrapper_name)
        model_type = wrapper.get_framework()
        assert isinstance(model_type, str) and len(model_type) > 0
        assert model_type in optimizer.get_input_formats()
        assert model_type in wrapper.get_output_formats()

        filepath = tmpfolder / uuid.uuid4().hex
        io_specs = wrapper.get_io_specification()

        model_path = Path(model_path)
        optimizer.set_compiled_model_path(filepath)
        optimizer.init()
        optimizer.compile(model_path, io_specs)
        assert os.path.exists(filepath)
        os.remove(filepath)

    def test_onnx_model_optimization(
        self,
        tmpfolder: Path,
        modelwrappersamples: Samples,
        optimizersamples: Samples,
    ):
        """
        Tests saving model to onnx format with modelwrappers
        and converting it using optimizers.

        List of methods that are being tested
        --------------------------------
        ModelWrapper.save_to_onnx()

        Used fixtures
        -------------
        tmpfolder - to get a folder where compiled model will be placed
        optimizersamples - to get optimizers instances.
        modelwrappersamples - to get modelwrappers instances.
        """
        import copy

        for wrapper_name in modelwrappersamples:
            wrapper = modelwrappersamples.get(wrapper_name)
            original_io_spec = wrapper.get_io_specification()

            filename = uuid.uuid4().hex
            filepath = tmpfolder / filename
            wrapper.save_to_onnx(filepath)
            assert os.path.exists(filepath)

            for optimizer_name in optimizersamples:
                optimizer = optimizersamples.get(optimizer_name)
                io_spec = copy.deepcopy(original_io_spec)
                # TODO: In future there might be no shared model types,
                # so method may throw an exception
                model_type = wrapper.get_framework()
                optimizer_type = optimizer.get_framework()
                if not converter_registry.find_all_paths(
                    optimizer_type, model_type
                ):
                    continue
                assert isinstance(model_type, str)
                assert isinstance(model_type, str)

                optimizer.set_input_type("onnx")
                compiled_model_path = filename + "_" + optimizer.inputtype
                compiled_model_path = tmpfolder / compiled_model_path
                optimizer.set_compiled_model_path(compiled_model_path)
                optimizer.init()
                optimizer.compile(filepath, io_spec)

                assert os.path.exists(compiled_model_path)
                os.remove(compiled_model_path)
            os.remove(filepath)

    @pytest.mark.compat_matrix(ModelWrapper, Optimizer)
    @pytest.mark.parametrize(
        "model_cls,optimizer_cls",
        [
            pytest.param(cls1, cls2, marks=[expect_fail])
            if (cls1.__name__, cls2.__name__) in EXPECTED_FAIL
            else pytest.param(cls1, cls2, marks=[skip])
            if (cls1.__name__, cls2.__name__) in SKIP
            else (cls1, cls2)
            for cls1 in MODELWRAPPER_SUBCLASSES
            for cls2 in OPTIMIZER_SUBCLASSES
        ],
    )
    def test_matrix(
        self,
        model_cls: Type[ModelWrapper],
        optimizer_cls: Type[Optimizer],
    ):
        model, optimizer, platform = prepare_objects(model_cls, optimizer_cls)
        if model_cls.__name__ == "PHI2":
            if optimizer_cls.__name__ == "GPTQSparseGPTOptimizer":
                pytest.xfail("Running this test is currently not supported")

        try:
            pipeline_runner = PipelineRunner(
                dataset=model.dataset,
                optimizers=[optimizer],
                model_wrapper=model,
                platform=platform,
            )
            pipeline_runner.run(run_benchmarks=False)
            assert optimizer.compiled_model_path.exists()
        finally:
            remove_file_or_dir(optimizer.compiled_model_path)
            remove_file_or_dir(model.model_path)
