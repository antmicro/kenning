# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest

from kenning.core.dataset import Dataset
from kenning.dataconverters.modelwrapper_dataconverter import (
    ModelWrapperDataConverter,
)
from kenning.datasets.magic_wand_dataset import MagicWandDataset
from kenning.datasets.pet_dataset import PetDataset
from kenning.datasets.visual_wake_words_dataset import VisualWakeWordsDataset
from kenning.modelwrappers.classification.tensorflow_pet_dataset import (
    TensorFlowPetDatasetMobileNetV2,
)
from kenning.modelwrappers.classification.tflite_magic_wand import (
    MagicWandModelWrapper,
)
from kenning.modelwrappers.classification.tflite_person_detection import (
    PersonDetectionModelWrapper,
)
from kenning.optimizers.iree import IREECompiler
from kenning.optimizers.tflite import TFLiteCompiler
from kenning.optimizers.tvm import TVMCompiler
from kenning.runtimes.iree import IREERuntime
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.runtimes.tvm import TVMRuntime
from kenning.tests.core.conftest import get_reduced_dataset_path
from kenning.utils.pipeline_runner import PipelineRunner
from kenning.utils.resource_manager import ResourceURI


def truncate_test_fraction(dataset: Dataset, batch_size: int):
    """
    Set size of dataset's test fraction to match used batch size.

    Parameters
    ----------
    dataset : Dataset
        Dataset object where fraction will be set
    batch_size : int
        Size of the batch
    """
    dataset.split_fraction_test = (
        int(len(dataset.dataX) * dataset.split_fraction_test)
        // batch_size
        * batch_size
        / len(dataset.dataX)
    )


@pytest.mark.parametrize(
    "batch_size,expectation",
    [
        (1, does_not_raise()),
        (16, does_not_raise()),
        (32, does_not_raise()),
        (-1, pytest.raises(AssertionError)),
    ],
)
def test_scenario_pet_dataset_tflite(batch_size, expectation, tmpfolder):
    with expectation:
        dataset = PetDataset(
            root=get_reduced_dataset_path(PetDataset),
            download_dataset=False,
            batch_size=batch_size,
            standardize=False,
        )
        truncate_test_fraction(dataset, batch_size)

        tmp_model_path = tmpfolder / f"model-{batch_size}.tflite"

        model_path = ResourceURI(
            TensorFlowPetDatasetMobileNetV2.pretrained_model_uri
        )

        model = TensorFlowPetDatasetMobileNetV2(
            model_path=model_path, dataset=dataset
        )

        model.save_io_specification(model.model_path)
        compiler = TFLiteCompiler(
            dataset=dataset,
            compiled_model_path=tmp_model_path,
            model_framework="keras",
            target="default",
            inferenceinputtype="float32",
            inferenceoutputtype="float32",
        )
        compiler.init()
        compiler.compile(input_model_path=Path(model_path))

        runtime = TFLiteRuntime(model_path=tmp_model_path)

        dataconverter = ModelWrapperDataConverter(model)

        pipeline_runner = PipelineRunner(
            dataset=dataset,
            dataconverter=dataconverter,
            optimizers=[compiler],
            runtime=runtime,
            model_wrapper=model,
        )

        pipeline_runner.run()

        assert compiler.dataset.batch_size == batch_size

        for model_input_spec in model.io_specification["input"]:
            assert model_input_spec["shape"][0] == batch_size

        for runtime_input_spec in runtime.input_spec:
            assert runtime_input_spec["shape"][0] == batch_size


@pytest.mark.parametrize(
    "batch_size,expectation",
    [
        (1, does_not_raise()),
        (24, does_not_raise()),
        (48, does_not_raise()),
        (64, does_not_raise()),
        (0, pytest.raises(AssertionError)),
        (-1, pytest.raises(AssertionError)),
    ],
)
def test_scenario_tflite_tvm_magic_wand(batch_size, expectation, tmpfolder):
    with expectation:
        compiled_model_path_tflite = (
            tmpfolder / f"model-magicwand-{batch_size}.tflite"
        )
        compiled_model_path_tvm = (
            tmpfolder / f"model-magicwand-{batch_size}.tar"
        )

        dataset = MagicWandDataset(
            root=get_reduced_dataset_path(MagicWandDataset),
            batch_size=batch_size,
            download_dataset=False,
        )
        truncate_test_fraction(dataset, batch_size)

        model_path = ResourceURI(MagicWandModelWrapper.pretrained_model_uri)

        model = MagicWandModelWrapper(
            model_path=model_path, dataset=dataset, from_file=True
        )
        model.save_io_specification(model.model_path)
        compiler_tflite = TFLiteCompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path_tflite,
            model_framework="keras",
            target="default",
            inferenceinputtype="float32",
            inferenceoutputtype="float32",
        )
        compiler_tflite.init()
        compiler_tflite.compile(input_model_path=Path(model.model_path))

        compiler_tvm = TVMCompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path_tvm,
            model_framework="tflite",
        )
        compiler_tvm.init()
        compiler_tvm.compile(input_model_path=Path(compiled_model_path_tflite))

        runtime = TVMRuntime(model_path=compiled_model_path_tvm)

        dataconverter = ModelWrapperDataConverter(model)

        pipeline_runner = PipelineRunner(
            dataset=dataset,
            dataconverter=dataconverter,
            optimizers=[compiler_tvm],
            runtime=runtime,
            model_wrapper=model,
        )

        pipeline_runner.run()

        assert compiler_tflite.dataset.batch_size == batch_size
        assert compiler_tvm.dataset.batch_size == batch_size

        for model_input_spec in model.io_specification["processed_input"]:
            assert model_input_spec["shape"][0] == batch_size

        for runtime_input_spec in runtime.processed_input_spec:
            assert runtime_input_spec["shape"][0] == batch_size


@pytest.mark.slow
@pytest.mark.parametrize(
    "batch_size,expectation",
    [
        (1, does_not_raise()),
        (16, does_not_raise()),
        (32, does_not_raise()),
        (0, pytest.raises(AssertionError)),
        (-1, pytest.raises(AssertionError)),
    ],
)
def test_scenario_tflite_person_detection(batch_size, expectation, tmpfolder):
    with expectation:
        compiled_model_path = tmpfolder / f"model-{batch_size}.vmfb"

        dataset = VisualWakeWordsDataset(
            root=get_reduced_dataset_path(VisualWakeWordsDataset),
            batch_size=batch_size,
            download_dataset=False,
        )
        truncate_test_fraction(dataset, batch_size)

        model_path = ResourceURI(
            PersonDetectionModelWrapper.pretrained_model_uri
        )

        model = PersonDetectionModelWrapper(
            model_path=model_path, dataset=dataset
        )
        model.save_io_specification(model.model_path)

        compiler = IREECompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            backend="llvm-cpu",
            model_framework="tflite",
            compiler_args=[
                "iree-llvm-debug-symbols=false",
                "iree-vm-bytecode-module-strip-source-map=true",
                "iree-vm-emit-polyglot-zip=false",
                "iree-llvm-target-triple=riscv32-pc-linux-elf",
                "iree-llvm-target-cpu=generic-rv32",
                "iree-llvm-target-cpu-features=+m,+f,+zvl512b,+zve32x,+zve32f",
                "iree-llvm-target-abi=ilp32",
            ],
        )
        compiler.init()
        compiler.compile(
            input_model_path=Path(model_path),
        )
        runtime = IREERuntime(model_path=compiled_model_path)

        runtime.prepare_local()

        assert compiler.dataset.batch_size == batch_size

        for model_input_spec in model.io_specification["processed_input"]:
            assert model_input_spec["shape"][0] == batch_size

        for runtime_input_spec in runtime.processed_input_spec:
            assert runtime_input_spec["shape"][0] == batch_size
