from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest

from kenning.optimizers.iree import IREECompiler
from kenning.optimizers.tflite import TFLiteCompiler
from kenning.optimizers.tvm import TVMCompiler
from kenning.core.measurements import MeasurementsCollector
from kenning.datasets.magic_wand_dataset import MagicWandDataset
from kenning.datasets.pet_dataset import PetDataset
from kenning.datasets.visual_wake_words_dataset import VisualWakeWordsDataset
from kenning.modelwrappers.classification.tensorflow_pet_dataset import \
    TensorFlowPetDatasetMobileNetV2
from kenning.modelwrappers.classification.tflite_magic_wand import \
    MagicWandModelWrapper
from kenning.modelwrappers.classification.tflite_person_detection import \
    PersonDetectionModelWrapper
from kenning.runtimes.renode import RenodeRuntime
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.runtimes.tvm import TVMRuntime
from kenning.tests.core.conftest import get_reduced_dataset_path
from kenning.utils.resource_manager import ResourceURI, ResourceManager


@pytest.mark.parametrize(
    'batch_sizes,expectation',
    [
        ([1, 16, 32], does_not_raise()),
        ([-1], pytest.raises(AssertionError))
    ],
    ids=[
        'batch_sizes_valid',
        'batch_sizes_invalid'
    ]
)
def test_scenario_pet_dataset_tflite(
        batch_sizes,
        expectation,
        tmpfolder):

    for batch_size in batch_sizes:
        with expectation:
            dataset = PetDataset(
                root=get_reduced_dataset_path(PetDataset),
                download_dataset=False,
                batch_size=batch_size,
                standardize=False
            )

            tmp_model_path = tmpfolder / f'model-{batch_size}.tflite'

            model_path = ResourceURI(TensorFlowPetDatasetMobileNetV2.
                                     pretrained_model_uri)

            model = TensorFlowPetDatasetMobileNetV2(
                model_path=model_path,
                dataset=dataset
            )

            model.save_io_specification(model.model_path)
            compiler = TFLiteCompiler(
                dataset=dataset,
                compiled_model_path=tmp_model_path,
                modelframework='keras',
                target='default',
                inferenceinputtype='float32',
                inferenceoutputtype='float32'
            )
            compiler.compile(
                input_model_path=Path(model_path)
            )

            runtime = TFLiteRuntime(
                protocol=None,
                model_path=tmp_model_path
            )

            runtime.run_locally(
                dataset,
                model,
                tmp_model_path
            )

            assert compiler.dataset.batch_size == batch_size

            for model_input_spec in model.io_specification['input']:
                assert model_input_spec['shape'][0] == batch_size

            for runtime_input_spec in runtime.input_spec:
                assert runtime_input_spec['shape'][0] == batch_size


@pytest.mark.parametrize(
    'batch_sizes,expectation',
    [
        ([1, 24, 48, 64], does_not_raise()),
        ([0, -1], pytest.raises(AssertionError))
    ],
    ids=[
        'batch_sizes_valid',
        'batch_sizes_invalid'
    ]
)
def test_scenario_tflite_tvm_magic_wand(
        batch_sizes,
        expectation,
        tmpfolder):

    with expectation:

        for batch_size in batch_sizes:

            ResourceManager().clear_cache()

            compiled_model_path_tflite = tmpfolder / f'model-magicwand-{batch_size}.tflite' # noqa E501
            compiled_model_path_tvm = tmpfolder / f'model-magicwand-{batch_size}.tar' # noqa E501

            dataset = MagicWandDataset(
                root=get_reduced_dataset_path(MagicWandDataset),
                batch_size=batch_size,
                download_dataset=False
            )

            model_path = ResourceURI(MagicWandModelWrapper.
                                     pretrained_model_uri)

            model = MagicWandModelWrapper(
                model_path=model_path,
                dataset=dataset,
                from_file=True
            )
            model.save_io_specification(model.model_path)
            compiler_tflite = TFLiteCompiler(
                dataset=dataset,
                compiled_model_path=compiled_model_path_tflite,
                modelframework='keras',
                target='default',
                inferenceinputtype='float32',
                inferenceoutputtype='float32'
            )
            compiler_tflite.compile(
                input_model_path=Path(model.model_path)
            )

            compiler_tvm = TVMCompiler(
                dataset=dataset,
                compiled_model_path=compiled_model_path_tvm,
                modelframework='tflite'
            )
            compiler_tvm.compile(
                input_model_path=Path(compiled_model_path_tflite)
            )

            runtime = TVMRuntime(
                protocol=None,
                model_path=compiled_model_path_tvm
            )

            runtime.run_locally(
                dataset,
                model,
                compiled_model_path_tvm
            )

            MeasurementsCollector.clear()

            assert compiler_tflite.dataset.batch_size == batch_size
            assert compiler_tvm.dataset.batch_size == batch_size

            for model_input_spec in model.io_specification['input']:
                assert model_input_spec['shape'][0] == batch_size

            for runtime_input_spec in runtime.input_spec:
                assert runtime_input_spec['shape'][0] == batch_size


@pytest.mark.slow
@pytest.mark.parametrize(
    'batch_sizes,expectation',
    [
        ([1, 16, 32], does_not_raise()),
        ([0, -1], pytest.raises(AssertionError))
    ],
    ids=[
        'batch_sizes_valid',
        'batch_sizes_invalid'
    ]
)
def test_scenario_tflite_person_detection(
        batch_sizes,
        expectation,
        tmpfolder):
    with expectation:
        for batch_size in batch_sizes:
            compiled_model_path = tmpfolder / f'model-{batch_size}.vmfb'

            dataset = VisualWakeWordsDataset(
                root=get_reduced_dataset_path(VisualWakeWordsDataset),
                batch_size=batch_size,
                download_dataset=False
            )

            model_path = ResourceURI(PersonDetectionModelWrapper.
                                     pretrained_model_uri)

            model = PersonDetectionModelWrapper(
                model_path=model_path,
                dataset=dataset
            )
            model.save_io_specification(model.model_path)

            compiler = IREECompiler(
                dataset=dataset,
                compiled_model_path=compiled_model_path,
                backend='llvm-cpu',
                modelframework='tflite',
                compiler_args=[
                    "iree-llvm-debug-symbols=false",
                    "iree-vm-bytecode-module-strip-source-map=true",
                    "iree-vm-emit-polyglot-zip=false",
                    "iree-llvm-target-triple=riscv32-pc-linux-elf",
                    "iree-llvm-target-cpu=generic-rv32",
                    "iree-llvm-target-cpu-features=+m,+f,+zvl512b,+zve32x,+zve32f", # noqa E501
                    "iree-llvm-target-abi=ilp32"]
            )
            compiler.compile(
                input_model_path=Path(model_path),
            )

            runtime = RenodeRuntime(
                protocol=None,
                runtime_binary_path=Path('/renode-resources/springbok/'
                                         'iree_runtime'),
                platform_resc_path=Path('/renode-resources/springbok/'
                                        'springbok.resc')
            )

            assert compiler.dataset.batch_size == batch_size

            for model_input_spec in model.io_specification['input']:
                assert model_input_spec['shape'][0] == batch_size

            runtime.read_io_specification(model.io_specification)
            for runtime_input_spec in runtime.input_spec:
                assert runtime_input_spec['shape'][0] == batch_size
