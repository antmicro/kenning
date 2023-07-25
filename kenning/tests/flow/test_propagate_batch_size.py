from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest

from kenning.compilers.iree import IREECompiler
from kenning.compilers.tflite import TFLiteCompiler
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
def test_scenario_pet_database_tflite(
        batch_sizes,
        expectation,
        tmpfolder,
        datasetimages_parametrized):

    for batch_size in batch_sizes:
        with expectation:
            dataset = PetDataset(
                root=datasetimages_parametrized(350).path,
                download_dataset=False,
                batch_size=batch_size,
                standardize=False
            )

            tmp_model_path = tmpfolder / f'model-{batch_size}.tflite'

            model = TensorFlowPetDatasetMobileNetV2(
                modelpath='./kenning/resources/models/classification/'
                          'tensorflow_pet_dataset_mobilenetv2.h5',
                dataset=dataset
            )

            model.save_io_specification(model.modelpath)
            compiler = TFLiteCompiler(
                dataset=dataset,
                compiled_model_path=tmp_model_path,
                modelframework='keras',
                target='default',
                inferenceinputtype='float32',
                inferenceoutputtype='float32'
            )
            compiler.compile(
                inputmodelpath='./kenning/resources/models/classification/'
                               'tensorflow_pet_dataset_mobilenetv2.h5'
            )

            runtime = TFLiteRuntime(
                protocol=None,
                modelpath=tmp_model_path
            )

            print(f'batch {batch_size} path {tmp_model_path} '
                  f'rt modelpath {runtime.modelpath}')

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
def test_scenario_tflite_magic_wand(
        batch_sizes,
        expectation,
        tmpfolder,
        datasetimages_parametrized):

    with expectation:

        for batch_size in batch_sizes:

            compiled_model_path = tmpfolder / f'model-{batch_size}.tflite'

            dataset = MagicWandDataset(
                root=tmpfolder,
                batch_size=batch_size,
                download_dataset=True
            )
            model = MagicWandModelWrapper(
                modelpath='kenning/resources/models/classification/'
                          'magic_wand.h5',
                dataset=dataset,
                from_file=True
            )
            compiler = TFLiteCompiler(
                dataset=dataset,
                compiled_model_path=compiled_model_path,
                modelframework='keras',
                target='default',
                inferenceinputtype='float32',
                inferenceoutputtype='float32'
            )
            compiler.compile(
                inputmodelpath='kenning/resources/models/classification/'
                               'magic_wand.h5'
            )

            runtime = TFLiteRuntime(
                protocol=None,
                modelpath=compiled_model_path
            )
            runtime.run_locally(
                dataset,
                model,
                compiled_model_path
            )

            assert compiler.dataset.batch_size == batch_size

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
        tmpfolder,
        datasetimages_parametrized
):
    with expectation:
        for batch_size in batch_sizes:
            compiled_model_path = tmpfolder / f'model-{batch_size}.vmfb'

            dataset = VisualWakeWordsDataset(
                root=tmpfolder,
                batch_size=batch_size,
                download_dataset=True
            )
            model = PersonDetectionModelWrapper(
                modelpath='kenning/resources/models/classification/'
                          'person_detect.tflite',
                dataset=dataset
            )
            model.save_io_specification(model.modelpath)

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
                inputmodelpath='kenning/resources/models/classification/'
                               'person_detect.tflite',
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
