# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from contextlib import nullcontext as does_not_raise

import jsonschema.exceptions
import pytest

from kenning.core.model import ModelWrapper
from kenning.core.runtime import Runtime
from kenning.runners.modelruntime_runner import ModelRuntimeRunner

MODELRUNTIME_JSON_VALID = {
    'model_wrapper': {
        'type': 'kenning.modelwrappers.classification.tensorflow_pet_dataset.'
                'TensorFlowPetDatasetMobileNetV2',
        'parameters': {
            'model_path': 'kenning:///models/classification/'
                          'tensorflow_pet_dataset_mobilenetv2.h5'
        }
    },
    'dataset': {
        'type': 'kenning.datasets.pet_dataset.PetDataset',
        'parameters':
            {
                'dataset_root': 'build/pet-dataset',
                'download_dataset': False
            }
    },
    'runtime': {
        'type': 'kenning.runtimes.tflite.TFLiteRuntime',
        'parameters':
            {
                'save_model_path': 'kenning:///models/classification/'
                                   'magic_wand.tflite'
            }
    }
}

MODELRUNTIME_JSON_VALID_NO_DATASET = {
    'model_wrapper': {
        'type': 'kenning.modelwrappers.classification.tensorflow_pet_dataset.'
                'TensorFlowPetDatasetMobileNetV2',
        'parameters': {
            'model_path': 'kenning:///models/classification/'
                          'tensorflow_pet_dataset_mobilenetv2.h5'
        }
    },
    'runtime': {
        'type': 'kenning.runtimes.tflite.TFLiteRuntime',
        'parameters': {
            'save_model_path': 'kenning:///models/classification/'
                               'magic_wand.tflite'
        }
    }
}

MODELRUNTIME_JSON_INVALID_NO_RUNTIME = {
    'model_wrapper': {
        'type': 'kenning.modelwrappers.classification.tensorflow_pet_dataset.'
                'TensorFlowPetDatasetMobileNetV2',
        'parameters': {
            'model_path': 'kenning:///models/classification/'
                          'tensorflow_pet_dataset_mobilenetv2.h5'
        }
    },
    'dataset': {
        'type': 'kenning.datasets.pet_dataset.PetDataset',
        'parameters':
            {
                'dataset_root': './build/pet-dataset',
                'download_dataset': False
            }
    },
}

MODELRUNTIME_JSON_INVALID_NO_MODELWRAPPER = {
    'dataset': {
        'type': 'kenning.datasets.pet_dataset.PetDataset',
        'parameters':
            {
                'dataset_root': './build/pet-dataset',
                'download_dataset': False
            }
    },
    'runtime': {
        'type': 'kenning.runtimes.tflite.TFLiteRuntime',
        'parameters':
            {
                'save_model_path': 'kenning:///models/'
                                   'classification/magic_wand.tflite'
            }
    }
}


@pytest.fixture
def valid_argparse_namespace(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(
        runtime=tmp_path / 'dir/runtime.json',
        model_wrapper=tmp_path / 'dir/modelwrapper.json',
        dataset=tmp_path / 'dir/dataset.json'
    )


@pytest.fixture
def invalid_argparse_namespace(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(
        runtime=tmp_path / 'dir/runtime-invalid.json',
        model_wrapper=tmp_path / 'dir/modelwrapper.json',
        dataset=tmp_path / 'dir/dataset.json'
    )


class TestModelRuntimeRunner:

    @pytest.mark.xdist_group(name='use_resources')
    @pytest.mark.parametrize(
        'json_dict,expectation',
        [
            (MODELRUNTIME_JSON_VALID, does_not_raise()),
            (MODELRUNTIME_JSON_VALID_NO_DATASET, does_not_raise()),
        ],
        ids=[
            'valid-scenario',
            'valid-no-dataset',
        ]
    )
    def test_create_object_from_json_scenario_valid(
            self,
            json_dict,
            expectation,
            datasetimages):
        """
        Scenario testing for valid usage of ModelRuntimeRunner.from_json,
        should create a ModelRuntimeRunner instance, with ModelWrapper and
        Runtime objects as attributes
        """
        if 'dataset' in json_dict:
            json_dict['dataset']['parameters']['dataset_root'] = \
                str(datasetimages.path)

        with expectation:
            runner = ModelRuntimeRunner.from_json(
                json_dict,
                inputs_sources={},
                inputs_specs={},
                outputs={})

            assert isinstance(runner, ModelRuntimeRunner)

            assert hasattr(runner, 'model')
            assert hasattr(runner, 'runtime')

            assert isinstance(runner.model, ModelWrapper)
            assert isinstance(runner.runtime, Runtime)

            runner.cleanup()

    @pytest.mark.xdist_group(name='use_resources')
    @pytest.mark.parametrize(
        'json_dict,expectation',
        [
            (MODELRUNTIME_JSON_INVALID_NO_RUNTIME,
             pytest.raises(jsonschema.exceptions.ValidationError)),
            (MODELRUNTIME_JSON_INVALID_NO_MODELWRAPPER,
             pytest.raises(jsonschema.exceptions.ValidationError)),
        ],
        ids=[
            'invalid-no-runtime',
            'invalid-no-modelwrapper',
        ]
    )
    def test_create_object_from_json_should_not_create_and_raise(
            self,
            json_dict,
            expectation):
        """
        Scenario testing for invalid usage of ModelRuntimeRunner.from_json,
        should raise ValidationError exception and not create an object
        """
        runner = None
        with expectation:
            runner = ModelRuntimeRunner.from_json(
                json_dict,
                inputs_sources={},
                inputs_specs={},
                outputs={})

        assert runner is None

    @pytest.mark.xdist_group(name='use_resources')
    def test_create_object_from_argparse_scenario_valid(
            self,
            valid_argparse_namespace,
            mock_configuration_file_contents_modelruntime_runner):
        """
        Scenario testing for invalid usage of ModelRuntimeRunner.from_argparse,
        should raise ValidationError exception
        """
        runner = ModelRuntimeRunner.from_argparse(
            valid_argparse_namespace,
            inputs_sources={},
            inputs_specs={},
            outputs={}
        )

        with does_not_raise():
            assert isinstance(runner, ModelRuntimeRunner)

            assert hasattr(runner, 'model')
            assert hasattr(runner, 'runtime')

            assert isinstance(runner.model, ModelWrapper)
            assert isinstance(runner.runtime, Runtime)

            runner.cleanup()

    @pytest.mark.xdist_group(name='use_resources')
    def test_create_object_from_argparse_scenario_invalid(
            self,
            invalid_argparse_namespace,
            mock_configuration_file_contents_modelruntime_runner):
        """
        Scenario testing for invalid usage of ModelRuntimeRunner.from_argparse,
        should raise ValidationError exception and not create an object
        """
        runner = None
        with pytest.raises(jsonschema.exceptions.ValidationError):
            runner = ModelRuntimeRunner.from_argparse(
                invalid_argparse_namespace,
                inputs_sources={},
                inputs_specs={},
                outputs={}
            )
        assert runner is None
