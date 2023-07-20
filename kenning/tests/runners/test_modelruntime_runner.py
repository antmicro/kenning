from contextlib import nullcontext as does_not_raise

import jsonschema.exceptions
import pytest

from kenning.runners.modelruntime_runner import ModelRuntimeRunner

MODELRUNTIME_JSON_VALID = {
    'model_wrapper': {
        'type': 'kenning.modelwrappers.classification.tensorflow_pet_dataset.'
                'TensorFlowPetDatasetMobileNetV2',
        'parameters': {
            'model_path': 'kenning/resources/models/classification/'
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
                'save_model_path': './kenning/resources/models/classification/'
                                   'magic_wand.tflite'
            }
    }
}

MODELRUNTIME_JSON_VALID_NO_DATASET = {
    'model_wrapper': {
        'type': 'kenning.modelwrappers.classification.tensorflow_pet_dataset.'
                'TensorFlowPetDatasetMobileNetV2',
        'parameters': {
            'model_path': './kenning/resources/models/classification/'
                          'tensorflow_pet_dataset_mobilenetv2.h5'
        }
    },
    'runtime': {
        'type': 'kenning.runtimes.tflite.TFLiteRuntime',
        'parameters': {
            'save_model_path': './kenning/resources/models/classification/'
                               'magic_wand.tflite'
        }
    }
}

MODELRUNTIME_JSON_INVALID_NO_RUNTIME = {
    'model_wrapper': {
        'type': 'kenning.modelwrappers.classification.tensorflow_pet_dataset.'
                'TensorFlowPetDatasetMobileNetV2',
        'parameters': {
            'model_path': './kenning/resources/models/classification/'
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
                'save_model_path': './kenning/resources/models/'
                                   'classification/magic_wand.tflite'
            }
    }
}


class TestModelRuntimeRunner:

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
    def test_create_object_from_json(
            self,
            json_dict,
            expectation,
            datasetimages):
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

            runner.cleanup()

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
    def test_create_object_should_not_create_and_raise(
            self,
            json_dict,
            expectation):
        runner = None
        with expectation:
            runner = ModelRuntimeRunner.from_json(
                json_dict,
                inputs_sources={},
                inputs_specs={},
                outputs={})

        assert runner is None
