import argparse
from pathlib import Path
from contextlib import nullcontext as does_not_raise
from typing import Dict, Type

import jsonschema
import pytest
from kenning.utils.resource_manager import ResourceURI

from kenning.runners.modelruntime_runner import ModelRuntimeRunner
from kenning.runtimes.onnx import ONNXRuntime
from kenning.utils.args_manager import (
    get_parsed_json_dict,
    get_parsed_args_dict)


class TestArgsManagerWrapper:

    JSON_SCHEMA_PYTHON_TYPES_IREERUNTIME = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'save_model_path': {
                'real_name': 'model_path',
                'convert-type': Path,
                'type': ['string'],
                'description': 'Path where the model will be uploaded',
                'default': 'model.vmfb'},
            'driver': {
                'real_name': 'driver',
                'description': 'Name of the runtime target',
                'enum': ['cuda', 'local-sync', 'local-task', 'vulkan']},
            'disable_performance_measurements': {
                'real_name': 'disable_performance_measurements',
                'convert-type': bool,
                'type': ['boolean'],
                'description': 'Disable collection and processing of performance metrics', # noqa E501
                'default': False}},
        'required': ['driver']}
    VALID_JSON_DICT_PYTHON_TYPES_IREERUNTIME = {
        'save_model_path': 'build/yolov4.onnx',
        'driver': 'cuda',
        'disable_performance_measurements': True
    }
    VALID_RESULT_PYTHON_TYPES_IREERUNTIME = {
        'disable_performance_measurements': True,
        'driver': 'cuda',
        'model_path': Path('build/yolov4.onnx')}

    JSON_SCHEMA_OBJECT_TYPE_MODELRUNTIME_RUNNER = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'model_wrapper': {
                'real_name': 'model_wrapper',
                'convert-type': object,
                'type': ['object'],
                'description': 'Path to JSON describing the ModelWrapper '
                               'object, following its argument structure'},
            'dataset': {
                'real_name': 'dataset',
                'convert-type': object,
                'type': ['object'],
                'description': 'Path to JSON describing the Dataset object, '
                               'following its argument structure'},
            'runtime': {
                'real_name': 'runtime',
                'convert-type': object,
                'type': ['object'],
                'description': 'Path to JSON describing the Runtime object, '
                               'following its argument structure'}},
        'required': ['model_wrapper', 'runtime']
    }
    VALID_JSON_DICT_OBJECT_TYPE_MODELRUNTIME_RUNNER = {
        'model_wrapper': {
            'type': 'kenning.modelwrappers.detectors.yolov4.ONNXYOLOV4',
            'parameters': {
                'model_path': 'build/yolov4.onnx'}
        },
        'runtime': {
            'type': 'kenning.runtimes.onnx.ONNXRuntime',
            'parameters': {
                'save_model_path': 'build/yolov4.onnx'
            }
        }
    }
    VALID_RESULT_OBJECT_TYPE_MODELRUNTIME_RUNNER = {
        'model_wrapper': {
            'type': 'kenning.modelwrappers.detectors.yolov4.ONNXYOLOV4',
            'parameters': {
                'model_path': 'build/yolov4.onnx'}
        },
        'runtime': {
            'type': 'kenning.runtimes.onnx.ONNXRuntime',
            'parameters': {
                'save_model_path': 'build/yolov4.onnx'} # noqa E501
        }
    }

    INVALID_JSON_DICT_PYTHON_TYPES_IREERUNTIME_MISSING_REQUIRED = {
        'save_model_path': 'build/yolov4.onnx',
        'disable_performance_measurements': True
    }

    INVALID_JSON_DICT_OBJECT_TYPE_MODELRUNTIME_RUNNER_MISSING_REQUIRED = {
        'model_wrapper': {
            'type': 'kenning.modelwrappers.detectors.yolov4.ONNXYOLOV4',
            'parameters': {
                'model_path': 'build/yolov4.onnx'}
        },
    }

    @pytest.mark.fast
    @pytest.mark.parametrize(
        'schema,json_dict,expected_result,expectation',
        [
            (JSON_SCHEMA_PYTHON_TYPES_IREERUNTIME,
             VALID_JSON_DICT_PYTHON_TYPES_IREERUNTIME,
             VALID_RESULT_PYTHON_TYPES_IREERUNTIME,
             does_not_raise()),
            (JSON_SCHEMA_OBJECT_TYPE_MODELRUNTIME_RUNNER,
             VALID_JSON_DICT_OBJECT_TYPE_MODELRUNTIME_RUNNER,
             VALID_RESULT_OBJECT_TYPE_MODELRUNTIME_RUNNER,
             does_not_raise()),

            (JSON_SCHEMA_PYTHON_TYPES_IREERUNTIME,
             INVALID_JSON_DICT_PYTHON_TYPES_IREERUNTIME_MISSING_REQUIRED,
             {},
             pytest.raises(jsonschema.exceptions.ValidationError)),
            (JSON_SCHEMA_OBJECT_TYPE_MODELRUNTIME_RUNNER,
             INVALID_JSON_DICT_OBJECT_TYPE_MODELRUNTIME_RUNNER_MISSING_REQUIRED, # noqa E501
             {},
             pytest.raises(jsonschema.exceptions.ValidationError))
        ],
        ids=[
            'valid_python_types',
            'valid_object_type',
            'invalid_missing_required_value_python_types',
            'invalid_missing_required_value_object_type'
        ])
    def test_get_parsed_json_dicts_and_check_schema_validity(
            self,
            schema: Dict,
            json_dict: Dict,
            expected_result: Dict,
            expectation):
        """
        Tests the get_parsed_json_dict method.

        Things being tested:
        * If the dict is validated with the schema correctly
        * If the returned parsed dict is correct, e.g. ArgManager should add
        missing values, convert parameters to the correct types
        """

        with expectation:
            parsed_json_dict = get_parsed_json_dict(schema, json_dict)
            assert expected_result == parsed_json_dict

    VALID_ARGPARSE_ARGS_PYTHON_TYPES_ONNXRUNTIME = \
        argparse.Namespace(
            save_model_path='build/yolov4.onnx',
            execution_providers=['CPUExecutionProvider'])
    VALID_RESULT_PYTHON_TYPES_ONNXRUNTIME = {
        'disable_performance_measurements': False,
        'execution_providers': ['CPUExecutionProvider'],
        'model_path': ResourceURI('build/yolov4.onnx')}

    INVALID_ARGPARSE_ARGS_PYTHON_TYPES_ONNXRUNTIME_UNDEFINED_ARG_NAME = \
        argparse.Namespace(
            model_path='build/yolov4.onnx',
            execution_providers=['CPUExecutionProvider'])
    VALID_RESULT_PYTHON_TYPES_ONNXRUNTIME_DEFAULT_MODELPATH = {
        'disable_performance_measurements': False,
        'execution_providers': ['CPUExecutionProvider'],
        'model_path': ResourceURI('model.tar')}

    VALID_ARGPARSE_ARGS_OBJECT_TYPE_MODELRUNTIME_RUNNER = \
        argparse.Namespace(
            runtime='runtime.json',
            model_wrapper='modelwrapper.json',
            dataset='dataset.json'
        )
    VALID_RESULT_OBJECT_TYPE_MODELRUNTIME_RUNNER = {
        'model_wrapper': {
            'type': 'kenning.modelwrappers.detectors.yolov4.ONNXYOLOV4',
            'parameters': {
                'model_path': 'kenning:///models/detection/yolov4.onnx'}
        },
        'dataset': {},
        'runtime': {
            'type': 'kenning.runtimes.onnx.ONNXRuntime',
            'parameters': {
                'save_model_path': 'kenning:///models/detection/yolov4.onnx'}
        }
    }

    INVALID_ARGPARSE_ARGS_OBJECT_TYPE_MODELRUNTIME_RUNNER_UNDEF_ARG_NAME = \
        argparse.Namespace(
            runtime='runtime-invalid.json',
            model_wrapper='modelwrapper.json',
            dataset='dataset.json'
        )

    @pytest.mark.fast
    @pytest.mark.usefixtures(
        'mock_configuration_file_contents_modelruntime_runner'
    )
    @pytest.mark.parametrize(
        'class_type,args,expected_result,expectation',
        [
            (ONNXRuntime,
             VALID_ARGPARSE_ARGS_PYTHON_TYPES_ONNXRUNTIME,
             VALID_RESULT_PYTHON_TYPES_ONNXRUNTIME,
             does_not_raise()),
            (ModelRuntimeRunner,
             VALID_ARGPARSE_ARGS_OBJECT_TYPE_MODELRUNTIME_RUNNER,
             VALID_RESULT_OBJECT_TYPE_MODELRUNTIME_RUNNER,
             does_not_raise()),
            (ONNXRuntime,
             INVALID_ARGPARSE_ARGS_PYTHON_TYPES_ONNXRUNTIME_UNDEFINED_ARG_NAME,
             VALID_RESULT_PYTHON_TYPES_ONNXRUNTIME_DEFAULT_MODELPATH,
             does_not_raise()),
            (ModelRuntimeRunner,
             INVALID_ARGPARSE_ARGS_OBJECT_TYPE_MODELRUNTIME_RUNNER_UNDEF_ARG_NAME,  # noqa E501
             VALID_RESULT_OBJECT_TYPE_MODELRUNTIME_RUNNER,
             pytest.raises(AssertionError)),
        ],
        ids=[
            'valid_python_types',
            'valid_object_type',
            'invalid_python_types_undefined_arg',
            'invalid_object_type_missing_required_value'
        ]
    )
    def test_get_parsed_args_dict_and_check_schema_validity(
            self,
            tmp_path,
            class_type: Type,
            args: argparse.Namespace,
            expected_result,
            expectation):
        """
        Tests the get_parsed_args_dict method.

        Things being tested:
        * If the dict is validated with the schema correctly
        * If the returned parsed dict is correct, e.g. ArgManager should add
        missing values, convert parameters to the correct types

        This test also sets paths for the ModelRuntimeRunner, since the JSON
        configuration files should exist
        """

        if args == TestArgsManagerWrapper.\
                VALID_ARGPARSE_ARGS_OBJECT_TYPE_MODELRUNTIME_RUNNER \
                or args == TestArgsManagerWrapper.\
                INVALID_ARGPARSE_ARGS_OBJECT_TYPE_MODELRUNTIME_RUNNER_UNDEF_ARG_NAME: # noqa E501
            args = argparse.Namespace(
                runtime=tmp_path / 'dir/' / args.runtime,
                model_wrapper=tmp_path / 'dir/' / args.model_wrapper,
                dataset=tmp_path / 'dir/' / args.dataset
            )

        with expectation:
            print('test')

            parsed_args_dict = get_parsed_args_dict(class_type, args)
            assert expected_result == parsed_args_dict
