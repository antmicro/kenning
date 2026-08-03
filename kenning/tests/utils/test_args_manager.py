# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from types import NoneType
from typing import Dict, List, Tuple, Type, Union

import jsonschema
import pytest

from kenning.core.exceptions import ArgsManagerConvertError
from kenning.runners.modelruntime_runner import ModelRuntimeRunner
from kenning.runtimes.onnx import ONNXRuntime
from kenning.utils.args_manager import (
    ArgumentsHandler,
    get_parsed_args_dict,
    get_parsed_json_dict,
)
from kenning.utils.resource_manager import ResourceURI


class TestArgsManagerWrapper:
    JSON_SCHEMA_PYTHON_TYPES_IREERUNTIME = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "save_model_path": {
                "real_name": "model_path",
                "convert-type": Path,
                "type": ["string"],
                "description": "Path where the model will be uploaded",
                "default": "model.vmfb",
            },
            "driver": {
                "real_name": "driver",
                "description": "Name of the runtime target",
                "enum": ["cuda", "local-sync", "local-task", "vulkan"],
            },
            "disable_performance_measurements": {
                "real_name": "disable_performance_measurements",
                "convert-type": bool,
                "type": ["boolean"],
                "description": "Disable collection and processing of performance metrics",  # noqa: E501
                "default": False,
            },
        },
        "required": ["driver"],
    }
    VALID_JSON_DICT_PYTHON_TYPES_IREERUNTIME = {
        "save_model_path": "build/yolov4.onnx",
        "driver": "cuda",
        "disable_performance_measurements": True,
    }
    VALID_RESULT_PYTHON_TYPES_IREERUNTIME = {
        "disable_performance_measurements": True,
        "driver": "cuda",
        "model_path": Path("build/yolov4.onnx"),
    }

    JSON_SCHEMA_OBJECT_TYPE_MODELRUNTIME_RUNNER = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "model_wrapper": {
                "real_name": "model_wrapper",
                "convert-type": object,
                "type": ["object"],
                "description": "Path to JSON describing the ModelWrapper "
                "object, following its argument structure",
            },
            "dataset": {
                "real_name": "dataset",
                "convert-type": object,
                "type": ["object"],
                "description": "Path to JSON describing the Dataset object, "
                "following its argument structure",
            },
            "runtime": {
                "real_name": "runtime",
                "convert-type": object,
                "type": ["object"],
                "description": "Path to JSON describing the Runtime object, "
                "following its argument structure",
            },
        },
        "required": ["model_wrapper", "runtime"],
    }
    VALID_JSON_DICT_OBJECT_TYPE_MODELRUNTIME_RUNNER = {
        "model_wrapper": {
            "type": "kenning.modelwrappers.object_detection.yolov4.ONNXYOLOV4",
            "parameters": {"model_path": "build/yolov4.onnx"},
        },
        "runtime": {
            "type": "kenning.runtimes.onnx.ONNXRuntime",
            "parameters": {"save_model_path": "build/yolov4.onnx"},
        },
    }
    VALID_RESULT_OBJECT_TYPE_MODELRUNTIME_RUNNER = {
        "model_wrapper": {
            "type": "kenning.modelwrappers.object_detection.yolov4.ONNXYOLOV4",
            "parameters": {"model_path": "build/yolov4.onnx"},
        },
        "runtime": {
            "type": "kenning.runtimes.onnx.ONNXRuntime",
            "parameters": {"save_model_path": "build/yolov4.onnx"},
        },
    }

    INVALID_JSON_DICT_PYTHON_TYPES_IREERUNTIME_MISSING_REQUIRED = {
        "save_model_path": "build/yolov4.onnx",
        "disable_performance_measurements": True,
    }

    INVALID_JSON_DICT_OBJECT_TYPE_MODELRUNTIME_RUNNER_MISSING_REQUIRED = {
        "model_wrapper": {
            "type": "kenning.modelwrappers.object_detection.yolov4.ONNXYOLOV4",
            "parameters": {"model_path": "build/yolov4.onnx"},
        },
    }

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "schema,json_dict,expected_result,expectation",
        [
            (
                JSON_SCHEMA_PYTHON_TYPES_IREERUNTIME,
                VALID_JSON_DICT_PYTHON_TYPES_IREERUNTIME,
                VALID_RESULT_PYTHON_TYPES_IREERUNTIME,
                does_not_raise(),
            ),
            (
                JSON_SCHEMA_OBJECT_TYPE_MODELRUNTIME_RUNNER,
                VALID_JSON_DICT_OBJECT_TYPE_MODELRUNTIME_RUNNER,
                VALID_RESULT_OBJECT_TYPE_MODELRUNTIME_RUNNER,
                does_not_raise(),
            ),
            (
                JSON_SCHEMA_PYTHON_TYPES_IREERUNTIME,
                INVALID_JSON_DICT_PYTHON_TYPES_IREERUNTIME_MISSING_REQUIRED,
                {},
                pytest.raises(jsonschema.exceptions.ValidationError),
            ),
            (
                JSON_SCHEMA_OBJECT_TYPE_MODELRUNTIME_RUNNER,
                INVALID_JSON_DICT_OBJECT_TYPE_MODELRUNTIME_RUNNER_MISSING_REQUIRED,
                {},
                pytest.raises(jsonschema.exceptions.ValidationError),
            ),
        ],
        ids=[
            "valid_python_types",
            "valid_object_type",
            "invalid_missing_required_value_python_types",
            "invalid_missing_required_value_object_type",
        ],
    )
    def test_get_parsed_json_dicts_and_check_schema_validity(
        self, schema: Dict, json_dict: Dict, expected_result: Dict, expectation
    ):
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

    VALID_ARGPARSE_ARGS_PYTHON_TYPES_ONNXRUNTIME = argparse.Namespace(
        save_model_path="build/yolov4.onnx",
        execution_providers=["CPUExecutionProvider"],
    )
    VALID_RESULT_PYTHON_TYPES_ONNXRUNTIME = {
        "batch_size": 1,
        "disable_performance_measurements": False,
        "execution_providers": ["CPUExecutionProvider"],
        "model_path": ResourceURI("build/yolov4.onnx"),
    }

    INVALID_ARGPARSE_ARGS_PYTHON_TYPES_ONNXRUNTIME_UNDEFINED_ARG_NAME = (
        argparse.Namespace(
            model_path="build/yolov4.onnx",
            execution_providers=["CPUExecutionProvider"],
        )
    )
    VALID_RESULT_PYTHON_TYPES_ONNXRUNTIME_DEFAULT_MODELPATH = {
        "batch_size": 1,
        "disable_performance_measurements": False,
        "execution_providers": ["CPUExecutionProvider"],
        "model_path": ResourceURI("model.tar"),
    }

    VALID_ARGPARSE_ARGS_OBJECT_TYPE_MODELRUNTIME_RUNNER = argparse.Namespace(
        runtime="runtime.json",
        model_wrapper="modelwrapper.json",
        dataset="dataset.json",
    )
    VALID_RESULT_OBJECT_TYPE_MODELRUNTIME_RUNNER = {
        "model_wrapper": {
            "type": "kenning.modelwrappers.object_detection.yolov4.ONNXYOLOV4",
            "parameters": {
                "model_path": "kenning:///models/detection/yolov4.onnx"
            },
        },
        "dataset": {},
        "runtime": {
            "type": "kenning.runtimes.onnx.ONNXRuntime",
            "parameters": {
                "save_model_path": "kenning:///models/detection/yolov4.onnx"
            },
        },
    }

    INVALID_ARGPARSE_ARGS_OBJECT_TYPE_MODELRUNTIME_RUNNER_UNDEF_ARG_NAME = (
        argparse.Namespace(
            runtime="runtime-invalid.json",
            model_wrapper="modelwrapper.json",
            dataset="dataset.json",
        )
    )

    @pytest.mark.fast
    @pytest.mark.usefixtures(
        "mock_configuration_file_contents_modelruntime_runner"
    )
    @pytest.mark.parametrize(
        "class_type,args,expected_result,expectation",
        [
            (
                ONNXRuntime,
                VALID_ARGPARSE_ARGS_PYTHON_TYPES_ONNXRUNTIME,
                VALID_RESULT_PYTHON_TYPES_ONNXRUNTIME,
                does_not_raise(),
            ),
            (
                ModelRuntimeRunner,
                VALID_ARGPARSE_ARGS_OBJECT_TYPE_MODELRUNTIME_RUNNER,
                VALID_RESULT_OBJECT_TYPE_MODELRUNTIME_RUNNER,
                does_not_raise(),
            ),
            (
                ONNXRuntime,
                INVALID_ARGPARSE_ARGS_PYTHON_TYPES_ONNXRUNTIME_UNDEFINED_ARG_NAME,
                VALID_RESULT_PYTHON_TYPES_ONNXRUNTIME_DEFAULT_MODELPATH,
                does_not_raise(),
            ),
            (
                ModelRuntimeRunner,
                INVALID_ARGPARSE_ARGS_OBJECT_TYPE_MODELRUNTIME_RUNNER_UNDEF_ARG_NAME,
                VALID_RESULT_OBJECT_TYPE_MODELRUNTIME_RUNNER,
                pytest.raises(AssertionError),
            ),
        ],
        ids=[
            "valid_python_types",
            "valid_object_type",
            "invalid_python_types_undefined_arg",
            "invalid_object_type_missing_required_value",
        ],
    )
    def test_get_parsed_args_dict_and_check_schema_validity(
        self,
        tmp_path,
        class_type: Type,
        args: argparse.Namespace,
        expected_result,
        expectation,
    ):
        """
        Tests the get_parsed_args_dict method.

        Things being tested:
        * If the dict is validated with the schema correctly
        * If the returned parsed dict is correct, e.g. ArgManager should add
        missing values, convert parameters to the correct types

        This test also sets paths for the ModelRuntimeRunner, since the JSON
        configuration files should exist
        """
        if (
            args
            == TestArgsManagerWrapper.VALID_ARGPARSE_ARGS_OBJECT_TYPE_MODELRUNTIME_RUNNER  # noqa: E501
            or args
            == TestArgsManagerWrapper.INVALID_ARGPARSE_ARGS_OBJECT_TYPE_MODELRUNTIME_RUNNER_UNDEF_ARG_NAME  # noqa: E501
        ):
            args = argparse.Namespace(
                runtime=tmp_path / "dir/" / args.runtime,
                model_wrapper=tmp_path / "dir/" / args.model_wrapper,
                dataset=tmp_path / "dir/" / args.dataset,
            )

        with expectation:
            print("test")

            parsed_args_dict = get_parsed_args_dict(class_type, args)
            assert expected_result == parsed_args_dict

    @pytest.mark.parametrize(
        "value,type,desired_value, expectation",
        [
            (
                (("323", "98.45"), ([1, 5, 1, 3], ["9.5", "1.4"], ["5"])),
                Tuple[
                    Tuple[int, float], Tuple[List[int], List[float], List[int]]
                ],
                ((323, 98.45), ([1, 5, 1, 3], [9.5, 1.4], [5])),
                does_not_raise(),
            ),
            (
                (((((["/dev/null", "/proc"],),), 56),),),
                Tuple[Tuple[Tuple[Tuple[Tuple[List[Path]]], float]]],
                ((((([Path("/dev/null"), Path("/proc")],),), 56.0),),),
                does_not_raise(),
            ),
            (
                ((((([6565, "/proc"],),), 56),),),
                Tuple[Tuple[Tuple[Tuple[Tuple[List[Path]]], float]]],
                None,
                pytest.raises(ValueError),
            ),
            (
                (((((["/dev/null", "/proc"],),), 56),),),
                Tuple[Tuple[Tuple[Tuple[List[Path]]], float]],
                None,
                pytest.raises(ValueError),
            ),
            (
                ((None, None),),
                Tuple[Tuple[NoneType, NoneType]],
                ((None, None),),
                does_not_raise(),
            ),
            (
                # Complex cast cannot change container type (only types of
                # values inside the container).
                [((((["/dev/null", "/proc"],),), 56),)],
                Tuple[Tuple[Tuple[Tuple[Tuple[List[Path]]], float]]],
                [(((([Path("/dev/null"), Path("/proc")],),), 56.0),)],
                does_not_raise(),
            ),
        ],
    )
    def test_cast_complex_type_success(
        self, value, type, desired_value, expectation
    ):
        with expectation:
            assert desired_value == ArgumentsHandler.cast_complex_type(
                value, type
            )

    @pytest.mark.parametrize(
        "value,spec,desired_value, expectation",
        [
            (
                (("323", "98.45"), ([1, 5, 1, 3], ["9.5", "1.4"], ["5"])),
                {
                    "type": Union[
                        str,
                        Tuple[
                            Tuple[int, float],
                            Tuple[List[int], List[float], List[int]],
                        ],
                    ],
                    "enum": None,
                    "nullable": True,
                },
                ((323, 98.45), ([1, 5, 1, 3], [9.5, 1.4], [5])),
                does_not_raise(),
            ),
            (
                (((((["/dev/null", "/proc"],),), 56),),),
                {
                    "type": Tuple[
                        Tuple[Tuple[Tuple[Tuple[List[Path]]], float]]
                    ],
                    "enum": [
                        ((((([Path("/dev/null"), Path("/proc")],),), 56.0),),),
                        "lorem",
                    ],
                },
                ((((([Path("/dev/null"), Path("/proc")],),), 56.0),),),
                does_not_raise(),
            ),
            (
                ((((([6565, "/proc"],),), 56),),),
                {
                    "type": Tuple[
                        Tuple[Tuple[Tuple[Tuple[List[Path]]], float]]
                    ],
                    "enum": None,
                },
                None,
                pytest.raises(ArgsManagerConvertError),
            ),
            (
                (((((["/dev/null", "/proc"],),), 56),),),
                {
                    "type": Tuple[Tuple[Tuple[Tuple[List[Path]]], float]],
                    "enum": None,
                },
                None,
                pytest.raises(ArgsManagerConvertError),
            ),
            (
                ((None, None),),
                {
                    "type": Tuple[Tuple[NoneType, NoneType]],
                    "enum": None,
                },
                ((None, None),),
                does_not_raise(),
            ),
            (
                # Complex cast cannot change container type (only types of
                # values inside the container).
                [((((["/dev/null", "/proc"],),), 56),)],
                {
                    "type": Union[
                        Tuple[Path, float],
                        Tuple[Tuple[Tuple[Tuple[Tuple[List[Path]]], float]]],
                    ],
                    "enum": None,
                },
                [(((([Path("/dev/null"), Path("/proc")],),), 56.0),)],
                does_not_raise(),
            ),
            (
                ["a", "b"],
                {
                    "type": str,
                    "enum": ["a", "b", "c", "d", "e"],
                },
                ["a", "b"],
                does_not_raise(),
            ),
            (
                ["a", "b"],
                {
                    "type": str,
                    "enum": ["a", "c", "d", "e"],
                },
                None,
                pytest.raises(ArgsManagerConvertError),
            ),
            (
                ["a", "b"],
                {
                    "type": float,
                    "enum": ["a", "b", "c", "d", "e"],
                },
                None,
                pytest.raises(ArgsManagerConvertError),
            ),
        ],
    )
    def test_convert_value(self, value, spec, desired_value, expectation):
        with expectation:
            assert desired_value == ArgumentsHandler.convert_value(value, spec)

    @pytest.mark.parametrize(
        "value,spec,result",
        [
            (
                (("323", "98.45"), ([1, 5, 1, 3], ["9.5", "1.4"], ["5"])),
                {
                    "type": Union[
                        str,
                        Tuple[
                            Tuple[int, float],
                            Tuple[List[int], List[float], List[int]],
                        ],
                    ],
                    "enum": None,
                    "nullable": True,
                },
                True,
            ),
            (
                (((((["/dev/null", "/proc"],),), 56),),),
                {
                    "type": Tuple[
                        Tuple[Tuple[Tuple[Tuple[List[Path]]], float]]
                    ],
                    "enum": [
                        ((((([Path("/dev/null"), Path("/proc")],),), 56.0),),),
                        "lorem",
                    ],
                },
                True,
            ),
            (
                ((((([6565, "/proc"],),), 56),),),
                {
                    "type": Tuple[
                        Tuple[Tuple[Tuple[Tuple[List[Path]]], float]]
                    ],
                    "enum": None,
                },
                False,
            ),
            (
                (((((["/dev/null", "/proc"],),), 56),),),
                {
                    "type": Tuple[Tuple[Tuple[Tuple[List[Path]]], float]],
                    "enum": None,
                },
                False,
            ),
            (
                ((None, None),),
                {
                    "type": Tuple[Tuple[NoneType, NoneType]],
                    "enum": None,
                },
                True,
            ),
            (
                # Complex cast cannot change container type (only types of
                # values inside the container).
                [((((["/dev/null", "/proc"],),), 56),)],
                {
                    "type": Union[
                        Tuple[Path, float],
                        Tuple[Tuple[Tuple[Tuple[Tuple[List[Path]]], float]]],
                    ],
                    "enum": None,
                },
                True,
            ),
            (
                ["a", "b"],
                {
                    "type": str,
                    "enum": ["a", "b", "c", "d", "e"],
                },
                True,
            ),
            (
                ["a", "b"],
                {
                    "type": List[str],
                    "enum": ["a", "b", "c", "d", "e"],
                },
                True,
            ),
            (
                ["a", "b"],
                {
                    "type": str,
                    "enum": ["a", "c", "d", "e"],
                },
                False,
            ),
            (
                ["a", "b"],
                {
                    "type": float,
                    "enum": ["a", "b", "c", "d", "e"],
                },
                False,
            ),
            (
                [2, 8],
                {
                    "type": List[float],
                    "enum": None,
                    "list_range": (1, 3),
                    "item_range": (-1, 9),
                },
                True,
            ),
            (
                [2, 10],
                {
                    "type": List[float],
                    "enum": None,
                    "list_range": (1, 3),
                    "item_range": (-1, 9),
                },
                False,
            ),
            (
                [2, 8, 2, 4, 4],
                {
                    "type": List[int],
                    "enum": None,
                    "list_range": (1, 3),
                    "item_range": (-1, 9),
                },
                False,
            ),
        ],
    )
    def test_verify_type(self, value, spec, result):
        assert result == ArgumentsHandler.verify_type(value, spec)
