# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from contextlib import nullcontext as does_not_raise
from typing import Any, Dict

import pytest

from kenning.interfaces.io_spec_serializer import (
    MAX_LENGTH_ENTRY_FUNC_NAME,
    MAX_LENGTH_MODEL_NAME,
    MAX_MODEL_INPUT_DIM,
    MAX_MODEL_INPUT_NUM,
    MAX_MODEL_OUTPUT_DIM,
    MAX_MODEL_OUTPUT_NUM,
    IOSpecSerializer,
)


@pytest.fixture
def valid_io_spec() -> Dict[str, Any]:
    return copy.deepcopy(DICT_VALID)


DICT_VALID = {
    "input": [],
    "processed_input": [
        {"name": "in1", "shape": [56, 32, 323, 43], "dtype": "float16"},
        {"name": "in2", "shape": [4, 6], "dtype": "int32"},
    ],
    "processed_output": [],
    "output": [
        {
            "name": "out1",
            "shape": [6, 2, 4, 1],
            "dtype": "int64",
        },
        {"name": "out2", "shape": [3443, 4343, 2132, 9], "dtype": "float128"},
        {"name": "out3", "shape": [344], "dtype": "int5"},
    ],
    "entry_func": "test func name",
}

MODEL_NAME_VALID = "test model name"

SERIALIZED_IOSPEC = (
    b"\x02\x00\x00\x00\x04\x00\x00\x00\x02\x00\x00\x00\x38\x00\x00\x00\x20\x00"
    b"\x00\x00\x43\x01\x00\x00\x2b\x00\x00\x00\x04\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x02\x10\x00\x20\x03\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x02\x00\x00\x00\x04\x00"
    b"\x00\x00\x01\x00\x00\x00\x73\x0d\x00\x00\xf7\x10\x00\x00\x54\x08\x00\x00\x09\x00\x00\x00\x58"
    b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x40"
    b"\x02\x80\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x74"
    b"\x65\x73\x74\x20\x66\x75\x6e\x63\x20\x6e\x61\x6d\x65\x00\x00\x00\x00\x00\x00\x74\x65\x73\x74"
    b"\x20\x6d\x6f\x64\x65\x6c\x20\x6e\x61\x6d\x65\x00\x00\x00\x00\x00"
)


class TestIOSpecSerializer:
    def test_serialize_valid_io_spec(self):
        struct = IOSpecSerializer.io_spec_to_struct(
            DICT_VALID, MODEL_NAME_VALID
        )
        assert len(struct) == IOSpecSerializer.compute_iospec_struct_size()
        assert struct == SERIALIZED_IOSPEC

    @pytest.mark.parametrize(
        "dtype,code,size,expectation",
        [
            ("float32", 2, 32, does_not_raise()),
            ("uint8", 1, 8, does_not_raise()),
            ("int16", 0, 16, does_not_raise()),
            ("float", 0, 0, pytest.raises(ValueError)),
            ("float6int", 0, 0, pytest.raises(ValueError)),
            ("float332", 0, 0, pytest.raises(ValueError)),
            ("float31", 2, 31, does_not_raise()),
            ("float0", 0, 0, pytest.raises(ValueError)),
            ("uint9", 1, 9, does_not_raise()),
            ("e8m0fnu_float8", 14, 8, does_not_raise()),
            ("e2m1fn_float4", 17, 4, does_not_raise()),
        ],
    )
    def test_serialize_io_spec_with_different_dtypes(
        self,
        valid_io_spec: Dict[str, Any],
        dtype: str,
        code: int,
        size: int,
        expectation,
    ):
        output_key = "output"
        valid_io_spec[output_key][0]["dtype"] = dtype

        with expectation:
            struct = IOSpecSerializer.io_spec_to_struct(valid_io_spec)
            types = IOSpecSerializer.io_spec_parse_types(
                valid_io_spec[output_key]
            )
            assert len(struct) == IOSpecSerializer.compute_iospec_struct_size()
            assert types[0][0] == code
            assert types[0][1] == size

    @pytest.mark.parametrize(
        "inputs_num,expectation",
        [
            (1, does_not_raise()),
            (MAX_MODEL_INPUT_NUM, does_not_raise()),
            (MAX_MODEL_INPUT_NUM + 1, pytest.raises(ValueError)),
            (MAX_MODEL_INPUT_NUM + 100, pytest.raises(ValueError)),
        ],
    )
    def test_serialize_io_spec_with_different_inputs_num(
        self, valid_io_spec: Dict[str, Any], inputs_num, expectation
    ):
        input_key = (
            "processed_input"
            if "processed_input" in valid_io_spec
            else "input"
        )
        valid_io_spec[input_key] = inputs_num * [valid_io_spec[input_key][0]]

        with expectation:
            struct = IOSpecSerializer.io_spec_to_struct(valid_io_spec)

            assert len(struct) == IOSpecSerializer.compute_iospec_struct_size()

    @pytest.mark.parametrize(
        "input_dim,expectation",
        [
            (1, does_not_raise()),
            (MAX_MODEL_INPUT_DIM, does_not_raise()),
            (MAX_MODEL_INPUT_DIM + 1, pytest.raises(ValueError)),
            (MAX_MODEL_INPUT_DIM + 100, pytest.raises(ValueError)),
        ],
    )
    def test_serialize_io_spec_with_different_input_dim(
        self, valid_io_spec: Dict[str, Any], input_dim, expectation
    ):
        input_key = (
            "processed_input"
            if "processed_input" in valid_io_spec
            else "input"
        )
        dims = (input_dim - len(valid_io_spec[input_key][0]["shape"])) * [
            1,
        ]
        valid_io_spec[input_key][0]["shape"] = (
            *valid_io_spec[input_key][0]["shape"],
            *dims,
        )

        with expectation:
            struct = IOSpecSerializer.io_spec_to_struct(valid_io_spec)

            assert len(struct) == IOSpecSerializer.compute_iospec_struct_size()

    @pytest.mark.parametrize(
        "outputs_num,expectation",
        [
            (1, does_not_raise()),
            (MAX_MODEL_OUTPUT_NUM, does_not_raise()),
            (MAX_MODEL_OUTPUT_NUM + 1, pytest.raises(ValueError)),
            (MAX_MODEL_OUTPUT_NUM + 100, pytest.raises(ValueError)),
        ],
    )
    def test_serialize_io_spec_with_different_output_num(
        self, valid_io_spec: Dict[str, Any], outputs_num, expectation
    ):
        valid_io_spec["output"] = outputs_num * [valid_io_spec["output"][0]]

        with expectation:
            struct = IOSpecSerializer.io_spec_to_struct(valid_io_spec)

            assert len(struct) == IOSpecSerializer.compute_iospec_struct_size()

    @pytest.mark.parametrize(
        "output_dim,expectation",
        [
            (1, does_not_raise()),
            (MAX_MODEL_OUTPUT_DIM, does_not_raise()),
            (MAX_MODEL_OUTPUT_DIM + 1, pytest.raises(ValueError)),
            (MAX_MODEL_OUTPUT_DIM + 100, pytest.raises(ValueError)),
        ],
    )
    def test_serialize_io_spec_with_different_output_dim(
        self, valid_io_spec: Dict[str, Any], output_dim, expectation
    ):
        output_key = "output"
        dims = (output_dim - len(valid_io_spec[output_key][0]["shape"])) * [
            1,
        ]
        valid_io_spec[output_key][0]["shape"] = (
            *valid_io_spec[output_key][0]["shape"],
            *dims,
        )

        with expectation:
            struct = IOSpecSerializer.io_spec_to_struct(valid_io_spec)

            assert len(struct) == IOSpecSerializer.compute_iospec_struct_size()

    @pytest.mark.parametrize(
        "entry_func_name_len,expectation",
        [
            (1, does_not_raise()),
            (MAX_LENGTH_ENTRY_FUNC_NAME, does_not_raise()),
            (MAX_LENGTH_ENTRY_FUNC_NAME + 1, pytest.raises(ValueError)),
            (MAX_LENGTH_ENTRY_FUNC_NAME + 100, pytest.raises(ValueError)),
        ],
    )
    def test_serialize_io_spec_with_different_entry_func(
        self, valid_io_spec: Dict[str, Any], entry_func_name_len, expectation
    ):
        entry_func = "a" * entry_func_name_len
        valid_io_spec["entry_func"] = entry_func
        with expectation:
            struct = IOSpecSerializer.io_spec_to_struct(valid_io_spec)

            assert len(struct) == IOSpecSerializer.compute_iospec_struct_size()

    @pytest.mark.parametrize(
        "model_name_len,expectation",
        [
            (1, does_not_raise()),
            (MAX_LENGTH_MODEL_NAME, does_not_raise()),
            (MAX_LENGTH_MODEL_NAME + 1, pytest.raises(ValueError)),
            (MAX_LENGTH_MODEL_NAME + 100, pytest.raises(ValueError)),
        ],
    )
    def test_serialize_io_spec_with_different_model_name(
        self, valid_io_spec: Dict[str, Any], model_name_len, expectation
    ):
        model_name = "a" * model_name_len

        with expectation:
            struct = IOSpecSerializer.io_spec_to_struct(
                valid_io_spec, model_name=model_name
            )

            assert len(struct) == IOSpecSerializer.compute_iospec_struct_size()
