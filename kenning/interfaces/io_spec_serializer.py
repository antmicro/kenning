# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Serializer for model input and output specificatios, that
are being sent over UART.
"""

import math
import re
from collections import namedtuple
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

# Maximum number of input and output tensors in the model
MAX_MODEL_INPUT_NUM = 2
MAX_MODEL_OUTPUT_NUM = 12
# Maximum number of dimensions for input and output tensors
MAX_MODEL_INPUT_DIM = 4
MAX_MODEL_OUTPUT_DIM = 4
# Maximum lengths of strings designating
# the model's name and is entry point
MAX_LENGTH_ENTRY_FUNC_NAME = 20
MAX_LENGTH_MODEL_NAME = 20

# Length of C data types, that are in the IOSPEC struct
STANDARD_INTEGER_SIZE = 4  # uint32_t
STANDARD_CHARACTER_SIZE = 1  # uint8_t (char)
STANDARD_TYPE_SIZE = 2  # packed struct of 2x uint8_t


class IOSpecStructFieldType(Enum):
    """
    Enum with valid data types for the IO spec struct.
    """

    STANDARD_INTEGER = 0, STANDARD_INTEGER_SIZE  # C uint32_t
    STANDARD_CHARACTER = 1, STANDARD_CHARACTER_SIZE  # C char/uint8_t
    STANDARD_TYPE = 2, STANDARD_TYPE_SIZE  # packed C struct of 2x uint8_t

    def __new__(cls, value, length):
        member = object.__new__(cls)
        member._value_ = value
        member.length = length
        return member


# Dictionary matching strings parsed from the JSON designating
# data types to data type codes for the
# input_data_type and output_data_type fields.
# Codes compatible with DLPack's DLDataTypeCode.
# Includes all data types supported by Kenning Zephyr Runtime.
IO_SPEC_STRUCT_DATA_TYPE_CODE = {
    "int": 0,
    "uint": 1,
    "float": 2,
    "opaque_handle": 3,
    "bfloat": 4,
    "complex": 5,
    "bool": 6,
    "e3m4_float": 7,
    "e4m3_float": 8,
    "e4m3b11fnuz_float": 9,
    "e4m3fn_float": 10,
    "e4m3fnuz_float": 11,
    "e5m2_float": 12,
    "e5m2fnuz_float": 13,
    "e8m0fnu_float": 14,
    "e2m3fn_float": 15,
    "e3m2fn_float": 16,
    "e2m1fn_float": 17,
}


# Named tuple used in the IOSPEC_STRUCT_FIELDS below to store
# shape and type of an io_spec struct field
IOSpecFieldType = namedtuple("IOSpecFieldType", ["shape", "type"])

# IO Struct template (will be sent as bytes to the target device
# and converted into a C++ packed struct
# All fields must match, so this is used for validation)
# The struct contains properties of input and output tensors
# (their shape and data type), as well as Model name and its
# entry function name
# Keys: struct field names, Values: (shape, size of element)
# CAUTION: 3D arrays are not supported by the serializer!
IOSPEC_STRUCT_FIELDS = {
    "num_input": IOSpecFieldType((1,), IOSpecStructFieldType.STANDARD_INTEGER),
    "num_input_dim": IOSpecFieldType(
        (MAX_MODEL_INPUT_NUM,),
        IOSpecStructFieldType.STANDARD_INTEGER,
    ),
    "input_shape": IOSpecFieldType(
        (MAX_MODEL_INPUT_NUM, MAX_MODEL_INPUT_DIM),
        IOSpecStructFieldType.STANDARD_INTEGER,
    ),
    "input_data_type": IOSpecFieldType(
        (MAX_MODEL_INPUT_NUM,),
        IOSpecStructFieldType.STANDARD_TYPE,
    ),
    "num_output": IOSpecFieldType(
        (1,), IOSpecStructFieldType.STANDARD_INTEGER
    ),
    "num_output_dim": IOSpecFieldType(
        (MAX_MODEL_OUTPUT_NUM,),
        IOSpecStructFieldType.STANDARD_INTEGER,
    ),
    "output_shape": IOSpecFieldType(
        (MAX_MODEL_OUTPUT_NUM, MAX_MODEL_OUTPUT_DIM),
        IOSpecStructFieldType.STANDARD_INTEGER,
    ),
    "output_data_type": IOSpecFieldType(
        (MAX_MODEL_OUTPUT_NUM,),
        IOSpecStructFieldType.STANDARD_TYPE,
    ),
    "entry_func": IOSpecFieldType(
        (MAX_LENGTH_ENTRY_FUNC_NAME,),
        IOSpecStructFieldType.STANDARD_CHARACTER,
    ),
    "model_name": IOSpecFieldType(
        (MAX_LENGTH_MODEL_NAME,),
        IOSpecStructFieldType.STANDARD_CHARACTER,
    ),
}


class IOSpecSerializer:
    """
    Serializer for a python dictionary (from a parsed JSON file), containing
    Machine Learning model's input and output specifications (shapes and
    data types of tensors, as well as some other information) - that data
    is referred to as IOSPEC.

    The serialized data is sent over UART to the target device, where it
    is read directly into a packed C struct (IOSPEC struct).

    For that reason, the format has to be binary-compatible between the
    apps - otherwise unexpected, hard to debug errors may occur.

    Therefore, extensive validation is performad during the serialization
    process (which is conducted by calling the _io_spec_to_struct method
    below - it returns raw bytes).
    """

    @classmethod
    def compute_iospec_struct_size(cls) -> int:
        """
        Method used to compute the expected IOSPEC struct size, used for
        validating the serialized data.

        Returns
        -------
        int
            Valid IO struct size.
        """
        return sum(
            math.prod(field.shape) * field.type.length
            for field in IOSPEC_STRUCT_FIELDS.values()
        )

    @classmethod
    def _generate_iospec_struct_template(cls) -> Dict[str, Any]:
        """
        Method used to generate a template based on IOSPEC_STRUCT_FIELDS, to
        fill with IOSPEC data. It serves to ensure correct order of
        serialized elements.

        Returns
        -------
        Dict[str, Any]
            IO specification structure.
        """
        return {name: None for name in IOSPEC_STRUCT_FIELDS.keys()}

    @classmethod
    def _generate_bytestream_from_iospec_struct(
        cls,
        struct_template: Dict[str, Any],
        byteorder: Literal["little", "big"] = "little",
    ) -> bytes:
        """
        Method used to convert IOSPEC struct template to a byte array, that
        will be sent to the target device and read as a packed C struct.
        Usage: Get a dictionary from function generate_iospec_struct_template.
        Fill in the parameters defined in IOSPEC_STRUCT_FIELDS. Pass the
        filled-in template to this function.

        Parameters
        ----------
        struct_template : Dict[str, Any]
            Correctly filled out iospec struct template.
        byteorder : Literal['little', 'big']
            Byteorder of the struct (either 'little' or 'big').

        Returns
        -------
        bytes
            Ready to send to the target device.

        Raises
        ------
        ValueError
            Raised on validation fails (when values put in the struct_template
            do not match the IOSPEC_STRUCT_FIELDS dictionary).
        """
        # Validating, whether there are unexpected fields in the struct
        for name, field in struct_template.items():
            if name not in IOSPEC_STRUCT_FIELDS:
                raise ValueError(
                    f"Invalid struct field: '{name}' with value: {field}"
                )
            # Validating whether all fields have values assigned
            if field is None:
                raise ValueError(f"Unassigned struct field: '{name}'")
            # Validating, whether all fields have correct sizes
            tmp_field = field
            if type(field) is not list:
                tmp_field = [field]
            shape = IOSPEC_STRUCT_FIELDS[name].shape
            if len(tmp_field) > shape[0]:
                raise ValueError(
                    f"Invalid struct field shape: {name} dimension 0 "
                    f"should be less than {shape[0]}, but is: {len(tmp_field)}"
                )
            if len(shape) > 2:
                raise ValueError(
                    f"Invalid struct field shape: {name}, should be less than "
                    f"2 dimensions, but there are {len(shape)} dimensions."
                )
            if len(shape) > 1:
                for dim in tmp_field:
                    if len(dim) > shape[1]:
                        raise ValueError(
                            f"Invalid struct field shape: {name} dimension 1 "
                            f"should be at most {shape[1]}, but is: {len(dim)}"
                        )

        # Validating, whether all expected fields are in the struct
        for name in IOSPEC_STRUCT_FIELDS.keys():
            if name not in struct_template:
                raise ValueError(f"Missing struct field: {name}")

        # functions for converting various data types to raw byte streams
        def int_to_bytes(num: int) -> bytes:
            return num.to_bytes(4, byteorder=byteorder, signed=False)

        def array_to_bytes(array: List, shape: Tuple) -> bytes:
            # Multi-dimensional lists with non-homogeneous shape cannot be
            # directly converted to a numpy array, so we are converting all 2D
            # lists to single-dimensional (this function cannot take 3D lists)
            dimensions = len(shape)
            if dimensions == 1:
                shape = shape[0]
            elif dimensions == 2:
                temp_array = []
                for sub_array in array:
                    temp_array.extend(sub_array)
                array = temp_array
                shape = math.prod(shape)
            else:
                raise ValueError(
                    f"The struct field given is {dimensions}-dimensional, "
                    f"up to 2 dimensions are supported. List shape: {shape}."
                )
            a_np = np.array(array, dtype=np.uint32)
            a_pad = np.pad(a_np, (0, shape - a_np.shape[0]))
            return a_pad.astype(np.uint32).tobytes("C")

        def str_to_bytes(string: str, length: int) -> bytes:
            return string.ljust(length, "\0")[:length].encode("ascii")

        def type_array_to_bytes(
            data_types: List[Tuple[int, int]], shape: int
        ) -> bytes:
            arr = []
            for data_type in data_types:
                # Every type is comprised of 2 uint8_t values
                # (DLPack-compliant dtype code and size in bits)
                # We load values in reverse order, (big/little endian)
                # It's written as 16-bit int, but then read as 8-bit ints
                full_type = (0b11111111 & data_type[1]) << 8
                full_type += 0b11111111 & data_type[0]
                arr.append(full_type)
            a_np = np.array(arr, dtype=np.uint16)
            a_pad = np.pad(a_np, (0, shape - a_np.shape[0]))
            return a_pad.astype(np.uint16).tobytes("C")

        # Generating the byte stream using the helper functions
        result = bytes()
        for name, field in struct_template.items():
            if (
                IOSPEC_STRUCT_FIELDS[name].type
                == IOSpecStructFieldType.STANDARD_CHARACTER
            ):
                result += str_to_bytes(
                    field,
                    math.prod(IOSPEC_STRUCT_FIELDS[name].shape)
                    * IOSPEC_STRUCT_FIELDS[name].type.length,
                )
            elif (
                IOSPEC_STRUCT_FIELDS[name].type
                == IOSpecStructFieldType.STANDARD_INTEGER
            ):
                if math.prod(IOSPEC_STRUCT_FIELDS[name].shape) == 1:
                    if type(field) is list:
                        result += int_to_bytes(sum(field))
                    else:
                        result += int_to_bytes(field)
                else:
                    result += array_to_bytes(
                        field, IOSPEC_STRUCT_FIELDS[name].shape
                    )
            elif (
                IOSPEC_STRUCT_FIELDS[name].type
                == IOSpecStructFieldType.STANDARD_TYPE
            ):
                result += type_array_to_bytes(
                    field, IOSPEC_STRUCT_FIELDS[name].shape[0]
                )
            else:
                raise ValueError(
                    f"Unrecognized struct field type at field {name}."
                )

        # Validating the final bytestream size
        correct_struct_size = cls.compute_iospec_struct_size()
        if len(result) != correct_struct_size:
            raise ValueError(
                f"Invalid struct length, should be {correct_struct_size}, "
                f"but is: {len(result)}"
            )

        # Returning the result
        return result

    @classmethod
    def io_spec_parse_types(
        cls,
        values: Dict[int, Any],
    ) -> List[Tuple[int, int]]:
        """
        Method takes parsed JSON structure with 'dtype' fields, that are
        strings in format: 'float32', 'int8' etc. It parses them into tuples:
        (code, size), where size is given in bits and code denotes the data
        type (see IO_SPEC_STRUCT_DATA_TYPE_CODE).

        Parameters
        ----------
        values : Dict[int, Any]
            Objects persed from JSON. Each value in the dictionary is a
            dictionary, containing a 'dtype' key, with a string value:
            'float32', 'int8' and similar values.

        Returns
        -------
        List[Tuple[int, int]]
            List of (code, size) tuples, where codes correspond to DLPack's
            DLDataTypeCode values and size is given in bits.

        Raises
        ------
        ValueError
            Raised when type is unrecognized, or when type size given is
            larger than 255 bits.
        """
        dtypes = []
        for value in values:
            dtype = value["dtype"]
            dtype_size_list = re.findall(r"\d+$", dtype)
            if len(dtype_size_list) != 1:
                raise ValueError(f"Invalid dtype {dtype}. One size is enough!")
            dtype_size = int(dtype_size_list[0])
            dtype_name_list = re.findall(r"[a-z|0-9_]*[a-z]+", dtype)
            dtype_name = max(dtype_name_list)
            dtype_code = IO_SPEC_STRUCT_DATA_TYPE_CODE[dtype_name]
            if dtype_size > 255 or dtype_size == 0:
                raise ValueError(f"Invalid dtype {dtype} (invalid size).")
            dtypes.append((dtype_code, dtype_size))
        return dtypes

    @classmethod
    def io_spec_to_struct(
        cls,
        io_spec: Dict[str, Any],
        model_name: str = "module",
        byteorder: Literal["little", "big"] = "little",
    ) -> bytes:
        """
        Method used to convert IO spec in JSON form into struct that can be
        easily parsed by runtime.

        Parameters
        ----------
        io_spec : Dict[str, Any]
            Input IO specification.
        model_name : str
            Name of the model.
        byteorder : Literal['little', 'big']
            Byteorder of the struct (either 'little' or 'big').

        Returns
        -------
        bytes
            IO specification structure.

        Raises
        ------
        ValueError
            Raised when arguments are invalid.
        """
        # Getting the io spec struct template to fill-in
        struct_to_send = cls._generate_iospec_struct_template()

        # Depending on the model the JSON file can have both input and
        # processed_input keys in it - we want to choose processed_input,
        # if possible, because input pre-processing is done in Kenning
        # (so the target device gets processed input)
        input_key = (
            "processed_input" if "processed_input" in io_spec else "input"
        )

        # Post processing is also done in Kenning, not in the target device,
        # so we always want non-processed output
        output_key = "output"

        # Some models (for example IREE models) require calling an entry
        # function to start inference. For these models the entry function name
        # should be located in the iospec JSON file. Otherwise an empty string
        # will be assumed.
        entry_func = io_spec.get("entry_func", "")

        # Filling in the template - inputs
        if len(io_spec[input_key]) > 0:
            struct_to_send["input_shape"] = [
                inp["shape"] for inp in io_spec[input_key]
            ]
            struct_to_send["num_input"] = len(struct_to_send["input_shape"])
            struct_to_send["num_input_dim"] = [
                len(inp) for inp in struct_to_send["input_shape"]
            ]
            input_dtypes = cls.io_spec_parse_types(io_spec[input_key])
            struct_to_send["input_data_type"] = input_dtypes
        else:  # No inputs
            struct_to_send["input_shape"] = [[]]
            struct_to_send["num_input"] = 0
            struct_to_send["num_input_dim"] = []
            struct_to_send["input_data_type"] = []

        # Filling in the template - outputs
        if len(io_spec[output_key]) > 0:
            struct_to_send["output_shape"] = [
                out["shape"] for out in io_spec[output_key]
            ]
            struct_to_send["num_output"] = len(struct_to_send["output_shape"])
            struct_to_send["num_output_dim"] = [
                len(out) for out in struct_to_send["output_shape"]
            ]
            output_dtypes = cls.io_spec_parse_types(io_spec[output_key])
            struct_to_send["output_data_type"] = output_dtypes
        else:  # No outputs
            struct_to_send["output_shape"] = [[]]
            struct_to_send["num_output"] = 0
            struct_to_send["num_output_dim"] = []
            struct_to_send["output_data_type"] = []

        struct_to_send["entry_func"] = entry_func
        struct_to_send["model_name"] = model_name

        # Validating whether values fit constraints
        if struct_to_send["num_input"] > MAX_MODEL_INPUT_NUM:
            raise ValueError(
                "Too many inputs: "
                f"{struct_to_send['num_input']} > {MAX_MODEL_INPUT_NUM}"
            )
        for dim in struct_to_send["num_input_dim"]:
            if dim > MAX_MODEL_INPUT_DIM:
                raise ValueError(
                    "Too many input dimensions: should be no more than "
                    f"{MAX_MODEL_INPUT_DIM}, but is {dim}"
                )
        if struct_to_send["num_output"] > MAX_MODEL_OUTPUT_NUM:
            raise ValueError(
                "Too many outputs: "
                f"{struct_to_send['num_output']} > {MAX_MODEL_OUTPUT_NUM}"
            )
        for dim in struct_to_send["num_output_dim"]:
            if dim > MAX_MODEL_OUTPUT_DIM:
                raise ValueError(
                    "Too many output dimensions: should be no more than "
                    f"{MAX_MODEL_OUTPUT_DIM}, but is {dim}"
                )
        if len(entry_func) > MAX_LENGTH_ENTRY_FUNC_NAME:
            raise ValueError(
                "Invalid entry func name (too long, should be no more than "
                f"{MAX_LENGTH_ENTRY_FUNC_NAME}): {entry_func}"
            )
        if len(model_name) > MAX_LENGTH_MODEL_NAME:
            raise ValueError(
                "Invalid model name (too long, should be no more than "
                f"{MAX_LENGTH_MODEL_NAME}): {model_name}"
            )

        # Returning bytestream generated from the filled-in template
        return cls._generate_bytestream_from_iospec_struct(
            struct_to_send, byteorder
        )
