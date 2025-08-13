# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides implementation of interface used by other Kenning components to manage
their input and output types.
"""

import builtins
import importlib
import json
import os
import traceback
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
from numpy.typing import ArrayLike

from kenning.core.exceptions import (
    IOCompatibilityError,
    IOSpecNotFound,
    IOSpecWrongFormat,
)
from kenning.interfaces.io_spec_serializer import IOSpecSerializer
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI

KENNING_DISABLE_IO_VALIDATION = os.environ.get(
    "KENNING_DISABLE_IO_VALIDATION", False
)


def _load_class(class_path: str) -> Type:
    """
    Load class or object from given path.

    Parameters
    ----------
    class_path : str
        Class path of object that will be imported.
        It can represent full path or built-in type.

    Returns
    -------
    Type
        Loaded class or object

    Raises
    ------
    ModuleNotFoundError
        Raised if class or object was not found.
    """
    if "." in class_path:
        module_name, cls_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls_type = getattr(module, cls_name, None)
    else:
        cls_type = getattr(builtins, class_path, None)
    if cls_type:
        return cls_type
    raise ModuleNotFoundError(f"Type under {class_path} cannot be found")


def disable_when_defined(
    should_disable: Optional[Any],
) -> Callable[[Callable], Callable]:
    """
    Function creating decorator, which disables function
    when specified `should_disable` evaluates to `True`.

    Parameters
    ----------
    should_disable : Optional[Any]
        Object, is defined when `bool(should_disable)` is True.

    Returns
    -------
    Callable[[Callable], Callable]
        Decorator returning empty function when should_disable is defined.
    """

    def _decorator(func: Callable):
        if not should_disable:
            return func

        def noop(*args, **kwargs):
            ...

        return noop

    return _decorator


class IOInterface(ABC):
    """
    Interface that provides methods for accessing input/output specifications
    and validating them.
    """

    @staticmethod
    def _validate_non_type_specs(
        single_output_spec: Dict,
        single_input_spec: Dict,
    ) -> bool:
        """
        Method validating two specification items for array-based IO.
        Specs should not contain `type` parameter.

        Parameters
        ----------
        single_output_spec : Dict
            Single item from output specification.
        single_input_spec : Dict
            Single item from input specification.

        Returns
        -------
        bool
            False if specifications do not match, True otherwise.
        """
        # dtype and shape imply each other
        if not ("dtype" in single_input_spec and "shape" in single_input_spec):
            KLogger.error("dtype and shape have to be defined")
            return False
        # validate dtype and shape
        if single_input_spec["dtype"] != single_output_spec["dtype"]:
            KLogger.error(
                "dtypes do not match: "
                f"{single_output_spec['dtype']} {single_input_spec['dtype']}"
            )
            return False
        input_shape = single_input_spec["shape"]
        output_shape = single_output_spec["shape"]
        if isinstance(input_shape[0], Iterable):
            # multiple valid shapes
            found_valid_shape = False
            for shape in input_shape:
                if IOInterface._validate_shape(
                    output_shape,
                    shape,
                ):
                    found_valid_shape = True
                    break

            if not found_valid_shape:
                KLogger.error(
                    f"shapes do not match: {output_shape} {input_shape}"
                )
                return False

        elif not IOInterface._validate_shape(
            output_shape,
            input_shape,
        ):
            KLogger.error(f"shapes do not match: {output_shape} {input_shape}")
            return False
        return True

    @staticmethod
    def _validate_type_specs(
        single_output_spec: Dict,
        single_input_spec: Dict,
    ) -> bool:
        """
        Method validating two specification items for non-array-based IO.
        Specs should contain `type` parameter.

        Parameters
        ----------
        single_output_spec : Dict
            Single item from output specification.
        single_input_spec : Dict
            Single item from input specification.

        Returns
        -------
        bool
            False if specifications do not match, True otherwise.

        Raises
        ------
        IOSpecWrongFormat
            Raised if unsupported `type` or `dtype` is used.
        """
        # validate type
        if (
            single_input_spec["type"] != "Any"
            and single_input_spec["type"] != single_output_spec["type"]
        ):
            KLogger.error(
                "types do not match: "
                f"{single_output_spec['type']} {single_input_spec['type']}"
            )
            return False
        if single_input_spec["type"] == "Any":
            return True
        # validate Dict type specifications
        if single_input_spec["type"] == "Dict":
            if not (
                "fields" in single_input_spec
                and "fields" in single_output_spec
            ):
                KLogger.error(
                    "Specifiaction with `type = Dict` must have `fields` defined"  # noqa: E501
                )
                return False
            if set(single_input_spec["fields"].keys()) != set(
                single_output_spec["fields"].keys()
            ):
                KLogger.error(
                    "fields' keys do not match: "
                    f"{single_output_spec['fields'].keys()} {single_input_spec['fields'].keys()}"  # noqa: E501
                )
                return False
            output_fields_specs, input_fields_specs = [], []
            for key, input_field in single_input_spec["fields"].items():
                output_fields_specs.append(single_output_spec["fields"][key])
                input_fields_specs.append(input_field)
            if not IOInterface.validate(
                {"*": output_fields_specs},
                {"*": input_fields_specs},
            ):
                return False
            return True
        elif single_input_spec["type"] != "List":
            raise IOSpecWrongFormat(
                f"Unsupported value of `type`: {single_input_spec['type']}"
            )
        # validate List type specification
        input_dtype = single_input_spec["dtype"]
        output_dtype = single_output_spec["dtype"]
        if type(input_dtype) is not type(output_dtype):
            KLogger.error(
                "dtypes have different types: "
                f"{type(output_dtype)} {type(input_dtype)}"
            )
            return False
        # validate dtype containing class path
        if isinstance(input_dtype, str):
            input_dtype = _load_class(input_dtype)
            output_dtype = _load_class(output_dtype)
            if not issubclass(output_dtype, input_dtype):
                KLogger.error(
                    "dtypes classes do not match: "
                    f"{output_dtype} {input_dtype}"
                )
                return False
        # validate dtype in Dict format
        elif isinstance(input_dtype, Dict):
            if set(input_dtype.keys()) != set(output_dtype.keys()):
                KLogger.error(
                    "dtypes' keys do not match: "
                    f"{output_dtype.keys()} {input_dtype.keys()}"
                )
                return False
            if not IOInterface._validate_type_specs(
                output_dtype,
                input_dtype,
            ):
                return False
        else:
            raise IOSpecWrongFormat(
                f"Unsupported specification `dtype`: {input_dtype}"
            )
        return True

    @staticmethod
    def validate(
        output_spec: Dict[str, List[Dict]], input_spec: Dict[str, List[Dict]]
    ) -> bool:
        """
        Method that checks compatibility between output of some object and
        input of another. Output is considered compatible if for each input
        assigned output and their type is compatible.

        Parameters
        ----------
        output_spec : Dict[str, List[Dict]]
            Specification of some object's output.
        input_spec : Dict[str, List[Dict]]
            Specification of some object's input.

        Returns
        -------
        bool
            True if there is no conflict.
        """
        if len(input_spec) > len(output_spec):
            return False

        for global_name, single_input_spec in input_spec.items():
            if global_name not in output_spec:
                KLogger.error(f"Output spec does not have {global_name} key")
                return False
            single_output_spec = output_spec[global_name]
            if len(single_input_spec) != len(single_output_spec):
                KLogger.error(
                    f"Non-matching number of {global_name} specification"
                )
                return False
            for item_output_spec, item_input_spec in zip(
                single_output_spec, single_input_spec
            ):
                try:
                    # validate specification without type
                    if "type" not in item_input_spec:
                        if not IOInterface._validate_non_type_specs(
                            item_output_spec, item_input_spec
                        ):
                            return False
                    # validate specification with type
                    elif not IOInterface._validate_type_specs(
                        item_output_spec, item_input_spec
                    ):
                        return False
                except KeyError as er:
                    traceback.print_tb(er.__traceback__)
                    KLogger.error(f"Missing key during spec validation: {er}")
                    return False

        return True

    @staticmethod
    @disable_when_defined(KENNING_DISABLE_IO_VALIDATION)
    def assert_data_format(
        data: Union[Iterable, Dict, List[Dict]], spec: List[Dict]
    ) -> None:
        """
        Method that checks if received data matches specification
        - shape and data type are the same as in specification.

        Parameters
        ----------
        data : Union[Iterable, Dict, List[Dict]]
            Object that should match specification.
        spec : List[Dict]
            Input or Output specification.

        Raises
        ------
        IOSpecWrongFormat
            Raised if unsupported `type` or `dtype` is used.
        IOCompatibilityError
            Raised if data does not match with specification.
        """
        if len(spec) != len(data):
            raise IOCompatibilityError(
                "Number of received data do not match specification"
            )

        for data_item, spec_item in zip(data, spec):
            if "type" in spec_item:
                if spec_item["type"] == "Any":
                    continue
                if spec_item["type"] == "List":
                    if not isinstance(data_item, List):
                        raise IOCompatibilityError(
                            f"Data should be a list, received: {type(data_item)}"  # noqa: E501
                        )
                    dtype = spec_item["dtype"]
                    # validate dtype in Dict format
                    if isinstance(dtype, Dict):
                        for item in data_item:
                            IOInterface.assert_data_format([item], [dtype])
                    # validate dtype containing class path
                    elif isinstance(dtype, str):
                        type_cls = _load_class(dtype)
                        for item in data_item:
                            if not isinstance(item, type_cls):
                                raise IOCompatibilityError(
                                    "Data does not have right type, required:"
                                    f" {type_cls}, received: {type(data)}"
                                )
                    else:
                        raise IOSpecWrongFormat(
                            f"Unsupported type of `dtype`: {dtype}"
                        )
                elif spec_item["type"] == "Dict":
                    if "fields" not in spec_item:
                        raise IOSpecWrongFormat(
                            "type `Dict` requires to have `fields` defined"
                        )
                    # validate all required fields
                    if not isinstance(data_item, Dict) or set(
                        data_item.keys()
                    ) != set(spec_item["fields"].keys()):
                        raise IOCompatibilityError(
                            "Data is not an Dict or does not have all required key, "  # noqa: E501
                            f"data: {data_item} spec: {spec_item}"
                        )
                    for key, spec_dict_item in spec_item["fields"].items():
                        IOInterface.assert_data_format(
                            [data_item[key]], [spec_dict_item]
                        )
                else:
                    raise IOSpecWrongFormat(
                        f"Unsupported specification `type`: {spec_item['type']}"  # noqa: E501
                    )
                continue

            array_item = data_item
            if not hasattr(array_item, "shape") and not hasattr(
                array_item, "dtype"
            ):
                try:
                    array_item = np.asarray(array_item)
                except Exception as ex:
                    raise IOCompatibilityError(
                        "Shape and dtype of received object"
                        f" cannot be retrieved: {array_item}"
                    ) from ex
            # Validate shape and dtype
            IOInterface._check_shape_dtype(array_item, spec_item)
        return True

    @staticmethod
    def _check_shape_dtype(entry: ArrayLike, entry_spec: Dict):
        """
        Checks if shape and data type match the specification.

        Parameters
        ----------
        entry : ArrayLike
            Array like object with shape and dtype attribute.
        entry_spec : Dict
            Specification of the given entry.

        Raises
        ------
        IOCompatibilityError
            Raised if entry shape does not match with specification.
        """
        # Check shape
        shape = tuple(entry.shape)
        check_shapes = (
            entry_spec["shape"]
            if isinstance(entry_spec["shape"][0], Iterable)
            else [entry_spec["shape"]]
        )
        valid_shape = False
        # Check full shapes
        for entry_shape in check_shapes:
            if IOInterface._validate_shape(shape, entry_shape):
                valid_shape = True
                break
        if not valid_shape:
            # Check shapes without batch size (first dimension)
            for entry_shape in check_shapes:
                if len(shape) == len(
                    entry_shape
                ) and IOInterface._validate_shape(shape[1:], entry_shape[1:]):
                    valid_shape = True
                    KLogger.warn(
                        "Shape with different batch size,"
                        f" required: {entry_shape}, received: {shape}"
                    )
                    break
        if not valid_shape:
            raise IOCompatibilityError(
                "Wrong shape, required: "
                f"{entry_spec['shape']}, received: {shape}"
            )

        # Check type
        if getattr(
            entry_spec,
            "prequantized_dtype",
            entry_spec["dtype"],
        ) not in str(entry.dtype):
            KLogger.warning(
                "Wrong data type, required: "
                f"{entry_spec['dtype']}, received: {str(entry.dtype)}"
            )

    @abstractmethod
    def get_io_specification(self) -> Dict[str, List[Dict]]:
        """
        Returns dictionary with `input` and `output` keys that map to input and
        output specifications.

        A single specification is a list of dictionaries with names, shapes and
        dtypes for each layer.

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary that conveys input and output
            layers specification.
        """
        ...

    @classmethod
    def parse_io_specification_from_json(
        cls, json_dict: Dict
    ) -> Dict[str, List[Dict]]:
        """
        Return dictionary with 'input' and 'output' keys that will map to input
        and output specification of an object created by the argument json
        schema.

        A single specification is a list of dictionaries with names, shapes and
        dtypes for each layer.

        Since no object initialization is done for this method, some IO
        specification may be incomplete, this method fills in -1 in case
        the information is missing from the JSON dictionary.

        Parameters
        ----------
        json_dict : Dict
            Parameters for object constructor in JSON format.

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary that conveys input and output layers specification.
        """
        ...

    @staticmethod
    def serialize_io_specification_for_uart(
        json_dict: Dict[str, List[Dict]]
    ) -> bytes:
        return IOSpecSerializer.io_spec_to_struct(json_dict)

    def save_io_specification(self, path: PathOrURI):
        """
        Saves input/output specification to a file named `path` + `.json`. This
        function uses `get_io_specification()` function to get the properties.

        Parameters
        ----------
        path : PathOrURI
            Path that is used to store the input/output specification.
        """
        spec_path = path.with_suffix(path.suffix + ".json")

        if not spec_path.parent.exists():
            spec_path.parent.mkdir(parents=True, exist_ok=True)

        with open(spec_path, "w") as f:
            json.dump(self.get_io_specification(), f)

    def load_io_specification(self, path: PathOrURI) -> Dict[str, List[Dict]]:
        """
        Loads input/output specification from a file named `path` + `.json`.

        Parameters
        ----------
        path : PathOrURI
            Path that is used to store the input/output specification.

        Returns
        -------
        Dict[str, List[Dict]]
            Loaded IO specification.
        """
        spec_path = path.with_suffix(path.suffix + ".json")

        with open(spec_path, "r") as f:
            return json.load(f)

    @staticmethod
    def find_spec(
        io_spec: Dict[str, List[Dict]], io_type: str, io_name: str
    ) -> Dict[str, Any]:
        """
        Find single IO specification based on type (input, output) and name.

        Parameters
        ----------
        io_spec : Dict[str, List[Dict]]
            IO specification to be searched in.
        io_type : str
            Type of IO (input, output, processed_input or processed_output).
        io_name : str
            Name of the IO.

        Returns
        -------
        Dict[str, Any]
            Specification of single IO.

        Raises
        ------
        IOSpecNotFound
            Raised when the input/output of given name was not
            found in the specification
        """
        for spec in io_spec[io_type]:
            if spec["name"] == io_name:
                return spec

        raise IOSpecNotFound(
            f"{io_type} spec with name {io_name} not found in IO "
            f"specification:\n{io_spec}"
        )

    @staticmethod
    def _validate_shape(
        output_shape: Tuple[int, ...], input_shape: Tuple[int, ...]
    ) -> bool:
        """
        Methods that checks shapes compatibility. If the length of dimension is
        set to -1 it means that any length at this dimension is compatible.

        Parameters
        ----------
        output_shape : Tuple[int, ...]
            First shape.
        input_shape : Tuple[int, ...]
            Second shape.

        Returns
        -------
        bool
            True if there is no conflict.
        """
        if len(output_shape) != len(input_shape):
            return False
        for l1, l2 in zip(output_shape, input_shape):
            if not (l1 == l2 or l2 == -1 or l1 == -1):
                return False
        return True
