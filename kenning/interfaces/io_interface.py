# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides implementation of interface used by other Kenning components to manage
their input and output types.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from kenning.utils.resource_manager import PathOrURI


class IOInterface(ABC):
    """
    Interface that provides methods for accessing input/output specifications
    and validating them.
    """

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
            if global_name in output_spec.keys():
                try:
                    single_output_spec = output_spec[global_name]
                    # dtype and type are mutually exclusive
                    if ("dtype" in single_input_spec) == (
                        "type" in single_input_spec
                    ):
                        return False
                    # validate dtype and shape
                    if (
                        "dtype" in single_input_spec
                        and "shape" in single_input_spec
                    ):
                        if (
                            single_input_spec["dtype"]
                            != single_output_spec["dtype"]
                        ):
                            return False
                        if isinstance(single_input_spec["shape"], list):
                            # multiple valid shapes
                            found_valid_shape = False
                            for input_shape in single_input_spec["shape"]:
                                if IOInterface._validate_shape(
                                    single_output_spec["shape"], input_shape
                                ):
                                    found_valid_shape = True
                                    break

                            if not found_valid_shape:
                                return False

                        else:
                            return IOInterface._validate_shape(
                                single_output_spec["shape"],
                                single_input_spec["shape"],
                            )
                    # dtype and shape imply each other
                    elif (
                        "dtype" in single_input_spec
                        or "shape" in single_input_spec
                    ):
                        return False
                    # validate type
                    if "type" in single_input_spec:
                        if (
                            single_input_spec["type"] != "Any"
                            and single_input_spec["type"]
                            != single_output_spec["type"]
                        ):
                            return False
                except KeyError:
                    return False
            else:
                return False

        return True

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
            if not (l1 == l2 or l2 == -1):
                return False
        return True


class IOCompatibilityError(Exception):
    """
    Exception is raised when input and output are not compatible.
    """

    def __init__(self, *args) -> bool:
        super().__init__(*args)


class IOSpecNotFound(Exception):
    """
    Exception is raised when IO specification is not found.
    """

    def __init__(self, *args) -> bool:
        super().__init__(*args)
