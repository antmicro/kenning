# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from itertools import chain, permutations, product
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from kenning.datasets.helpers.detection_and_segmentation import (
    DetectObject,
    SegmObject,
)
from kenning.interfaces.io_interface import IOCompatibilityError, IOInterface

LIST_SPEC = {
    "test1": [
        {
            "type": "List",
            "dtype": "kenning.datasets.helpers.detection_and_segmentation.SegmObject",  # noqa: E501
        }
    ],
}
LIST_LIST_SPEC = {
    "test1": [
        {
            "type": "List",
            "dtype": {
                "type": "List",
                "dtype": "kenning.datasets.helpers.detection_and_segmentation.SegmObject",  # noqa: E501
            },
        }
    ],
}
LIST_DICT_SPEC = {
    "test1": [
        {
            "type": "List",
            "dtype": {
                "type": "Dict",
                "fields": {
                    "test2": {
                        "type": "List",
                        "dtype": "kenning.datasets.helpers.detection_and_segmentation.SegmObject",  # noqa: E501
                    }
                },
            },
        }
    ],
}

# List of list with specs compatible with each other
SPECS_VALID = (
    (
        {"test1": [{"shape": (1, 2, 4, 8), "dtype": "float32"}]},
        {"test1": [{"shape": (1, 2, 4, 8), "dtype": "float32"}]},
        {"test1": [{"shape": (1, 2, -1, 8), "dtype": "float32"}]},
        {"test1": [{"shape": (1, -1, 4, -1), "dtype": "float32"}]},
    ),
    (
        {
            "test1": [
                {"shape": (32, 128), "dtype": "int8"},
                {"shape": (32, 2), "dtype": "int8"},
            ]
        },
        {
            "test1": [
                {"shape": (-1, 128), "dtype": "int8"},
                {"shape": (-1, 2), "dtype": "int8"},
            ]
        },
    ),
    (
        {
            "test1": [{"shape": (32, 2), "dtype": "float16"}],
            "test2": [{"shape": (-1, 16), "dtype": "float16"}],
        },
        {
            "test2": [{"shape": (2, -1), "dtype": "float16"}],
            "test1": [{"shape": (-1, 2), "dtype": "float16"}],
        },
    ),
    (LIST_SPEC, LIST_SPEC),
    (LIST_LIST_SPEC, LIST_LIST_SPEC),
    (LIST_DICT_SPEC, LIST_DICT_SPEC),
)

# List of list with specs not compatible with specs from SPECS_VALID
SPECS_INVALID = (
    (
        {"test1": [{"shape": (4, 2, -1, 8), "dtype": "float32"}]},
        {"test1": [{"shape": (1, 2, 4, 8), "dtype": "float16"}]},
    ),
    (
        {
            "test1": [
                {"shape": (-1, 128), "dtype": "int8"},
            ]
        },
        {
            "test1": [
                {"shape": (32, 128), "dtype": "int16"},
                {"shape": (32, -1), "dtype": "int8"},
            ]
        },
    ),
    (
        {
            "test1": [{"shape": (32, 2), "dtype": "float16"}],
            "test2": [{"shape": (3, 15), "dtype": "float16"}],
        },
        {
            "test2": [{"shape": (2, -1), "dtype": "float16"}],
            "test1": [{"shape": (-1, 2), "dtype": "float16"}],
            "test3": [{"shape": (-1, 2), "dtype": "float16"}],
        },
    ),
    (
        {
            "test1": [
                {
                    "type": "List",
                    "dtype": "kenning.datasets.helpers.detection_and_segmentation.DetectObject",  # noqa: E501
                }
            ]
        },
    ),
    (
        {
            "test1": [
                {
                    "type": "List",
                    "dtype": {
                        "type": "List",
                        "dtype": "kenning.datasets.helpers.detection_and_segmentation.DetectObject",  # noqa: E501
                    },
                }
            ]
        },
        LIST_SPEC,
    ),
    (
        {
            "test1": [
                {
                    "type": "List",
                    "dtype": {
                        "type": "Dict",
                        "fields": {
                            "test2": {
                                "type": "List",
                                "dtype": "kenning.datasets.helpers.detection_and_segmentation.DetectObject",  # noqa: E501
                            }
                        },
                    },
                }
            ]
        },
        LIST_SPEC,
        LIST_LIST_SPEC,
    ),
)

DETECT_OBJ = DetectObject("test", 0, 0, 1, 1, 0.5, False)
SEGM_OBJ = SegmObject("test", "./mask", 0, 0, 1, 1, "mask", 0.5, False)

# List of list of data compatible with specs from SPECS_VALID
DATA_VALID = (
    (np.ones((1, 2, 4, 8), dtype=np.float32),),
    ([np.ones((32, 128), dtype=np.int8), np.zeros((32, 2), dtype=np.int8)],),
    (np.ones((32, 2), dtype=np.float16),),
    ([SEGM_OBJ, SEGM_OBJ],),
    ([[SEGM_OBJ, SEGM_OBJ]],),
    ([{"test2": [SEGM_OBJ, SEGM_OBJ]}],),
)

# List of list of data not compatible with specs from SPECS_VALID
DATA_INVALID = (
    (np.ones((5, 2, 4, 8), dtype=np.float32),),
    (np.ones((32, 128), dtype=np.int8),),
    (np.ones((32,), dtype=np.float128),),
    ([SEGM_OBJ, DETECT_OBJ],),
    ([SEGM_OBJ, SEGM_OBJ],),
    ([{"test": [SEGM_OBJ, SEGM_OBJ]}],),
)


class TestIOInterface:
    @pytest.mark.parametrize(
        "specs,expect_fail",
        chain(
            *[
                product(permutations(specs, 2), [False])
                for specs in SPECS_VALID
            ],
            *[
                product(product(specs_valid, specs_invalid), [True])
                for specs_valid, specs_invalid in zip(
                    SPECS_VALID, SPECS_INVALID
                )
            ],
        ),
    )
    def test_validate(self, specs: Tuple[Dict, Dict], expect_fail: bool):
        expected_result = not expect_fail
        assert IOInterface.validate(specs[0], specs[1]) == expected_result

    @pytest.mark.parametrize(
        "spec,data",
        chain(
            *[
                product(specs, data)
                for specs, data in zip(SPECS_VALID, DATA_VALID)
            ]
        ),
    )
    def test_assert_data_format_valid(self, spec: Dict, data: Any):
        IOInterface.assert_data_format(data, spec["test1"])

    @pytest.mark.parametrize(
        "spec,data",
        chain(
            *[
                product(specs, data)
                for specs, data in zip(SPECS_VALID, DATA_INVALID)
            ]
        ),
    )
    def test_assert_data_format_invalid(self, spec: Dict, data: Any):
        with pytest.raises(IOCompatibilityError):
            IOInterface.assert_data_format(data, spec["test1"])
