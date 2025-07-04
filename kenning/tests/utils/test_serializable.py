# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext as does_not_raise
from typing import Any, Dict

import pytest

from kenning.utils.serializable import Serializable


def serializables_equal(left: Serializable, right: Serializable) -> bool:
    assert left.serializable_fields == right.serializable_fields
    for field, _, _ in left.serializable_fields:
        assert getattr(left, str(field)) == getattr(
            right, str(field)
        ), f"Differing field: {str(field)}"


class ExampleClass(Serializable):
    serializable_fields = (
        ("a", int, 6),
        ("b", int, 7),
        ("c", bool, 1),
        ("d", int, 3),
        ("e", int, 8),
        ("f", int, 15),
        ("g", int, 3),
    )

    def __init__(self, values: Dict[str, Any] = {}):
        for key, value in values.items():
            setattr(self, key, value)


class TestSerializable:
    @pytest.mark.parametrize(
        "test_fields,serialized_fields,expectation",
        [
            # Valid fields
            (
                {
                    "a": 0x0A,
                    "b": 0x1C,
                    "c": 0x01,
                    "d": 0x00,
                    "e": 0xFF,
                    "f": 0x5DF3,
                    "g": 0x01,
                },
                b"\x0A\x27\xFE\xE7\xBB\x01",
                does_not_raise(),
            ),
            # Valid fields with an extra field, that should be ignored
            (
                {
                    "a": 0x0A,
                    "b": 0x1C,
                    "c": 0x01,
                    "d": 0x00,
                    "e": 0xFF,
                    "f": 0x5DF3,
                    "g": 0x01,
                    "redundant": 0x00,
                },
                b"\x0A\x27\xFE\xE7\xBB\x01",
                does_not_raise(),
            ),
            # Invalid field value ('e' field too large)
            (
                {
                    "a": 0x0A,
                    "b": 0x1C,
                    "c": 0x01,
                    "d": 0x00,
                    "e": 0x100,
                    "f": 0x5DF3,
                    "g": 0x01,
                },
                b"\x00\x00\x00\x00\x00\x00",
                pytest.raises(ValueError),
            ),
            # Invalid field value (less than 0)
            (
                {
                    "a": 0x0A,
                    "b": 0x1C,
                    "c": 0x01,
                    "d": 0x00,
                    "e": 0x100,
                    "f": -0x0040,
                    "g": 0x01,
                },
                b"\x00\x00\x00\x00\x00\x00",
                pytest.raises(ValueError),
            ),
        ],
    )
    def test_to_bytes(
        self,
        test_fields: Dict[str, Any],
        serialized_fields: bytes,
        expectation,
    ):
        with expectation:
            fields = ExampleClass(test_fields)
            assert serialized_fields == fields.to_bytes()

    @pytest.mark.parametrize(
        "serialized_fields,deserialized_fields,expectation",
        [
            # Too short bytestream (only 1 byte)
            (b"\x44", {}, pytest.raises(ValueError)),
            # Empty bytestream
            (b"", {}, pytest.raises(ValueError)),
            # Too long bytestream word (7 bytes)
            (
                b"\x44\x40\x00\x44\x40\x00\x76",
                {},
                pytest.raises(ValueError),
            ),
            # Too long bytestream (16 bytes)
            (
                b"\x44\x40\x00\x10\x01\x56\x89\xC0\x44\x40\x00\x10\x01\x56\x89\xC0",
                {},
                pytest.raises(ValueError),
            ),
            # Valid fields
            (
                b"\x01\x2F\x1E\x02\x02\x07",
                {
                    "a": 0x01,
                    "b": 0x3C,
                    "c": 0x01,
                    "d": 0x00,
                    "e": 0x0F,
                    "f": 0x0101,
                    "g": 0x07,
                },
                does_not_raise(),
            ),
        ],
    )
    def test_from_bytes(
        self,
        serialized_fields: bytes,
        deserialized_fields: Dict[str, Any],
        expectation,
    ):
        with expectation:
            serializables_equal(
                ExampleClass(deserialized_fields),
                ExampleClass.from_bytes(serialized_fields),
            )

    def test_compatibility(self):
        test_object = ExampleClass(
            {
                "a": 0x3F,
                "b": 0x13,
                "c": 0x01,
                "d": 0x04,
                "e": 0x00,
                "f": 0x3FAC,
                "g": 0x04,
            },
        )
        serialized = test_object.to_bytes()
        serializables_equal(test_object, ExampleClass.from_bytes(serialized))
