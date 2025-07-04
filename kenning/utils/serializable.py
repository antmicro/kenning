# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Universal mechanism for bitwise packing and serialization of values.

Usage:

Create a class, with a property 'serializable_fields', which should be tuple
containing a list of all fields to be serialized. Values placed first in the
tuple will be placed in the youngest bits of the serialized value, ones placed
last in the tuple will be placed in the oldest bits. Details on the tuple's
format are available in the 'Serializable' class docstring. The class should
inherit from 'serializable'. All fields listed in 'serializable_fields' will
be available as class properties. Methods 'bits', 'size', 'to_bytes' and
'from_bytes' will be available.
"""

from abc import ABC
from math import ceil

from kenning.utils.logger import KLogger


class Serializable(ABC):
    """
    Parent class, for classes that need to perform serialization and
    de-serialization on their fields, that have size below 1 byte.

    List of fields to serialize/de-serialize should be given in a class
    property 'serializable_fields', which is a tuple.

    Format:

    serializable_fields = (
        (<field_name>, <field_type>, <field_bits>)
    )

    * field_name - Should either be a string, or have the '__str__' method
      implemented. Class property storing value of the field should have
      the same name.
    * field_type - Type of the field. Needs to have the default constructor
      available (constructor with no arguments except 'self').
    * field_bits - Size of the field in bits.
    """

    serializable_fields = ()

    def __init__(self):
        for field in self.serializable_fields:
            setattr(self, str(field[0]), 0)

    @classmethod
    def bits(cls) -> int:
        """
        Computes size in bits of the object after serializing.

        Returns
        -------
        int
            Size in bits.
        """
        result = 0
        for _, _, field_bits in cls.serializable_fields:
            result += field_bits
        return result

    @classmethod
    def size(cls) -> int:
        """
        Computes size of the object after serializing.

        Returns
        -------
        int
            Size in bytes.
        """
        return ceil(cls.bits() / 8)

    def to_bytes(self) -> bytes:
        """
        Performs bitwise packing and serialization of fields in an inheriting
        class. List of fields to serialize is given in the
        'serializable_fields' tuple.


        Returns
        -------
        bytes
            Serialized object.

        Raises
        ------
        ValueError
            Value not fitting in the specified number of bits, or value is less
            than 0.
        """
        data = 0
        bits_packed = 0
        for field, field_type, bits in self.serializable_fields:
            value = int(getattr(self, str(field)))
            if value < 0 or value >= (1 << bits):
                raise ValueError(
                    f"Invalid {str(field)} value: {value} (must be less than"
                    f" {(1 << bits)} and non-negative)."
                )
            data += value << bits_packed
            bits_packed += bits
        return data.to_bytes(self.size(), "little")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Serializable":
        """
        Performs bitwise de-serialization of bytes. List of fields to
        de-serialize is given in the 'serializable_fields' tuple.

        Parameters
        ----------
        data : bytes
            List of flag names to unpack.

        Returns
        -------
        Serializable
            Object of an inheriting class, with fields set to proper values.

        Raises
        ------
        ValueError
            Wrong number of bytes to de-serialize.
        """
        if len(data) != cls.size():
            raise ValueError(
                f"Invalid data length: {data} (must be {cls.size()})."
            )
        new_object = cls()
        data = int.from_bytes(data, "little")
        for field, field_type, bits in cls.serializable_fields:
            value = data % (1 << bits)
            data = data >> bits
            try:
                value = field_type(value)
            except ValueError:
                KLogger.warning(
                    f"Invalid field {field}: {value} (default constructor"
                    " returned value error, assigning value None)"
                )
                value = None
            setattr(new_object, str(field), value)
        return new_object
