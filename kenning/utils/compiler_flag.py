# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Class for easily handling compiler flags in optimizers.
"""
from typing import List


class CompilerFlag:
    """
    Basic class for representing compiler flags.
    """

    def __init__(self, flag: "CompilerFlag | str"):
        """
        Create a new compiler flag. This can be a single flag
        or a key-value pair.

        Parameters
        ----------
        flag: CompilerFlag | str
            The flag to be parsed.
        """
        if isinstance(type(flag), str):
            self.key = flag.key
            self.value = flag.value

        else:
            if "=" in flag:
                self.key, self.value = flag.split("=", 1)
            else:
                self.key = flag
                self.value = None

    def __str__(self):
        if self.value is None:
            return self.key
        else:
            return f"{self.key}={self.value}"

    def is_same_flag(self, other) -> bool:
        """
        Return whether the flags have the same key.
        """
        return self.key == other.key

    def has_same_flag(self, others: "List[CompilerFlag]") -> bool:
        """
        Determine whether there are flags that are similar
        to the current one's key.
        """
        return any(self.is_same_flag(x) for x in others)


def merge_compiler_flags(
    user_flags: List[CompilerFlag | str],
    default_flags: List[CompilerFlag | str],
    tostring: bool = False,
) -> List[CompilerFlag]:
    """
    Merge two lists of compiler flags together.

    Parameters
    ----------
    user_flags: List[CompilerFlag | str]
        List of flags set by the user. This takes precedence in
        case of key conflict.

    default_flags: List[CompilerFlag | str]
        List of flags default in kenning. In case of missing user flags,
        this serves as the default.

    tostring: bool
        Automatically convert to a string that can be passed to
        a subprocess

    Returns
    -------
    List[CompilerFlag]
        Merged list of the flags with respect to keys.
    """
    # Copy each flag, do not modify original list.
    flags = [CompilerFlag(flag) for flag in user_flags]
    for def_flag in default_flags:
        def_flag = CompilerFlag(def_flag)
        if not def_flag.has_same_flag(flags):
            flags.append(def_flag)

    if tostring:
        return " ".join(map(str, flags))

    return flags
