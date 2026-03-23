# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from kenning.utils.compiler_flag import CompilerFlag, merge_compiler_flags


class TestCompilerFlags:
    def test_compilerflag_str_and_parsing(self) -> None:
        f1 = CompilerFlag("-O2")
        assert f1.key == "-O2"
        assert f1.value is None
        assert str(f1) == "-O2"

        f2 = CompilerFlag("-flag1=value1=value2")
        assert f2.key == "-flag1"
        assert f2.value == "value1=value2"
        assert str(f2) == "-flag1=value1=value2"

        f3 = CompilerFlag(f2)
        assert f3.key == "-flag1" and f3.value == "value1=value2"

    def test_compilerflag_comparisons(self) -> None:
        a = CompilerFlag("-mcpu=cortex-m4")
        b = CompilerFlag("-mcpu=cortex-m4-r")
        c = CompilerFlag("-O3")
        assert a.is_same_flag(b)
        assert not a.is_same_flag(c)
        assert a.has_same_flag([b, c])
        assert not c.has_same_flag([a])

    def test_merge_compiler_flags_none_and_types(self) -> None:
        merged = merge_compiler_flags(None, None, tostring=False)
        assert merged == []

        merged = merge_compiler_flags(["-a"], None, tostring=False)
        assert set(map(str, merged)) == {"-a"}

        merged = merge_compiler_flags(None, ["-b=2"], tostring=False)
        assert set(map(str, merged)) == {"-b=2"}

        s = merge_compiler_flags(["-x"], ["-y=1"], tostring=True)
        assert isinstance(s, str)
        assert set(s.split()) == {"-x", "-y=1"}

    def test_merge_prefers_user_over_default_behavior(self) -> None:
        user = ["-mcpu=cortex-m4"]
        default = ["-mcpu=cortex-m4", "-O3"]
        merged = merge_compiler_flags(user, default, tostring=False)

        assert set(map(str, merged)) == {"-mcpu=cortex-m4", "-O3"}

    def test_merge_type_errors(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            merge_compiler_flags("not-a-list", ["-a"])

        with pytest.raises(TypeError):
            merge_compiler_flags(["-a"], "not-a-list")
