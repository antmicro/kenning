# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from kenning.core.exceptions import ClassProxyError
from kenning.datasets.magic_wand_dataset import MagicWandDataset
from kenning.utils import class_proxy


class TestClassProxy:
    class _Foo:
        def a(self, x, y, z):
            pass

        @staticmethod
        def _b():
            pass

        @property
        def c():
            pass

    def test_inherit(self):
        class Foo:
            def a():
                pass

        Proxy = class_proxy.inherit(Foo)
        p = Proxy()
        assert isinstance(p, Foo)

        # Making sure, that modifications to the dynamically created
        # inheriting class do not affect the original class.
        def foo():
            pass

        Proxy.b = foo
        assert foo == getattr(Proxy, "b")
        with pytest.raises(AttributeError):
            getattr(Foo, "b")

    def test_get_class_methods(self):
        assert set(["a", "_b", "c"]) == class_proxy.get_class_methods(
            self._Foo
        )

    def test_override_method(self):
        interceptor = Mock(return_value=14)
        FooPrim = class_proxy.inherit(self._Foo)
        class_proxy.inject_interceptor(FooPrim, "a", interceptor)
        fp = FooPrim()
        assert 14 == fp.a(2, 4, 7, kwargstest=98)
        assert 1 == interceptor.call_count
        call_args, _ = interceptor.call_args
        name, args, kwargs = call_args
        assert "a" == name
        assert (2, 4, 7) == args
        assert {"kwargstest": 98} == kwargs

    def test_override_method_does_not_exist(self):
        interceptor = Mock()
        FooPrim = class_proxy.inherit(self._Foo)
        with pytest.raises(ClassProxyError):
            class_proxy.inject_interceptor(
                FooPrim, "thismethoddoesnotexist", interceptor
            )

    def test_create_class_proxy(self):
        interceptor = Mock(return_value=14)
        Proxy = class_proxy.create_class_proxy(self._Foo, interceptor)
        p = Proxy()
        assert 14 == p.a(2, 5, keyword_argument="test")
        assert 1 == interceptor.call_count
        call_args, _ = interceptor.call_args
        name, args, kwargs = call_args
        assert "a" == name
        assert (2, 5) == args
        assert {"keyword_argument": "test"} == kwargs
        assert 14 == p._b(10)
        assert 2 == interceptor.call_count
        call_args, _ = interceptor.call_args
        name, args, kwargs = call_args
        assert "_b" == name
        assert (10,) == args
        assert {} == kwargs
        assert 14 == p.c
        assert 3 == interceptor.call_count
        call_args, _ = interceptor.call_args
        name, args, kwargs = call_args
        assert "c" == name
        assert () == args
        assert {} == kwargs
        assert 14 == p.fmsk
        assert 4 == interceptor.call_count
        call_args, _ = interceptor.call_args
        name, args, kwargs = call_args
        assert "__getattribute__" == name
        assert ("fmsk",) == args
        assert {} == kwargs
        p.fmsk = 9
        assert 5 == interceptor.call_count
        call_args, _ = interceptor.call_args
        name, args, kwargs = call_args
        assert "__setattr__" == name
        assert ("fmsk", 9) == args
        assert {} == kwargs

    def test_create_proxy_from_dataset(delf):
        interceptor = Mock(return_value=19)
        Proxy = class_proxy.create_class_proxy(MagicWandDataset, interceptor)
        p = Proxy()
        assert 19 == p.evaluate(2, 5, keyword_argument="test")
        assert 1 == interceptor.call_count
        call_args, _ = interceptor.call_args
        name, args, kwargs = call_args
        assert "evaluate" == name
        assert (2, 5) == args
        assert {"keyword_argument": "test"} == kwargs
        assert 19 == p.indices
        assert 2 == interceptor.call_count
