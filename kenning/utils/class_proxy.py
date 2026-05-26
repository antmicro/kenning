# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A set of functions, that allow for intercepting interactions with objects of a
given class, and routing them to pre-defined functions.

Intended use case is to automatically create proxy objects for various classes,
and that way reroute, control or change method calls, property/field accesses,
and field writes.

This can be utilized in automated dependency injection, in order to route calls
to a correct class instance, or through a different communication channel.
"""

import inspect
from typing import Any, Callable, Dict, Set, Tuple

from kenning.core.exceptions import ClassProxyError

InterceptorType = Callable[Tuple[str, Tuple[Any], Dict[str, Any]], Any]


def inherit(_class: type) -> type:
    """
    Dynamically creates inheriting classes.

    Parameters
    ----------
    _class: type
        Base class for the inheritance.

    Returns
    -------
    type
        Class inheriting from '_class', and identical to it.
    """
    return type(f"{_class.__name__}_Proxy", (_class,), {})


def get_class_methods(_class: type) -> Set[str]:
    """
    Gets all methods and properties of a class.

    Parameters
    ----------
    _class: type
        The class to check.

    Returns
    -------
    Set[str]
        A set of all method and property names in '_class'.
    """
    return set(
        filter(
            lambda x: (
                inspect.isfunction(getattr(_class, x))
                or isinstance(getattr(_class, x), property)
            ),
            dir(_class),
        )
    )


def inject_interceptor(
    _class: type, method_name: str, interceptor: InterceptorType
):
    """
    Sets up an 'interceptor' for a given method or property in a given class.
    The interceptor is a function that will be called each time that specific
    method or property is invoked on any object of that class.

    The original method or property will be completely removed from the class,
    so a call to the interceptor will take place instead of the method/property
    call).

    Parameters
    ----------
    _class: type
        The class to check.
    method_name: str
        Name of the method or property to intercept.
    interceptor: InterceptorType
        A function, that takes 3 parameters: name of the intercepted method or
        property, all positional arguments of the method/property call (except
        for the first 'self' argument when applicable), and all keyword
        arguments of the method/property call.

    Raises
    ------
    ClassProxyError
        Raised when attempting to intercept calls to a non-existent method or
        property.
    """
    if method_name not in get_class_methods(_class):
        raise ClassProxyError(
            f"Attempted to override non-existent method {method_name} in"
            f" {_class}."
        )
    is_property = isinstance(getattr(_class, method_name), property)

    def stub(*args, **kwargs: Any):
        if is_property:
            # Since there is no static properties in Python, the first
            # argument will always be 'self', therefore we need to discard it.
            args = args[1:]
        return interceptor(method_name, args, kwargs)

    if is_property:
        stub = property(stub)
    else:
        # We are wrapping this in 'staticmethod' because we don't want our
        # 'stub' method to get the 'self' argument (it would be useless, since
        # 'self' would be a reference to the proxy).
        stub = staticmethod(stub)
    setattr(_class, method_name, stub)


def inject_interceptors(_class: type, interceptor: InterceptorType):
    """
    Sets up an 'interceptor' function for the given class.

    Any method call or property access will result in the interceptor being
    called, and name of that method/property will be passed as a parameter to
    the interceptor.

    An assignment to any attribute, that is not a method or property will be
    registered as a __setattr__ method call. Therefore interceptor will be
    called and the string '__setattr__' will be passed as a parameter to the
    interceptor.

    A read from any attribute, that is not a method or property will be
    registered as a __getattribute__ call. Therefore interceptor will be
    called and the string '__getattribute__' will be passed as a parameter to
    the interceptor.

    Parameters
    ----------
    _class: type
        The class to check.
    interceptor: InterceptorType
        A function, that takes 3 parameters: name of the intercepted method or
        property, all positional arguments of the method/property call (except
        for the first 'self' argument when applicable), and all keyword
        arguments of the method/property call.
    """

    def getattribute_intercept(self, attribute_name: str):
        not_method = attribute_name not in get_class_methods(_class)
        is_property = not_method is False and isinstance(
            getattr(_class, attribute_name), property
        )

        def stub(*args, **kwargs: Any):
            if not_method:
                args += (attribute_name,)
            return interceptor(
                "__getattribute__" if not_method else attribute_name,
                args,
                kwargs,
            )

        stub = staticmethod(stub)
        if is_property or not_method:
            return stub()
        return stub

    _class.__getattribute__ = getattribute_intercept

    def setattr_intercept(self, attribute_name, value):
        return interceptor("__setattr__", (attribute_name, value), {})

    _class.__setattr__ = setattr_intercept


def create_class_proxy(_class: type, interceptor: InterceptorType) -> type:
    """
    Creates a class, that inherits from the given class, and has all the same
    methods and properties, but any interaction with the class is routed to the
    specified 'interceptor' function.

    Any method call or property access will result in the interceptor being
    called, and name of that method/property will be passed as a parameter to
    the interceptor.

    An assignment to any attribute, that is not a method or property will be
    registered as a __setattr__ method call. Therefore interceptor will be
    called and the string '__setattr__' will be passed as a parameter to the
    interceptor.

    A read from any attribute, that is not a method or property will be
    registered as a __getattribute__ call. Therefore interceptor will be
    called and the string '__getattribute__' will be passed as a parameter to
    the interceptor.

    Parameters
    ----------
    _class: type
        The class to check.
    interceptor: InterceptorType
        A function, that takes 3 parameters: name of the intercepted method or
        property, all positional arguments of the method/property call (except
        for the first 'self' argument when applicable), and all keyword
        arguments of the method/property call.

    Returns
    -------
    type
        A class inheriting from '_class', that routes all method/property calls
        to the given interceptor function.
    """
    proxy = inherit(_class)
    inject_interceptors(proxy, interceptor)

    def default_constructor(self): ...

    proxy.__init__ = default_constructor
    return proxy
