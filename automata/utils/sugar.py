"""
Author(s): Craig fouts
License: Apache 2.0 license
"""

from functools import singledispatch, wraps
from inspect import signature
from ._utils import get_kwargs, get_methods

@singledispatch
def attrmethod(method):
    @wraps(method)
    def wrapper(cls, *args, **kwargs):
        (bound := signature(method).bind(cls, *args, **kwargs)).apply_defaults()
        method_kwargs = {k: v for k, v in bound.arguments.items() if k != 'self'}

        for key, val in method_kwargs.items():
            if val is not None or not hasattr(cls, key):
                setattr(cls, key, val)

        return method(cls, *args, **kwargs)
    return wrapper

@attrmethod.register(str)
def _(prefix='', suffix=''):
    def decorator(method):
        @wraps(method)
        def wrapper(cls, *args, **kwargs):
            (bound := signature(method).bind(cls, *args, **kwargs)).apply_defaults()
            method_kwargs = {k: v for k, v in bound.arguments.items() if k != 'self'}

            for key, val in method_kwargs.items():
                key = prefix + key + suffix

                if val is not None or not hasattr(cls, key):
                    setattr(cls, key, val)

            return method(cls, *args, **kwargs)
        return wrapper
    return decorator

@singledispatch
def buildmethod(method):
    @wraps(method)
    def wrapper(cls, *args, **kwargs):
        kwargs.update(zip(method.__code__.co_varnames, (cls, *args)))
        builders = get_methods(cls, '_build')

        for builder in filter(lambda x: x != method.__name__, builders):
            build = getattr(cls, builder)
            build_kwargs = get_kwargs(build, **kwargs)
            update = build(**build_kwargs)

            if update is not None:
                kwargs.update(update)

        return method(**kwargs)
    return wrapper

@buildmethod.register(str)
def _(*builders):
    def decorator(method):
        @wraps(method)
        def wrapper(cls, *args, **kwargs):
            kwargs.update(zip(method.__code__.co_varnames, (cls, *args)))

            for builder in filter(lambda x: hasattr(cls, x), builders):
                build = getattr(cls, builder)
                build_kwargs = get_kwargs(build, **kwargs)
                update = build(**build_kwargs)

                if update is not None:
                    kwargs.update(update)

            return method(**kwargs)
        return wrapper
    return decorator
