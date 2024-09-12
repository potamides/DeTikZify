from collections.abc import Callable
from functools import cache, wraps
from typing import Any

def cast_cache(cast_func: Callable[..., Any]):
    """
    functools.cache which takes a user-defined function to convert arguments
    into something immutable so it can be cached.
    """
    def decorator(func):
        # https://stackoverflow.com/a/19319626
        class MethodDecoratorAdapter:
            """
            Allows you to use the same decorator on methods and functions,
            hiding the self argument from the decorator.
            """
            def __init__(self, func):
                self.func = func
                self.is_method = False

            @cache
            def cast_func(self, _):
                if self.is_method:
                    return self.func(self.instance, *self.args, **self.kwargs)
                else:
                    return self.func(*self.args, **self.kwargs)

            def __get__(self, instance, _):
                if not self.is_method:
                    self.is_method = True
                self.instance = instance
                return self

            def __call__(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
                return self.cast_func(cast_func(*args, **kwargs))

        return wraps(func)(MethodDecoratorAdapter(func))

    return decorator

# https://stackoverflow.com/a/12377059
def listify(fn=None, wrapper=list):
    """
    A decorator which wraps a function's return value in ``list(...)``.

    Useful when an algorithm can be expressed more cleanly as a generator but
    the function should return an list.

    Example::

        >>> @listify
        ... def get_lengths(iterable):
        ...     for i in iterable:
        ...         yield len(i)
        >>> get_lengths(["spam", "eggs"])
        [4, 4]
        >>>
        >>> @listify(wrapper=tuple)
        ... def get_lengths_tuple(iterable):
        ...     for i in iterable:
        ...         yield len(i)
        >>> get_lengths_tuple(["foo", "bar"])
        (3, 3)
    """
    def listify_return(fn):
        @wraps(fn)
        def listify_helper(*args, **kw):
            return wrapper(fn(*args, **kw))
        return listify_helper
    if fn is None:
        return listify_return
    return listify_return(fn)
