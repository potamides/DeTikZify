from collections import defaultdict
from collections.abc import Callable
from copy import copy
from functools import cache, wraps
from typing import Any

def cache_cast(cast_func: Callable[..., Any]):
    """
    functools.cache which takes a user-defined function to convert arguments
    into something immutable so it can be cached.
    """
    def decorator(func):
        cache_args, cache_kwargs = None, None
        @cache
        def cached_func(_):
            return func(*cache_args, **cache_kwargs)
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            nonlocal cache_args, cache_kwargs
            cache_args, cache_kwargs = args, kwargs
            return cached_func(cast_func(*args, **kwargs))
        return wrapped_func
    return decorator

def cast(cls, object):
    clone = copy(object)
    clone.__class__ = cls
    return clone

# https://stackoverflow.com/a/12377059
def listify(fn=None, wrapper=list):
    """
    A decorator which wraps a function's return value in ``list(...)``.

    Useful when an algorithm can be expressed more cleanly as a generator but
    the function should return a list.

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

def batchify(fn=None):
    def batch(list_of_dicts):
        batch_dict = defaultdict(list)
        for d in list_of_dicts:
            for k, v in d.items():
                batch_dict[k].append(v)
        return batch_dict
    return listify(fn=fn, wrapper=batch)
