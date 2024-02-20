from functools import cache, wraps

def cast_cache(cast_func, cast_back_func=lambda obj: obj):
    """
    functools.cache which takes a user-defined function to convert arguments
    into something immutable so it can be cached.
    """
    def decorator(f):
        @cache
        def cast_f(*args, **kwargs):
            return f(
            *map(cast_back_func, args),
            **dict(zip(kwargs.keys(), map(cast_back_func, kwargs.values())))
            )
        @wraps(f)
        def wrapper(*args, **kwargs):
            return cast_f(
                *map(cast_func, args),
                **dict(zip(kwargs.keys(), map(cast_func, kwargs.values())))
            )
        return wrapper
    return decorator
