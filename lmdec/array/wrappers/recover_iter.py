from decorator import decorator
from warnings import warn


@decorator
def recover_last_value(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except KeyboardInterrupt as e:
        _self = args[0]
        if _self.last_value is not None:
            warn("Captured `KeyboardInterrupt`. Exiting with last value")
            return _self.last_value
        else:
            warn('No iterations were complete.')
            raise KeyboardInterrupt from e
