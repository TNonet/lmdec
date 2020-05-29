from decorator import decorator
from warnings import warn


@decorator
def recover_last_value(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except KeyboardInterrupt as e:
        _self = args[0]
        if _self.history.iter['last_value']:
            if _self.warn:
                warn("Captured `KeyboardInterrupt`. Exiting with last value")
            return _self.history.iter['last_value']
        else:
            if _self.warn:
                warn('No iterations were complete.')
            raise KeyboardInterrupt from e
