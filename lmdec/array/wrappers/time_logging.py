import time
from decorator import decorator
from inspect import getfullargspec
from typing import Tuple, Any, Union


@decorator
def time_param_log(func, *args, **kwargs) -> Union[Tuple[Any, ...], Any]:
    """
    Wrapper to record:
        start time
        end times
        non-default parameters


    Parameters
    ----------
    func : Callable
           Function to wrap around.
           Must have optional parameter log

    Returns
    -------
    r : Any
        function result

    flog: dict
          function execution logging

    Notes
    -----
    Default is always to log, unless specified otherwise
    https://stackoverflow.com/questions/34832573/python-decorator-to-display-passed-and-default-kwargs
    """

    argspec = getfullargspec(func)
    non_defaults = len(args) - len(argspec.defaults)
    defaults_arg_names = argspec.args[non_defaults:]
    params = dict(zip(defaults_arg_names, args[non_defaults:]))

    if params['log'] < 0:
        raise Exception("Logging must be non-negative integer.")
    elif params['log']:
        params = {k: (v if isinstance(v, (float, str, int, bool)) else str(v)) for k, v in params.items()}
        wrapper_log = {'start': time.time(), 'params': params}
        return_tuple = func(*args, **kwargs)
        wrapper_log['end'] = time.time()

        if len(return_tuple) == 2:
            return_object, flog = return_tuple
        else:
            return_object, flog = return_tuple[0:-1], return_tuple[-1]
        flog.update(wrapper_log)

        if isinstance(return_object, tuple):
            return (*return_object, flog)
        else:
            return return_object, flog
    else:
        return func(*args, **kwargs)
