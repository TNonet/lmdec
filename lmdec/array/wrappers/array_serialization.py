import numpy as np
import dask.array as da
from inspect import getfullargspec, signature


class array_serializer:
    """Ensures functions that utilize dask operations still return proper types.

    def some_function(x: Union[ndarray, Array]) -> Union[ndarray, Array]:
        # Some Dask Operations
        y: dask.array.core.Array = dask.array.operation(x)
        ...
        return y

    If some_function is called with a numpy array return a numpy array
        Some array types cannot work with dask arrays

    If some_function is called with a dask array return a dask array
    """

    def __init__(self, array_arg: str):
        """
        :param array_arg: str, that matches the function definition f.

            @array_serializer(x)
            def some_function(array, x, ...):
                ...
                return y

            #Y will have same type as input argument x

        """
        self.array_arg = array_arg

    def __call__(self, f):

        def wrapped_f(*args, **kwargs):
            """
            Ensure that when:
                f(*args, **kwargs)
            is called that the returned array types match the type of array_arg.

            2 Cases (Assuming only returning 1 array):
                1. f(*args, **kwargs) returns X array, array_arg is Y
                    return Y(f(*args, **kwargs))
                2. f(*args, **kwargs) returns numpy array, array_arg is numpy
                    return f(*args, **kwargs)
            """
            arg_list = getfullargspec(f).args
            array_arg_location = arg_list.index(self.array_arg)
            arg_type = type(args[array_arg_location])
            if arg_type is np.ndarray:
                arg_func = np.array
            elif arg_type is da.core.Array:
                arg_func = da.array
            else:
                raise Exception('Strange Array Type')

            return_object = f(*args, *kwargs)

            if isinstance(return_object, (list, tuple)):
                return_object = list(return_object)
                for i in range(len(return_object)):
                    if isinstance(return_object[i], (da.core.Array, np.ndarray)) \
                            and not isinstance(return_object[i], arg_type):
                        return_object[i] = arg_func(return_object[i])
                return (*return_object,)
            else:
                if isinstance(return_object, (da.core.Array, np.ndarray)) and not isinstance(return_object, arg_type):
                    return_object = arg_func(return_object)
                return return_object

        wrapped_f.__signature__ = signature(f)

        return wrapped_f
