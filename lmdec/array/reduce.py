from typing import Union, List, Generator, Callable

import dask

import numpy as np


def tree_reduction(items: Union[List[dask.array.core.Array],
                                Generator[dask.array.core.Array, None, None]],
                   binary_operator: Callable[[dask.array.core.Array, dask.array.core.Array],
                                             dask.array.core.Array]) -> dask.array.core.Array:
    items = list(items)

    while len(items) > 1:
        reduced_items = []
        for i in range(0, len(items) - 1, 2):
            i, j = items.pop(), items.pop()  # pop() returns the last item but python executes left to right.
            item = binary_operator(j, i)  # Therefore, we must reverse the order to keep the tree stable.
            reduced_items.append(item)
        items = items + list(reversed(reduced_items))  # must flip order of new `reduced_items` for tree stability

    return items[0]


def linear_reduction(items: Union[List[dask.array.core.Array],
                                  Generator[dask.array.core.Array, None, None]],
                     binary_operator: Callable[[dask.array.core.Array, dask.array.core.Array],
                                               dask.array.core.Array]) -> dask.array.core.Array:
    if isinstance(items, list):
        items = iter(items)

    reduced = next(items)
    for item in items:
        reduced = binary_operator(reduced, item)
    return reduced


def matrix_chain(items: Union[List[dask.array.core.Array],
                              Generator[dask.array.core.Array, None, None]],
                 binary_operator: Callable[[dask.array.core.Array, dask.array.core.Array],
                                           dask.array.core.Array]) -> dask.array.core.Array:
    """ TODO: Implement efficient chain for multiplication

    Parameters
    ----------
    items
    binary_operator

    Returns
    -------

    Notes
    -----
    Source:
        http://www.personal.kent.edu/~rmuhamma/Algorithms/MyAlgorithms/Dynamic/chainMatrixMult.htm

    """
    raise NotImplementedError

    items = list(items)
    n = len(items)
    m = np.zeros((n, n))

    for l in range(2, n):
        for i in range(n - l + 1):
            j = i + l - 1
            m[i, j] = float('inf')
            for k in range(i, j-1):
                pass
                # q = m[i, k] + m[k + 1, j] + items[i-1]*items[k]*items[j]