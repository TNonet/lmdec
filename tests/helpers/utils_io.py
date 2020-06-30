from itertools import product
from typing import Tuple, Generator

import numpy as np
from dask import array as da
from sparse import COO

from lmdec.array.chained import ChainedArray
from lmdec.array.stacked import StackedArray


def tree_array(current_depth: int = 0, max_depth: int = 4, sparse_prob: float = .33, stacked_chained_prob: float = .33,
               size: Tuple[int, ...] = (5, 5)):
    if any(not (0 <= x <= 1) for x in [sparse_prob, stacked_chained_prob]) or sparse_prob + stacked_chained_prob > 1:
        raise ValueError(f'expected valid probability values, but got {(sparse_prob, stacked_chained_prob)}')

    if current_depth == max_depth:
        # Prevent Stacked or Chained Array from being generate if at max_depth
        sparse_prob += stacked_chained_prob / 2
        stacked_chained_prob = 0

    if len(size) == 1 or max(size) == np.product(size):
        # Prevents sparse array from being degenerate
        sparse_prob = 0
        stacked_chained_prob = 0

    dense_prob = 1 - sparse_prob - stacked_chained_prob

    i = np.random.random()

    if i <= dense_prob:
        return da.random.random(size)
    if i <= dense_prob + sparse_prob:
        return random_sparse(size)
    else:
        return random_delayedarray(current_depth=current_depth, max_depth=max_depth, sparse_prob=sparse_prob,
                                   stacked_chained_prob=stacked_chained_prob, size=size)


def random_delayedarray(current_depth: int = 0, max_depth: int = 4, sparse_prob: float = .33,
                        stacked_chained_prob: float = .33, size: Tuple[int, ...] = (5, 5)):
    if np.random.random() <= 0.5:
        # Generate StackedArray
        array_type = StackedArray
        if len(size) == 1 or (len(size) == 2 and product(size) == max(size)):
            # This covers cases of a (1D array or a 1D array with a degenerate axis)
            # (P, ) or (N, 1) or (1, P)
            shapes = [size for _ in range(4)]
        else:
            n, p = size
            shapes = [(n, p), (n, 1), (p,), (1, p)]

    else:
        array_type = ChainedArray
        if len(size) == 1:
            n, p = size[0], size[0]
        else:
            n, p = size

        shapes = generate_chained_shapes((n, p))

    return array_type([tree_array(current_depth=current_depth + 1, max_depth=max_depth, sparse_prob=sparse_prob,
                                  stacked_chained_prob=stacked_chained_prob, size=s) for s in shapes])


def generate_chained_shapes(shape: Tuple[int, int], depth=4,
                            max_size: int = 10) -> Generator[Tuple[int, ...], None, None]:
    """

    Parameters
    ----------
    shape
    depth
    max_size

    Returns
    -------

    """
    n, p = shape

    if depth < 2:
        raise ValueError(f'expected depth to be greater or equal to 2, but got {depth}')

    if np.random.random() > 0.5:
        rhs = n
        yield (n,)
    else:
        rhs = np.random.randint(2, max_size)
        yield (n, rhs)

    for i in range(depth - 2):
        if np.random.random() > 0.5:
            yield (rhs,)
        else:
            lhs = rhs
            rhs = np.random.randint(2, max_size)
            yield (lhs, rhs)

    if rhs == p and np.random.random() > 0.5:
        yield (p,)
    else:
        yield (rhs, p)


def random_sparse(size: Tuple[int, ...]):
    """ Generates a random sparse matrix of shape 'size'

    Parameters
    ----------
    size : tuple of int

    Returns
    -------
    sparse : array_like, shape size)

    """
    nnz = int(np.random.random() * np.product(size))

    data = np.random.random(nnz)

    dimns = [np.arange(0, s) for s in size]

    coords = np.array(list(product(*dimns)))

    coords_index = np.arange(0, np.product(size))
    coords_index = np.random.choice(coords_index, nnz, replace=False)
    coords_index = list(sorted(coords_index))

    coords = coords[coords_index].T

    return da.from_array(COO(coords, data=data, shape=size, has_duplicates=False, sorted=True, fill_value=0),
                         chunks=size)