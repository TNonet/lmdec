from typing import Tuple, Union

import dask.array as da
import numpy as np
from dask.array.linalg import tsqr
from dask.array.random import RandomState

from lmdec.array.random import array_split
from lmdec.array.scaled import ScaledArray
from lmdec.array.wrappers.time_logging import time_param_log


@time_param_log
def v_init(array: ScaledArray, v: Union[da.Array, np.ndarray], log: int = 0) -> Union[da.Array, Tuple[da.Array, dict]]:
    """Computes estimation of left Eigenvectors of `a` from an initial guess `v`

    Parameters
    ----------
    array : ScaledArray
        array of which the left eigenvectors will be estimated
    v : array_like
        Initial guess of right right hand eigenvectors of `array`.
        Usually will come from a SVD of a subarray of `array`
    log : bool
        See logging in `time_param_log`

    Returns
    -------
    U : Dask Array
        estimation of left Eigenvectors of `array`
    """
    x_k = array.dot(v)

    U, _, _ = tsqr(x_k, compute_svd=True)

    # u = u.rechunk('auto')

    if log:
        return U, {}
    else:
        return U


@time_param_log
def sub_svd_init(array: ScaledArray,
                 k: int,
                 warm_start_row_factor: float = 5,
                 seed: int = 42,
                 log: int = 1) -> Union[da.core.Array, Tuple[da.core.Array, dict]]:
    """Attempts to compute a better approximation of the top k left eigenvectors of a matrix using a sample of the
    rows of that matrix.

    Parameters
    ----------
    array : ScaledArray
        array of which the top `k` left eigenvectors will be estimated
    k : int
        number of components to estimate
    warm_start_row_factor : float
        multiplier for the number of rows of `array` to sample.

        number of rows = k * warm_start_row_factor
    seed : int
        random seed for which rows to select from array
    log : bool
        See logging in `time_param_log`

    Returns
    -------
    U : Dask Array
        estimation of left Eigenvectors of `array`

    Notes
    -----
    Shuffle A so that the desired rows are on top

    A.shape = (m, n)

    A = [A_start,
        A_rest]

    A_start.shape = (k, n)
    A_rest.shape = (m-k, n)

    U, S, V' <- SVD(A_start.T)

    U.shape <- (n, m_start)

    return top k columns of U[:, 0:k].
    """
    sub_log = max(log - 1, 0)

    rows = warm_start_row_factor * k
    n, p = array.shape
    row_fraction = rows / n

    I, _ = array_split(array_shape=array.shape, f=row_fraction, axis=0, seed=seed, log=0)

    sub_array = array[I, :].T
    sub_array = sub_array.rechunk({0: -1, 1: 'auto'})
    _sub_svd_return = _sub_svd(array, sub_array, k=k, log=sub_log)

    if sub_log:
        U, _sub_svd_log = _sub_svd_return
        flog = {_sub_svd.__name__: _sub_svd_log}
    else:
        flog = {}
        U = _sub_svd_return

    if log:
        return U, flog
    else:
        return U


@time_param_log
def _sub_svd(array: "ScaledArray",
             sub_array: "ScaledArray",
             k: int = 5,
             log: int = 1) -> Union[da.core.Array,
                                    Tuple[da.core.Array, dict]]:
    """Helper function for computing SVD of `sub_array`

    Parameters
    ----------
    array : ScaledArray
        array of which the top `k` left eigenvectors will be estimated
    sub_array : ScaledArray
        array of which the top `k` left eigenvectors will be calculated
    k : int
        number of eigenvectors of `sub_array` to compute
    log : bool
        See logging in `time_param_log`

    Returns
    -------
    U : Dask Array
        estimation of left Eigenvectors of `array`
    """
    # VSU' <--- SVD of A'
    V, _, _ = tsqr(sub_array.array, compute_svd=True)  # SVD of A' -> VSU'
    U = v_init(array, V[:, :k])

    U = U.rechunk({0: 'auto', 1: -1})
    if log:
        return U, {}
    else:
        return U


@time_param_log
def rnormal_start(array: ScaledArray,
                  k: int,
                  seed: int = 42,
                  log: int = 1) -> Union[da.core.Array,
                                         Tuple[da.core.Array, dict]]:
    """Initializes a gaussian normal matrix of the size (N, K)

    Where array is of size (N, P)

    Parameters
    ----------
    array : ScaledArray
        array of which shape will be used to allow for Power Iteration
    k : int
        number of columns in `omega`
    seed : int
        seed to set random generator
   log : bool
        See logging in `time_param_log`

    Returns
    -------
    omega : Dask Array
        0th iteration of power Method
    """
    m, n = array.shape
    state = RandomState(seed)

    omega = state.standard_normal(
        size=(m, k), chunks=(array.chunks[0], (k,)))

    if log:
        return omega, {}
    else:
        return omega
