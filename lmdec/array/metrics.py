import typing
from typing import Tuple, Union, Callable

import dask.array as da
import numpy as np

from lmdec.array.random import array_split
from lmdec.array.transform import acc_format_svd
from lmdec.array.types import ArrayType, LargeArrayType
from lmdec.array.wrappers.time_logging import time_param_log

if typing.TYPE_CHECKING:
    from lmdec.array.scaled import ScaledArray


def approx_array_function(array_func: Callable,
                          p: float,
                          log: int = 1) -> Callable:
    @time_param_log
    def _f(array, log=log):
        flog = {}
        sub_log = max(log - 1, 0)  # Logging reflect number of sub function calls to record

        return_tuple = array_split(array.shape, f=p, axis=1, log=sub_log)
        if sub_log:
            index_set, _, array_split_log = return_tuple
            flog[array_split.__name__] = array_split_log
        else:
            index_set, _ = return_tuple

        array1 = array[index_set, :]

        if log:
            return array_func(array1), flog
        else:
            return array_func(array1)

    return _f


@time_param_log
def subspace_dist(vi: ArrayType,
                  vj: ArrayType,
                  s: ArrayType,
                  power: float = 2,
                  epsilon: float = 1e-6,
                  log: int = 0) -> Union[float, Tuple[float, dict]]:
    """Returns Weighted Distance between two subspaces

    =  1 - <l, s**pow>/sum(s**pow)

    Let:
        S(A) returns ordered vector of singular values, [l_1, l_2, ..., l_m] of A.
            where dim(A) = (p, q) and m = min(p, q).

    Parameters
    ----------
    vi : array_like, shape (N, K)
         V subspace 1
    vj : array_like, shape (N, K)
         V Subspace 2
    s : array_list, shape (K,)
        Singular Values corresponding to V Subspace 1
    power : Numeric
            Power to raise Singular Values, s, to weight larger values more.

    log

    Returns
    -------
    d : float
        The distance between vi and vj weighted by s
    """
    (ni, ki) = vi.shape
    (nj, kj) = vj.shape

    if ni == len(s):
        vi = vi.T
    if nj == len(s):
        vj = vj.T

    if vi.shape[0] != vj.shape[0]:
        raise ValueError("Shape Error between vi, {},  and vj, {}".format(vi.shape, vj.shape))

    _, l, _ = np.linalg.svd(vi.T.dot(vj))

    if max(l) > 1 + 1e-6 or min(l) < -1e-6:
        raise ValueError('Norm of vi or vj was not 1.')

    if power in [np.float('inf'), -1]:
        # Special cases
        s_index = np.argmin(s) if power == -1 else np.argmax(s)
        s_value = s[s_index]
        s = np.zeros_like(s)
        s[s_index] = s_value
        power = 1

    weighted_cos_dist = np.squeeze((s**power).dot(l))
    d = 1 - weighted_cos_dist/(np.sum(s**power) + epsilon)

    try:
        d = d.compute()
    except AttributeError:
        pass

    if log > 0:
        return d, {}
    else:
        return d


@time_param_log
def rmse_k(array: Union[LargeArrayType, "ScaledArray"],
           u: ArrayType,
           s: ArrayType,
           format_us: bool = True,
           log: int = 0) -> Union[float, Tuple[float, dict]]:
    """Computes RMSE_k Norm

    sqrt((1/nk)*(||(1/m)*AA'u - us||_{F})^2)

    Parameters
    ----------
    array : array_like, shape (N, P)
            Array being decomposed
    u : array_like, shape (N, K)
        Top K right singular vectors of array
    s : array_like, shape (K, )
        Top K singular values of array
    format_us : bool
                Whether or not to format U, S

    Returns
    -------
    d : float
        The root mean squared error for the top k right singular vectors and singular values

    Notes
    -----
    Assumes data went through acc_format_svd(..., square=True).

    Therefore -> s_i = (s_i)^2

    sqrt((1/nk) * sum(||(1/m)A'Au_i - u_i*s_i|| for i = 1, ..., k))

    Where ||.|| refers to (||.||_2)^2

    This is equivalent to

    sqrt((1/nkm)*(||A'Au - us||_{F})^2)
    """

    n, p = array.shape
    _, k = u.shape

    if format_us:
        u, s = acc_format_svd(u, s, array.shape, square=False)

    try:
        aatx = array.sym_mat_mult(u)
    except AttributeError:
        aatx = array.dot(array.T.dot(u))

    acc = np.linalg.norm(aatx - u.dot(s), ord='fro')
    acc /= np.sqrt(n*k*p)

    try:
        acc = acc.compute()
    except AttributeError:
        pass

    if log > 0:
        return acc, {}
    else:
        return acc


@time_param_log
def q_value_converge(si: ArrayType,
                     sj: ArrayType,
                     scale: bool = True,
                     norm: float = 2,
                     log: int = 0) -> Union[float, Tuple[float, dict]]:
    """
    Computes distance between si and sj

    Let step be an iterative generator that converges to x*.

    x_k = step(x_{k-1})

    {x_0, x_1, ..., x_k-1, x_k, x_k+1, ... x*}

    Quotient Accuracy is:

        ||x_k - x_{k-1}||/C ,

        Where:
            || . || is norm specified
            if scale is True:
                C = (||x_k|| + ||x_{k-1}||)/2
            else:
                C = 1

    Parameters
    ----------
    si : array_like, shape (K, )
         Singular Vectors at time i
    sj : array_like, shape (K, )
         Singular Vectors at time i+i = j
    scale : boolean
            Whether to scale the value by the average norm of si ans sj
    norm : numeric value that reflects which norm to use
           See https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
           Norms for Vectors
    log : int >= 0

    Returns
    ------
    d : float
    """

    if si.shape != sj.shape:
        raise ValueError('Shapes of si and sj must match. Current {} != {}'.format(si.shape, sj.shape))

    acc = da.linalg.norm(si - sj, norm)

    if scale:
        acc /= (da.linalg.norm(si, norm) + da.linalg.norm(sj, norm))/2

    try:
        acc = acc.compute()
    except AttributeError:
        pass

    if log:
        return acc, {}
    else:
        return acc
