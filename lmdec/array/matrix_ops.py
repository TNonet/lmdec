import time
from typing import Union, Optional, Tuple, TYPE_CHECKING

import numpy as np
from dask.array import broadcast_to
from dask.array.linalg import tsqr

from lmdec.array.types import ArrayType
from lmdec.array.wrappers.array_serialization import array_serializer
from lmdec.array.wrappers.time_logging import time_param_log

if TYPE_CHECKING:
    from lmdec.array.scaled import ScaledArray


@time_param_log
@array_serializer('x')
def subspace_to_SVD(x: ArrayType,
                    a: Optional["ScaledArray"] = None,
                    k: Optional[int] = None,
                    full_v: bool = False,
                    sqrt_s: bool = True,
                    log: int = 0) -> Union[Tuple[ArrayType, ArrayType, ArrayType],
                                           Tuple[ArrayType, ArrayType, ArrayType, dict]]:
    """Computes Truncated SVD of an array using an active subspace.

    Let A be a {N \times P} matrix
    Let x be a subspace of AA'

    U, S, V = subspace_to_SVD(x)
    USV.shape = (N, k)

    U, S, V = subspace_to_SVD(x, A, full_v=True)
    USV.shape == AA'.shape
    USV is low rank approximation to AA'

    Parameters
    ----------
    x : array_like, shape (N, K) or (N, )
        Active subspace of aa'
    a : array_like, shape (N, P), optional
        Array to be factored into USV from active subspace of x
    k : int
        Number of components of SVD to return
        1 <= k <= x.shape[1]
    full_v : bool
        Whether to return:
            V as a {K by K} matrix if full_v is False
            Or;
            V as a {K by P} matrix if full_v is True
                Requires a to be given as matrix
    sqrt_s : bool
        Whether to return the square root of the singular values of x.
    log : int >= 0
        Indicator in how many layers to log

    Returns
    -------
    u : (N, k) dask array
        Unitary array. Top k = min(K, k {if supplied}) left singular vectors of x
    s : (k, ) dask array
        Vector of of Top k = min(K, k {if supplied}) singular values in decreasing order
    v : (k, P) or (k, k) dask array
        Unitary array. Top k = min(K, k {if supplied}) right singular vectors of x.
        If full_v is True right singular vectors will be in (k, P)
        If full_v is false right singular vectors will be in (k, k)
    """
    flog = {}
    sub_log = max(log - 1, 0)
    U, S, V = tsqr(x, compute_svd=True)

    if full_v:
        dot_log = {'start': time.time()}
        x_t = a.T.dot(U)

        V, _, _ = tsqr(x_t, compute_svd=True)
        V = V.T
        dot_log['end'] = time.time()
        if sub_log:
            flog['dot'] = dot_log

    if sqrt_s:
        S = np.sqrt(S)

    if k:
        U, S, V = svd_to_trunc_svd(U, S, V, k=k)

    # U = U.rechunk({0: 'auto', 1: -1})
    # V = V.rechunk({0: 'auto', 1: -1})

    if log:
        return U, S, V, flog
    else:
        return U, S, V


def svd_to_trunc_svd(u: Optional[ArrayType] = None,
                     s: Optional[ArrayType] = None,
                     v: Optional[ArrayType] = None,
                     k: Optional[int] = None) -> Union[ArrayType, Tuple[ArrayType, ...]]:
    """Trims a full or partial SVD into a Truncated SVD with K Components

    Let X be an array of shape {n \times p}

    U, S, V = SVD(X)
    U of shape {n \times k}
    S of shape {k}
    V of shape {k \times p} where k = min(n, p)

    U, S, V = TruncSVD(X, k=10)
    U of shape {n \times k}
    S of shape {k}
    V of shape {k \times p}

    Parameters
    ----------
    u : array_like, shape (N, K)
        Left Singular Vectors of a matrix
    s : array_like, shape (K)
        Singular Values of a matrix
    v : array_like, shape (K, P)
        Right Singular Vectors of a matrix
    k : integer > 1
        Number of components in truncated SVD

    Returns
    -------
    uk: array_like, shape (N, k)
        Left Truncated Singular Vectors of a matrix
    s : array_like, shape (k)
        Truncated Singular Values of a matrix
    v : array_like, shape (k, P)
        Right Truncated Singular Vectors of a matrix
    """
    return_list = []

    if u is not None:
        u = u[:, :k]
        return_list.append(u)
    if s is not None:
        s = s[:k]
        return_list.append(s)
    if v is not None:
        v = v[:k, :]
        return_list.append(v)

    if len(return_list) == 1:
        return return_list[0]
    else:
        return (*return_list, )


def diag_dot(diag_array, x, return_diag=False):
    """Computes dot product between diag_array and x

    Parameters
    ----------
    diag_array : array_like, shape (K, ) or (K, 1)
                Diagonal entries of a diagonal maitrx
                [d1, d2, d3, ..., dk] -> [d1*e1, d2*e2, d2*e3, ..., dk*ek], where ei is the ith unit column vector
    x : array_like, shape (K, ...)
    return_diag : boolean
        If return_diag is True, return broadcasted array prepped for operation

    Returns
    -------
    out : array_like, shape of x
    """
    if len(x.shape) not in [1, 2]:
        raise ValueError("x must have (M, K) or (K, ). Current Shape = {}".format(x.shape))
    if diag_array.shape[0] != x.shape[0]:
        raise ValueError('shapes {} and {} not aligned: {} (dim 0 and 1) != {} (dim 0)'.format(diag_array.shape,
                                                                                               x.shape,
                                                                                               diag_array.shape[0],
                                                                                               x.shape[0]))
    if len(diag_array.shape) not in [1, 2]:
        raise ValueError('diag_array must have dimension (K, ) or (K, 1). Current shape = {}'.format(diag_array.shape))

    if len(x.shape) == 1:
        if len(diag_array.shape) == 2:
            d = np.squeeze(diag_array)
        else:
            d = diag_array
    else:
        if len(diag_array.shape) == 1:
            d = diag_array[:, np.newaxis]
        else:
            d = diag_array
        d = broadcast_to(d, x.shape)
    if return_diag:
        return d
    else:
        return np.multiply(d, x)
