from typing import Tuple, Union, Optional

import dask
import dask.array as da
import numpy as np
from dask.array.linalg import tsqr
from dask.array.random import RandomState

from lmdec.array.wrappers.time_logging import time_param_log
from lmdec.array.matrix_ops import svd_to_trunc_svd, subspace_to_SVD
from lmdec.array.scaled import ScaledArray
from lmdec.array.random import array_split
from lmdec.array.metrics import q_value_converge


@time_param_log
def v_init(a: ScaledArray,
           v: Union[da.Array, np.ndarray],
           s: Optional[Union[da.Array, np.ndarray]] = None,
           log: int = 0) -> Union[da.Array, Tuple[da.Array, dict]]:
    """

    Parameters
    ----------
    a
    v
    s:
    log

    Returns
    -------

    """
    x_k = a.dot(v)
    if s is not None:
        x_k = a.dot(np.diag(1 / s))
    u, _, _ = tsqr(x_k, compute_svd=True)

    u = u.rechunk('auto')

    if log:
        return u, {}
    else:
        return u


@time_param_log
def sub_svd_init(array: ScaledArray,
                 k: int,
                 warm_start_row_factor: int = 5,
                 seed: int = 42,
                 log: int = 1) -> Union[da.core.Array, Tuple[da.core.Array, dict]]:
    """Attempts to compute a better approximation of the right singular vectors of a matrix using a sample of the
    rows of that matrix.

    Parameters
    ----------
    array
    k
    warm_start_row_factor
    seed
    log

    Returns
    -------

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
    rows = warm_start_row_factor * k
    n, p = array.shape
    row_fraction = rows / n

    sub_log = max(log - 1, 0)

    I, Not_I = array_split(array_shape=array.shape, f=row_fraction, axis=0, seed=seed, log=0)

    sub_array = array[I, :].T

    sub_array = sub_array.rechunk({0: 'auto', 1: -1})

    _sub_svd_return = _sub_svd(array, sub_array, k=k, log=sub_log)

    if sub_log:
        U, _sub_svd_log = _sub_svd_return
        flog = {_sub_svd.__name__: _sub_svd_log}
    else:
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
    """

    Parameters
    ----------
    array
    sub_array
    k
    log

    Returns
    -------

    """
    # VSU' <--- SVD of A'
    V, _, _ = tsqr(sub_array.array, compute_svd=True)  # SVD of A' -> VSU'

    U = v_init(array, V[:, :k])

    if log:
        return U, {}
    else:
        return U


@time_param_log
def rnormal_start(array: da.core.Array,
                  k: int,
                  seed: int = 42,
                  log: int = 1) -> Union[da.core.Array,
                                         Tuple[da.core.Array, dict]]:
    """ Initializes a gaussian normal matrix for Power Method

    Parameters
    ----------
    array : Dask Array
        Array in Power Iteartion
    k : int
        Number of columns needed. Includes buffer
    seed : int
        Seed to set random generator
    log : int
        See logging in README.MD

    Returns
    -------
    omega : Dask Array
        0th iteration of power Method
    """
    m, n = array.shape
    state = RandomState(seed)

    omega = state.standard_normal(
        size=(n, k), chunks=(array.chunks[1], (k,)))

    if log:
        return omega, {}
    else:
        return omega


@time_param_log
def eigengap_init(a: ScaledArray, k: int, b_max: int, warm_start_row_factor: Union[int, float] = 5, tol: float = 1e-6,
                  seed: int = 42, log: int = 1, scoring_method: str = 'q-vals'):
    """
    Computes intelligent initial guess for:
        1) Active column (k + buffer) subspace of array
        2) The "locally optimal" size of buffer that balance:
            eigen_gap convergence rates
            matrix multiplication  times

    Parameters
    ----------
    a
    k
    b_max
    warm_start_row_factor
    tol
    seed
    log
    scoring_method

    Returns
    -------


    Notes
    --------
    Suppose A has eigenvalues (all m1) of:
        [s1, s2, s2, ...,                         s10, ... ]
        [10,  9,  5, 4.99, 4.98, 4.97, 4.96, 3, 1, .5, ... ]

    If one desires the top three most active subspaces of A.
    Using the block power method and a buffer size of 0 the convergence rate would be:
        ~ lambda_3 / lambda_4 = 1.002
        Therefore each iteration only improves the overall solution by .2%.

    However, if we were to select a buffer of 5 then the convergence rate would be:
        ~lambda_3 / lambda_9 = 1.667
        Now the convergence rate is much higher and will require many fewer iterations
        at the cost of each iteration taking more time.

    Moreover, if we select a buffer size of 6 then the convergence rate would be:
        ~lambda_3 / lambda_9 = 5.
        Here the marginal increase of compute dot products of an (n, 9) matrix vs an (n, 8) is
        more than made up for with the massive increase in convergence rate.

    Method
    ------

    A in R(m by n) s.t. m < n

    x in R(n by (k + b')) s.t. (k + b') << n

    Select (k + b_max)*warm_start_row_factor rows from A that form A1

    Perform Monitored SVD on A1
        U, S, V, log <- SVD(A1)
        log contains operation costs of SVD:
            Operational cost of A1.dot(x)

    Extrapolate from A1.dot(x) to determine cost of full scale dot product with A.

    Using S and k, find eigen_gap for potential buffers from 0 -> b_max
    Using desired_tolerance predict how many iterations it will take to find convergence
    Using cost of each operation find time to finish iterations.
    """
    flog = {}
    sub_log = max(0, log - 1)

    m, n = a.shape

    rows = warm_start_row_factor * (k + b_max)
    row_fraction = rows / m

    sub_svd_init_return = sub_svd_init(a,
                                       k=rows,
                                       row_sampling_fraction=row_fraction,
                                       seed=seed,
                                       log=sub_log)
    if sub_log:
        U, S, _, _sub_svd_start_log = sub_svd_init_return
        flog[sub_svd_init_return.__name__] = _sub_svd_start_log
    else:
        U, S, _ = sub_svd_init_return

    U_0, S_0 = svd_to_trunc_svd(u=U, s=S, k=k + b_max)
    S_0_k = svd_to_trunc_svd(s=S_0, k=k)

    S_0_k = S_0_k.persist()

    x = a.T.dot(U_0)
    Q, _ = tsqr(x)
    x = a.sym_mat_mult(Q)
    Q, _ = tsqr(x)

    U_1, S_1, _ = subspace_to_SVD(Q, a, log=0)

    S_1_k = svd_to_trunc_svd(s=S_1, k=k)

    U_1, S_1_k, = dask.persist(U_1, S_1_k)

    init_acc = q_value_converge(S_0_k, S_1_k, log=0)

    try:
        S = S_1.compute()
    except AttributeError:
        S = S_1

    S_gap = float('inf') * np.ones_like(S)
    S_gap[k:b_max] = S[k - 1] / S[k:b_max]

    req_iter = _project_accuracy(init_acc, S_gap, tol) / 6

    def k_cost(n_dimn):
        coeffs = np.array([0.00995696, 0.94308865])  # Found by brute force atm
        return np.polyval(coeffs, n_dimn)

    cost = _project_cost(k_cost, req_iter)

    buff_opt: int = np.argmin(cost)

    U_k = svd_to_trunc_svd(u=U_1, k=buff_opt)

    x = a.T.dot(U_k)

    if log:
        flog['S'] = S
        flog['req_iter'] = req_iter
        flog['costs'] = cost
        return x, flog
    else:
        return x


def _logbase(x: Union[np.ndarray, int, float],
             b: Union[np.ndarray, int, float]) -> np.ndarray:
    """
    Returns log_{b}{x}

    log_{b}(x) = ln(x)/ln(b)

    :param x: Numeric value or Array
    :param b: Numeric value or Array
    """
    return np.log(x) / np.log(b)


def _project_accuracy(acc_init: Union[int, float],
                      eigen_ratio: np.ndarray,
                      acc_final: Union[int, float]) -> np.ndarray:
    """
    Returns number of iterations to find desired tolerance.

    Suppose:
        Acc(i+c) = Acc(i)(1/eigen_ratio)^c

    If we desire Acc(i+c) <= acc_f

    Then c <= log_{acc_f/acc_0}(1/eigen_ratio)

    :param acc_init:
    :param eigen_ratio:
    :param acc_final:
    :return:
    """
    req_iter = _logbase(acc_final / acc_init, 1 / eigen_ratio)
    req_iter[req_iter == 0] = float('inf')
    return req_iter


def _project_cost(cost_func, req_iter):
    return [num_iter * cost_func(k) for k, num_iter in enumerate(req_iter)]
