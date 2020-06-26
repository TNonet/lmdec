import time
import warnings
from abc import ABCMeta, abstractmethod
from typing import Union, List, Optional, Tuple

import dask
import dask.array as da
import numpy as np
from dask.array.linalg import tsqr
from tqdm import tqdm

from lmdec.array.matrix_ops import subspace_to_SVD, subspace_to_V
from lmdec.array.metrics import q_value_converge, subspace_dist, rmse_k
from lmdec.array.random import array_constant_partition, cumulative_partition
from lmdec.array.types import ArrayType, LargeArrayType
from lmdec.array.wrappers.recover_iter import recover_last_value
from lmdec.decomp.init_methods import sub_svd_init, rnormal_start
from lmdec.decomp.utils import PowerMethodSummary


class _IterAlgo(metaclass=ABCMeta):
    """Base Class for Power Method Iteration Based Methods that fit the following form:

    Required Functions:
        Initialization Method, I, with parameters (data, **kwargs)
        Update Solution, u, with parameters (data, x_i, **kwargs)
        Update parameters, p, with parameters (data, x_i, acc_i, **kwargs)
        Calculate Solution Quality, s, with parameters (data, x_i, **kwargs)
        Check Solution, c, with parameters (data, x_i, **kwargs)
        Finalization Method, f, with parameters (data, x_i, **kwargs)


    Algorithm Overview

        x_0 <- I(data, **kwargs)

        for i in {0, 1, 2, ..., max_iter}:
            acc_i <- s(data, x_i, **kwargs)

            if c(x_i):
                break out of loop

            args, kwargs <- p(data, x_i, acc_i, *kwargs)
            x_i+1 <- u(data, x_i, *kwargs)

        x_f <- f(data, x_i, **kwargs)
    """

    def __init__(self,
                 max_iter: Optional[int] = None,
                 scoring_method: Optional[Union[List[str], str]] = None,
                 tol: Optional[Union[List[Union[float, int]], Union[float, int]]] = None,
                 factor: Optional[Union['str', float]] = None,
                 time_limit: Optional[int] = None,
                 warn: Optional[bool] = None):
        if max_iter is None:
            self.max_iter = 50
        else:
            if max_iter <= 0:
                raise ValueError('max_iter must be a postitive integer')
            if int(max_iter) != max_iter:
                raise ValueError('max_iter must be a integer')
            self.max_iter = max_iter
        if scoring_method is None:
            scoring_method = 'q-vals'
        if time_limit is None:
            self.time_limit = 1000
        else:
            if time_limit <= 0:
                raise ValueError("time_limit must be a positive amount")
            self.time_limit = time_limit
        if tol is None:
            tol = .1e-3

        if not isinstance(tol, list):
            tol = [tol]
        if not isinstance(scoring_method, list):
            scoring_method = [scoring_method]

        if len(scoring_method) != len(tol):
            raise ValueError('tolerance, tol, specification must match with convergence criteria, scoring_method,'
                             ' specification. There is no one to one mapping between \n'
                             ' {} and {}'.format(tol, scoring_method))

        if any(x not in ['q-vals', 'rmse', 'v-subspace'] for x in scoring_method):
            raise ValueError('Must use scoring method in {}'.format(['q-vals', 'rmse', 'v-subspace']))

        if factor is None:
            factor = None
        self.factor = factor

        if warn is None:
            warn = False

        self.warn = warn
        self.tol = tol
        self.scoring_method = scoring_method
        self.history = None
        self.num_iter = None
        self.array = None

    def _reset_history(self):
        summary = PowerMethodSummary(time={'start': None, 'stop': None, 'iter': [], 'step': [], 'acc': []},
                                     acc={'q-vals': [], 'rmse': [], 'v-subspace': []},
                                     iter={'U': [], 'S': [], 'V': [], 'last_value': []})

        self.history = summary

    @property
    def time(self):
        return self.history.time['stop'] - self.history.time['start']

    @abstractmethod
    def _initialization(self, data, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _solution_step(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _parameter_step(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _solution_accuracy(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _finalization(self, x, return_history, **kwargs):
        raise NotImplementedError

    @recover_last_value
    def svd(self, array: LargeArrayType, verbose: bool = True, return_history: bool = False, **kwargs):
        """ Computes an approximation to a k truncated Singular Value Decomposition.

        U, S, V <- svd(array)

        such that.

        min || array - USV || ( ||.|| is the frobenius Norm and V is already transposed)

        Over;
            U, S, V

        Where;
            U is orthonormal and has maximum rank k
            S is non-negative and decreasing of length k
            V.T is orthonormal and has maximum rank k

        Parameters
        ----------
        array : array_like
            array to be decomposed into U, S, V
        verbose : bool
            Flag whether to print out progress bar with tdqm.

        return_history : bool
            Flag whether to return tuple of (U, S, V) or the full history object.

            Useful when running with cluster submission.

        kwargs : dict

        Returns
        -------
        U : Dask Array
            Estimation of top k left singular vectors of 'array'
        S : Dask Array
            Estimation of top k singular values of 'array'
        V : Dask Array
            Estimation of top k right  singular vectors of 'array'
        """

        self._reset_history()

        self.history.time['start'] = time.time()

        x_k = self._initialization(array, **kwargs)

        converged = False
        for i in tqdm((range(1, self.max_iter + 1)), disable=not verbose):
            self.num_iter = i
            iter_start = time.time()
            acc_list = self._solution_accuracy(x_k, **kwargs)
            self.history.time['acc'].append(time.time() - iter_start)

            for method, acc, tol in zip(self.scoring_method, acc_list, self.tol):
                self.history.acc[method].append(acc)
                if acc <= tol:
                    converged = True

            if converged:
                break

            self._parameter_step(x_k, **kwargs)

            step_start = time.time()
            x_k = self._solution_step(x_k, **kwargs)  # x_k+1 <= AA'x_k
            self.history.time['step'].append(time.time() - step_start)

            iter_end = time.time()
            self.history.time['iter'].append(iter_end - iter_start)

            if time.time() - self.history.time['start'] > self.time_limit:
                break

        result = self._finalization(x_k, return_history=return_history, **kwargs)

        self.history.time['stop'] = time.time()

        if self.warn and not converged:
            warnings.warn("Did not converge. \n"
                          "Time Usage : {0:.2f}s of {1}s (Time Limit) \n"
                          "Iteration Usage : {2} of {3} (Iteration Limit)"
                          .format(self.time, self.time_limit, self.num_iter, self.max_iter))

        return result


class PowerMethod(_IterAlgo):

    def __init__(self, k: int = 10,
                 max_iter: int = 50,
                 scoring_method='q-vals',
                 tol=.1,
                 factor='n',
                 lmbd=.01,
                 buffer=10,
                 sub_svd_start: bool = True,
                 init_row_sampling_factor: int = 5,
                 time_limit=1000,
                 warn: bool = False):
        """ Implementation of the Power Method for computing the truncated SVD decomposition of an array

        Parameters
        ----------
        k : int
            number of components in the singular value decomposition

        max_iter : int
            maximum number of iterations to run before returning last value

        scoring_method : str or list of str
            accuracy method or list of  accuracy methods that will be evaluated at each iteration.

            accuracy methods:
            - 'rmse': see subspace_dist in lmdec.array.metrics for more information
                Is the slowest of the accuracy methods
                `rmse` /propto sqrt('q-vals')
            - 'q-vals': see q_value_converge in lmdec.array.metrics for more information
                Is the fastest of the accuracy methods
                `q-vals` /propto 'rmse'**2
            - 'v-subspace' see subspace_dist in lmdec.array.metrics for more information
                Is faster than 'rmse' but slower than 'q-vals'

            if `scoring_method` is a list, then `tol` must be a list of the same length too.

            if `scoring_method` is a list, convergence will be assigned if one of the tolerance criteria is found.

                convergence = any(if t >= method() for t, method in zip(tol, scoring_method))

        tol : float or list of floats
            tolerances needed for convergence determination that will be evaluated at each iteration.

        lmbd : float
            percentage of previous iteration to replace with Gaussian Noise.

            This is useful to prevent convergence on a lower magnitude eigenvector

            x = (1 - lmbd)*x + lmbda*N(0, ||x||_c)
            Note that Gaussian Noise is scaled by ||x||_c which is a vector of the column 2 norms of x

        buffer : int
            number of columns in addition to `k` used to sample subspace of array.

            Generally a larger buffer leads to convergence in few iterations, however, each iteration is more expensive.

               Accuracy Iteration i ~ lambda_k/lambda_{k+buffer} * Accuracy Iteration i - 1

        sub_svd_start : bool
            whether to initialize x0 with a svd of subset of rows of array or with gaussian matrices

        init_row_sampling_factor : int
            if `sub_svd_start` is True, the size of the subset of rows used to compute x0 will be:
                init_row_sampling_factor * (k + buffer)
        time_limit : int
            maximum number of seconds that the algorithm will start another iteration.

            if time_limit is X and iteration i ends at time Y > X:
                Algorithm will exit with the current iteration.

            However, it is worth noting that if iterations take a long time, it is possible that an iteration starts at
            time X - 1 and therefore would not exit until ~ X + time of an iteration.

            It is worth noting that one can keyboard interrupt the algorithm and the most recent solution will be
            stored in `last_value`.
        """

        super().__init__(max_iter=max_iter,
                         scoring_method=scoring_method,
                         tol=tol,
                         factor=factor,
                         time_limit=time_limit,
                         warn=warn)

        if int(k) <= 0:
            raise ValueError('k must be a positive integer')
        if int(buffer) < 0:
            raise ValueError('buffer must be a non-negative integer')
        self.k = int(k)
        self.buffer = int(buffer)
        self.lmbd = lmbd

        self.sub_svd_start = sub_svd_start
        if init_row_sampling_factor <= 0:
            raise ValueError('init_row_sampling_factor must be a positive value')
        self.init_row_sampling_factor = init_row_sampling_factor
        self.compute = True

    def project(self, x: Union[np.ndarray, da.core.Array], onto: Union[np.ndarray, da.core.Array],
                scale_center_x: bool = True):
        """ Projects `x` onto `onto`.

        Roughly equivalent to Proj_{onto}(x)

        Parameters
        ----------
        x : array_like, shape (N, P)
            Data to be project onto `x`.
        onto : array_like, shape (M, P)
            Subspace for `x` to be projected onto.
        scale_center_x : bool
            Whether to scale and/or center `x`, if specified in PowerMethod settings.
            if `scale_center_x` is False:
                x will not be scaled and/or centered similar to PowerMethod.svd(...) works

        Notes
        -----
        P in shape of `x` and shape of `onto` must match. In addition, for `scale_center_x` to be True, `P` must match
        the dimensions of PowerMethod.scaled_array.center_vector and PowerMethod.scaled_array.scale_vector

        Returns
        -------
        projected L array_like, shape (N, M)
            `x` projected onto `onto`
        """
        raise NotImplementedError

    def _initialization(self, data, **kwargs):
        vec_t = self.k + self.buffer

        if vec_t > min(data.shape):
            raise ValueError('Cannot find more than min(n,p) singular values of array function.'
                             'Currently k = {}, buffer = {}. k + b > min(n,p)'.format(self.k, self.buffer))

        self.array = da.array(data)

        if self.factor == 'n':
            self.factor = self.array.shape[0]
        elif self.factor == 'p':
            self.factor = self.array.shape[1]
        elif self.factor is None:
            self.factor = False

        if self.sub_svd_start:
            x = sub_svd_init(self.array,
                             k=vec_t,
                             warm_start_row_factor=self.init_row_sampling_factor,
                             log=0)

            if self.lmbd:
                c_norms = np.linalg.norm(x, 2, axis=0)
                x *= (1 - self.lmbd)
                x += (self.lmbd * c_norms / np.sqrt(x.shape[0])) * da.random.normal(size=x.shape)
        else:
            x = rnormal_start(self.array, vec_t, log=0)

        return x.persist()

    def _solution_step(self, x, **kwargs):
        q, _ = tsqr(x)
        x = self.array.dot(self.array.T.dot(q))
        if self.factor:
            x = x/self.factor
        return x.persist()

    def _parameter_step(self, x, **kwargs):
        pass

    def _solution_accuracy(self, x, **kwargs):
        if any(m in self.scoring_method for m in ['rmse', 'v-subspace']):
            U_k, S_k, V_k = subspace_to_SVD(x, self.array, sqrt_s=True, k=self.k, full_v=True, log=0)
        else:
            U_k, S_k, V_k = subspace_to_SVD(x, self.array, sqrt_s=True, k=self.k, full_v=False, log=0)

        U_k, S_k, V_k = dask.persist(U_k, S_k, V_k)

        self.history.iter['last_value'] = (U_k, S_k, V_k)
        acc_list = []
        for method in self.scoring_method:
            if method == 'q-vals':
                try:
                    prev_S_k = self.history.iter['S'][-1]
                    acc = q_value_converge(S_k, prev_S_k)
                except IndexError:
                    acc = float('INF')
                self.history.iter['S'].append(S_k.compute())
            elif method == 'rmse':
                acc = rmse_k(self.array, U_k, S_k ** 2, factor=self.factor)
            else:  # method == 'v-subspace'
                try:
                    prev_V_k = self.history.iter['V'][-1]
                    acc = subspace_dist(V_k.T, prev_V_k.T, S_k)
                except IndexError:
                    acc = float('INF')
                self.history.iter['V'].append(V_k.compute())
            acc_list.append(acc)
        return acc_list

    def _finalization(self, x, return_history, **kwargs):
        if self.history.iter['last_value'][2].shape[1] == self.array.shape[1]:
            # If V from last_value is calculated with full_v.
            pass
        else:
            U, S, _ = self.history.iter['last_value']
            V = subspace_to_V(x, self.array, self.k).persist()
            self.history.iter['last_value'] = U, S, V

        if return_history:
            return self.history
        else:
            return self.history.iter['last_value']


class _vPowerMethod(PowerMethod):
    """ Implementation of a seeded PowerMethod.

    See SuccessiveBatchedPowerMethod for more details.

    Notes
    -----
    Suppose A is an array of shape (N, P), that we decompose into A1, A2, ..., Ak.

    Let Aij k>=j>i>=i be [Ai; Ai+1; ... Aj-1; Aj]

    A = [A1; A2; A3, ... Ak], where Ai has shape (ni, p).
        ^^^^^^^
        A12
        ^^^^^^^^^^^
        A13

    Let the SVD of A1 be U1, S1, V1.

    Using U1 to seed the PowerMethod iteration of A12
    """

    def __init__(self, v_start: ArrayType, k=None, buffer=None, max_iter=None, scoring_method=None, tol=None,
                 full_svd: bool = False, factor=None, warn: bool = False):
        super().__init__(max_iter=max_iter,
                         scoring_method=scoring_method,
                         tol=tol,
                         factor=factor,
                         lmbd=1,
                         warn=warn)
        self.v_start = v_start.rechunk({0: 'auto', 1: -1})
        self.k = k
        self.buffer = buffer
        self.full_svd = full_svd

    def _initialization(self, data, **kwargs):
        self.array = data
        return self.array.dot(self.v_start)

    def _finalization(self, x, return_history, **kwargs) -> Union[da.core.Array,
                                                                  Tuple[da.core.Array, da.core.Array, da.core.Array]]:
        if not self.full_svd:
            return self.array.T.dot(x)
        else:
            return super()._finalization(x, return_history=False)


class SuccessiveBatchedPowerMethod(PowerMethod):
    """
    Implementation of Power Method that used an increasing large batch of rows of the array

    Notes
    -----
    Suppose A exists in R(n \times p)

    If we have reason to believe that that rows of A are generated from an underlying distribution, then a subset of the
    rows of A may be sufficient to find a high quality singular value decomposition of A

    In other words, it would be reasonable to assume that with just n' < n rows of A we can accurately learn the same
    factors that we would learn with all n rows.

    Algorithm:

       I_rows = [0:n1, 0:n2, 0:n3, ..., 0:n]
       U0, S0, V0 <- Direct Truncated SVD of Small Subset Rows of A'

       convergence <- False
       until convergence:
           i = i + 1
           Ui, Si, Vi <- PowerMethod(A[I_rows[i], :], V_init=V{i-1}), where V_init is the seed for PowerMethod instead
                                                                      of a saussian sample or small subset of rows.
           convergence = ||Si - S{i-1}||_2

       return Ui, Si, Vi
    """

    def __init__(self, k: int = 10,
                 max_sub_iter: int = 50,
                 factor: Optional[str] = 'n',
                 scoring_method='q-vals',
                 f=.1,
                 lmbd=.1,
                 tol=.1,
                 buffer=10,
                 sub_svd_start: bool = True,
                 init_row_sampling_factor: int = 5,
                 max_sub_time=1000,
                 warn: bool = False):
        super().__init__(k=k, max_iter=max_sub_iter, factor=factor,
                         scoring_method=scoring_method, tol=tol, buffer=buffer, sub_svd_start=sub_svd_start,
                         init_row_sampling_factor=init_row_sampling_factor, time_limit=max_sub_time, warn=warn)
        self.f = f
        self.history = None
        self.lmbd = lmbd
        self.sub_svds = None

    def _reset_history(self):
        super()._reset_history()
        self.history.acc['sub_svd_acc'] = []
        self.history.iter['sub_svd'] = []

    @recover_last_value
    def svd(self, array: LargeArrayType, verbose: bool = True, return_history: bool = False, **kwargs):

        self._reset_history()
        self.history.time['start'] = time.time()

        self.array = da.array(array)

        if self.factor == 'n':
            self.factor = self.array.shape[0]
        elif self.factor == 'p':
            self.factor = self.array.shape[1]
        elif self.factor is None:
            self.factor = False

        vec_t = self.k + self.buffer

        partitions = array_constant_partition(self.array.shape, f=self.f, min_size=vec_t)
        partitions = cumulative_partition(partitions)

        sub_array = self.array[partitions[0], :]

        if self.sub_svd_start == 'warm':
            x = sub_svd_init(sub_array,
                             k=vec_t,
                             warm_start_row_factor=self.init_row_sampling_factor,
                             log=0)
        else:
            x = rnormal_start(sub_array, k=vec_t, log=0)

        x = sub_array.T.dot(x)

        for part in tqdm(partitions[:-1], disable=not verbose):
            _PM = _vPowerMethod(v_start=x, k=self.k, buffer=self.buffer, max_iter=self.max_iter, factor=self.factor,
                                scoring_method=self.scoring_method, tol=self.tol, warn=self.warn)

            x = _PM.svd(self.array[part, :], verbose=False, **{'mask_nan': False, 'transpose': False})
            self.history.iter['last_value'] = _PM.history.iter['last_value']
            self.history.iter['sub_svd'].append(self.history.iter['last_value'])

            if self.lmbd:
                c_norms = np.linalg.norm(x, 2, axis=0)
                x *= (1 - self.lmbd)
                x += (self.lmbd * c_norms / np.sqrt(x.shape[0])) * da.random.normal(size=x.shape)

            if 'v-subspace' in self.scoring_method:
                self.history.iter['V'].append(_PM.history.iter['V'][-1])

            self.history.iter['S'].append(_PM.history.iter['S'][-1])
            self.history.acc['sub_svd_acc'].append(_PM.history.acc)
            self.history.time['iter'].append(_PM.history.time)

        _PM = _vPowerMethod(v_start=x, k=self.k, buffer=self.buffer, max_iter=self.max_iter, factor=self.factor,
                            scoring_method=self.scoring_method, tol=self.tol, full_svd=True, warn=self.warn)

        _PM.svd(self.array, verbose=False, **{'mask_nan': False, 'transpose': False})

        self.history.iter['last_value'] = _PM.history.iter['last_value']
        self.history.iter['sub_svd'].append(self.history.iter['last_value'])

        if return_history:
            return self.history
        else:
            return self.history.iter['last_value']
