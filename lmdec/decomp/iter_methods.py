from abc import ABCMeta, abstractmethod
from collections import namedtuple

from typing import Union, List, Optional, Tuple

import warnings

import dask
import dask.array as da
from dask.array.linalg import tsqr
import numpy as np
import time

from pandas_plink import read_plink1_bin, read_plink
from pathlib import Path

from lmdec.array.matrix_ops import subspace_to_SVD, subspace_to_V
from lmdec.array.scaled import ScaledArray
from lmdec.decomp.init_methods import sub_svd_init, rnormal_start
from lmdec.array.metrics import q_value_converge, subspace_dist, rmse_k
from lmdec.array.random import array_constant_partition, cumulative_partition
from lmdec.array.types import ArrayType, LargeArrayType
from lmdec.array.wrappers.recover_iter import recover_last_value


PowerMethodSummary = namedtuple('PowerMethodSummary', ['time', 'acc', 'iter'])


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
                 scale: Optional[bool] = None,
                 center: Optional[bool] = None,
                 factor: Optional[str] = None,
                 scoring_method: Optional[Union[List[str], str]] = None,
                 tol: Optional[Union[List[Union[float, int]], Union[float, int]]] = None,
                 time_limit: Optional[int] = None):
        if max_iter is None:
            self.max_iter = 50
        else:
            if max_iter <= 0:
                raise ValueError('max_iter must be a postitive integer')
            if int(max_iter) != max_iter:
                raise ValueError('max_iter must be a integer')
            self.max_iter = max_iter
        if scale is None:
            self._scale = True
        else:
            self._scale = scale
        if center is None:
            self._center = True
        else:
            self._center = center
        self._factor = factor
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

        self.tol = tol
        self.scoring_method = scoring_method
        self.history = None
        self.num_iter = None
        self.scaled_array = None

    def _reset_history(self):
        summary = PowerMethodSummary(time={'start': None, 'stop': None, 'iter': [], 'step': [], 'acc': []},
                                     acc={'q-vals': [], 'rmse': [], 'v-subspace': []},
                                     iter={'U': [], 'S': [], 'V': [], 'last_value': []})

        self.history = summary

    @property
    def scale(self):
        return self.scaled_array.scale_vector

    @property
    def center(self):
        return self.scaled_array.center_vector

    @property
    def factor(self):
        return self.scaled_array.factor_value

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
    def _finalization(self, x, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_array(array: Optional[LargeArrayType] = None, path_to_files: Optional[Union[str, Path]] = None,
                  bed: Optional[Union[str, Path]] = None, bim: Optional[Union[str, Path]] = None,
                  fam: Optional[Union[str, Path]] = None, persist: bool = False, transpose: bool = False,
                  mask_nan: bool = False, dtype: Optional[str] = 'uint8',
                  mask_fill: Union[str, int] = 'median') -> LargeArrayType:
        """Gathers array from possible methods

        Requires one of the following parameter configures to be satisfied to load an array:
            - array is specified and no other parameters are specified
            - path_to_files is specified and no other parameters are specified
            - bim AND fam AND bed are specified and no other parameters are specified

        Parameters
        ----------
        array : LargeArrayType, optional
                Union[Dask Array, Numpy Array]
        path_to_files : path_like, optional
            Assuming bim, fam, bed files are in the following format
                </path/to/data.bim>
                </path/to/data.fam>
                </path/to/data.bed>
            Then, path_to_files would be '/path/to/data'
        bed : path_like, optional
            '/path/to/data.bed'
        bim : path_like, optional
            '/path/to/data.bim'
        fam : path_like, optional
            '/path/to/data.fam'
        transpose : bool
            Whether `array` is stored/loaded in transposed format

            If A is stored/loaded as A.T but SVD(A) is desired set transpose flag to True
        persist: bool
            Whether to hold/compute `array` in memory or not.

            Will decrease computation time at the cost of greatly increasing memory requirements.
            If `array` can be stored safely in memory, setting persist to True is a reasonable option.
        mask_nan : bool
            Whether to mask nan values in `array`

            If `array` has no nan-values setting this flag to False will increase computation time.

        mask_fill : numeric of type `dtype` or 'mean' or 'median'
            If numeric, will be used as value to replace nan-values
            If 'mean' or 'median', method will be used to fill nan-values.

            #TODO: Implement Row or Column specific mask_fills by expanding ScaledArray functionality

        dtype : str, optional
            If persisting the array, convert to type `dtype`.

        Returns
        -------
        array : ScaledArray or Dask Array
            Loaded or converted Dask Array
        """
        if array is not None and not all([path_to_files, bed, bim, fam]):
            pass
        elif path_to_files is not None and not all([array, bed, bim, fam]):
            (_, _, G) = read_plink(path_to_files)
            array = G
        elif all(p is not None for p in [bed, bim, fam]) and not all([array, path_to_files]):
            G = read_plink1_bin(bed, bim, fam)
            array = G.data
        else:
            raise ValueError('Uninterpretable input.'
                             ' Please specify array, xor path_to_files, xor (bed and bim and fim)')

        if isinstance(array, (ScaledArray, da.core.Array)):
            pass
        elif isinstance(array, np.ndarray):
            array = da.from_array(array, chunks='auto')

        if len(array.shape) != 2:
            raise ValueError("Must be a 2-D array")

        if transpose:
            array = array.T
            array = array.rechunk({0: 'auto', 1: -1})

        if mask_nan:
            if isinstance(mask_fill, int):
                mask_fill = mask_fill
            elif mask_fill == 'median':
                mask_fill = round(da.nanmedian(da.nanmedian(array, axis=0), axis=0).compute())
            elif mask_fill == 'mean':
                mask_fill = round(da.nanmean(array).compute())
            else:
                raise NotImplementedError('No method of {}'.format(mask_fill))
            array = da.ma.masked_invalid(array)
            da.ma.set_fill_value(array, mask_fill)
            array = da.ma.filled(array)

        if persist:
            if dtype:
                array = array.astype(dtype)
            array = array.persist()

        return array

    @recover_last_value
    def svd(self, array: Optional[LargeArrayType] = None, path_to_files: Optional[Union[str, Path]] = None,
            bed: Optional[Union[str, Path]] = None, bim: Optional[Union[str, Path]] = None,
            fam: Optional[Union[str, Path]] = None, **kwargs):
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
        array : array_like, optional
            array to be decomposed into U, S, V
        path_to_files : path_like, optional
            path to bim, fam, and bed files that hold array to be decomposed into U, S, V

            Assuming bim, fam, bed files are in the following format
                </path/to/data.bim>
                </path/to/data.fam>
                </path/to/data.bed>
            Then, path_to_files would be '/path/to/data'
        bed : path_like, optional
            path to bed file that holds array to be decomposed into U, S, V

            '/path/to/data.bed'
        bim : path_like, optional
            path to bim file that holds array to be decomposed into U, S, V

            '/path/to/data.bim'
        fam : path_like, optional
            path to fam file that holds array to be decomposed into U, S, V

            '/path/to/data.fam'
        kwargs : dict

        Notes
        -----
        See get_array for details on kwargs and loading formats

        Returns
        -------
        U : Dask Array
            Estimation of top k left singular vectors of `array
        S : Dask Array
            Estimation of top k singular values of `array
        V : Dask Array
            Estimation of top k right  singular vectors of `array
        """

        self._reset_history()

        self.history.time['start'] = time.time()

        data = self.get_array(array, path_to_files, bed, bim, fam, **kwargs)

        x_k = self._initialization(data, **kwargs)

        converged = False
        for i in range(1, self.max_iter + 1):
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
            x_k = self._solution_step(x_k, **kwargs)
            self.history.time['step'].append(time.time() - step_start)

            iter_end = time.time()
            self.history.time['iter'].append(iter_end - iter_start)

            if time.time() - self.history.time['start'] > self.time_limit:
                break

        result = self._finalization(x_k, **kwargs)

        self.history.time['stop'] = time.time()

        if not converged:
            warnings.warn("Did not converge. \n"
                          "Time Usage : {0:.2f}s of {1}s (Time Limit) \n"
                          "Iteration Usage : {2} of {3} (Iteration Limit)"
                          .format(self.time, self.time_limit, self.num_iter, self.max_iter))

        return result


class PowerMethod(_IterAlgo):

    def __init__(self, k: int = 10,
                 max_iter: int = 50,
                 scale: bool = True,
                 center: bool = True,
                 std_method: str = 'normal',
                 factor: Optional[str] = 'n',
                 scoring_method='q-vals',
                 tol=.1,
                 lmbd=.01,
                 buffer=10,
                 sub_svd_start: bool = True,
                 init_row_sampling_factor: int = 5,
                 time_limit=1000):
        """ Standard Implementation of the Power Method for computing the truncated SVD decomposition of an array

        Parameters
        ----------
        k : int
            number of components in the singular value decomposition

        max_iter : int
            maximum number of iterations to run before returning best value

        scale : bool
            whether to scale array so the columns have standard deviation close to 1

            scaled after array is centered or not centered

        center
            whether to center the array so the columns have a mean close to 1

        factor : str, optional
            factor to divide singular value results by.

            if factor is None -> will not divide singular values
            if factor is 'n' -> will scale by number of rows
            if factor is 'p' -> will scale by number of columns

            In other words

            singular values of array, s(array) == s(1/factor * array array')

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

            if `scoring_method` is a list, then `tol` must be a list too.

            if `scoring_method` is a list, convergence will be assigned if one of the tolerance criteria is found.

                convergence = any(if t >= method() for t, method in zip(tol, scoring_method))

        tol : float or list of floats
            tolerances needed for convergence determination that will be evaluated at each iteration.

        lmbd : float
            percentage of previous iteration to replace with Gaussian Noise

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
                         scale=scale,
                         center=center,
                         factor=factor,
                         scoring_method=scoring_method,
                         tol=tol,
                         time_limit=time_limit)

        if int(k) <= 0:
            raise ValueError('k must be a positive integer')
        if int(buffer) < 0:
            raise ValueError('buffer must be a non-negative integer')
        self.k = int(k)
        self.buffer = int(buffer)
        self.lmbd = lmbd
        self.std_method = std_method

        self.sub_svd_start = sub_svd_start
        if init_row_sampling_factor <= 0:
            raise ValueError('init_row_sampling_factor must be a positive value')
        self.init_row_sampling_factor = init_row_sampling_factor
        self.compute = True

    def _initialization(self, data, **kwargs):
        vec_t = self.k + self.buffer

        if vec_t > min(data.shape):
            raise ValueError('Cannot find more than min(n,p) singular values of array function.'
                             'Currently k = {}, buffer = {}. k + b > min(n,p)'.format(self.k, self.buffer))

        if not isinstance(data, ScaledArray):
            self.scaled_array = ScaledArray(scale=self._scale, center=self._center, factor=self._factor,
                                            std_dist=self.std_method)
            self.scaled_array.fit(data)
        else:
            self.scaled_array = data

        if self.sub_svd_start:
            x = sub_svd_init(self.scaled_array,
                             k=vec_t,
                             warm_start_row_factor=self.init_row_sampling_factor,
                             log=0)

            if self.lmbd:
                c_norms = np.linalg.norm(x, 2, axis=0)
                x *= (1 - self.lmbd)
                x += (self.lmbd * c_norms / np.sqrt(x.shape[0])) * da.random.normal(size=x.shape)
        else:
            x = rnormal_start(self.scaled_array, vec_t, log=0)

        self.scaled_array.fit_x(x)
        return x.persist()

    def _solution_step(self, x, **kwargs):
        q, _ = tsqr(x)
        x = self.scaled_array.sym_mat_mult(x=q)  # x <- AA'q
        return x.persist()

    def _parameter_step(self, x, **kwargs):
        pass

    def _solution_accuracy(self, x, **kwargs):
        if any(m in self.scoring_method for m in ['rmse', 'v-subspace']):
            U_k, S_k, V_k = subspace_to_SVD(x, self.scaled_array, sqrt_s=True, k=self.k, full_v=True, log=0)
        else:
            U_k, S_k, V_k = subspace_to_SVD(x, self.scaled_array, sqrt_s=True, k=self.k, full_v=False, log=0)

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
                acc = rmse_k(self.scaled_array, U_k, S_k ** 2)
            else:  # method == 'v-subspace'
                try:
                    prev_V_k = self.history.iter['V'][-1]
                    acc = subspace_dist(V_k.T, prev_V_k.T, S_k)
                except IndexError:
                    acc = float('INF')
                self.history.iter['V'].append(V_k.compute())
            acc_list.append(acc)
        return acc_list

    def _finalization(self, x, **kwargs):
        if self.history.iter['last_value'][2].shape[1] == self.scaled_array.shape[1]:
            # If V from last_value is calculated with full_v.
            return self.history.iter['last_value']
        else:
            U, S, _ = self.history.iter['last_value']
            V = subspace_to_V(x, self.scaled_array, self.k).persist()
            return U, S, V


class _vPowerMethod(PowerMethod):

    def __init__(self, v_start: ArrayType, k=None, buffer=None, max_iter=None, scoring_method=None, tol=None,
                 full_svd: bool = False):
        super().__init__(max_iter=max_iter,
                         scoring_method=scoring_method,
                         tol=tol,
                         lmbd=1)
        self.v_start = v_start.rechunk({0: 'auto', 1: -1})
        self.k = k
        self.buffer = buffer
        self.full_svd = full_svd

    def _initialization(self, data, **kwargs):
        self.scaled_array = data
        self.scaled_array.fit_x(self.v_start)
        return self.scaled_array.dot(self.v_start)

    def _finalization(self, x, **kwargs) -> Union[da.core.Array, Tuple[da.core.Array, da.core.Array, da.core.Array]]:
        if not self.full_svd:
            return self.scaled_array.T.dot(x)
        else:
            return super()._finalization(x)


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
           Ui, Si, Vi <- PowerMethod(A[I_rows[i], :], V_init=V{i-1})
           convergence = ||Si - S{i-1}||_2

       return Ui, Si, Vi
    """

    def __init__(self, k: int = 10,
                 max_sub_iter: int = 50,
                 scale: bool = True,
                 center: bool = True,
                 factor: Optional[str] = 'n',
                 std_method : str = 'normal',
                 scoring_method='q-vals',
                 f=.1,
                 lmbd=.1,
                 tol=.1,
                 buffer=10,
                 sub_svd_start: bool = True,
                 init_row_sampling_factor: int = 5,
                 max_sub_time=1000):
        super().__init__(k=k, max_iter=max_sub_iter, scale=scale, center=center, factor=factor,
                         scoring_method=scoring_method, tol=tol, buffer=buffer, sub_svd_start=sub_svd_start,
                         init_row_sampling_factor=init_row_sampling_factor, time_limit=max_sub_time,
                         std_method=std_method)
        self.f = f
        self.history = None
        self.lmbd = lmbd
        self.sub_svds = None

    def _reset_history(self):
        super()._reset_history()
        self.history.acc['sub_svd_acc'] = []
        self.history.iter['sub_svd'] = []

    @recover_last_value
    def svd(self, array: Optional[LargeArrayType] = None, path_to_files: Optional[Union[str, Path]] = None,
            bed: Optional[Union[str, Path]] = None, bim: Optional[Union[str, Path]] = None,
            fam: Optional[Union[str, Path]] = None, **kwargs):

        self._reset_history()
        self.history.time['start'] = time.time()

        array = self.get_array(array, path_to_files, bed, bim, fam, **kwargs)

        if not isinstance(array, ScaledArray):
            self.scaled_array = ScaledArray(scale=self._scale, center=self._center, factor=self._factor)
            self.scaled_array.fit(array)
        else:
            self.scaled_array = array

        vec_t = self.k + self.buffer
        partitions = array_constant_partition(self.scaled_array.shape, f=self.f, min_size=vec_t)
        partitions = cumulative_partition(partitions)

        sub_array = self.scaled_array[partitions[0], :]

        if self.sub_svd_start == 'warm':
            x = sub_svd_init(sub_array,
                             k=vec_t,
                             warm_start_row_factor=self.init_row_sampling_factor,
                             log=0)
        else:
            x = rnormal_start(sub_array, k=vec_t, log=0)

        x = sub_array.T.dot(x)

        for i, part in enumerate(partitions[:-1]):
            _PM = _vPowerMethod(v_start=x, k=self.k, buffer=self.buffer, max_iter=self.max_iter,
                                scoring_method=self.scoring_method, tol=self.tol)

            x = _PM.svd(self.scaled_array[part, :], **{'mask_nan': False, 'transpose': False})
            self.history.iter['last_value'] = _PM.history.iter['last_value']
            self.history.iter['sub_svd'].append(self.history.iter['last_value'])

            if self.lmbd:
                c_norms = np.linalg.norm(x, 2, axis=0)
                x *= (1 - self.lmbd)
                x += (self.lmbd*c_norms/np.sqrt(x.shape[0])) * da.random.normal(size=x.shape)

            if 'v-subspace' in self.scoring_method:
                self.history.iter['V'].append(_PM.history.iter['V'][-1])

            self.history.iter['S'].append(_PM.history.iter['S'][-1])
            self.history.acc['sub_svd_acc'].append(_PM.history.acc)
            self.history.time['iter'].append(_PM.history.time)

        _PM = _vPowerMethod(v_start=x, k=self.k, buffer=self.buffer, max_iter=self.max_iter,
                            scoring_method=self.scoring_method, tol=self.tol, full_svd=True)

        return _PM.svd(self.scaled_array, **{'mask_nan': False, 'transpose': False})
