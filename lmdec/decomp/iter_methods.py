from abc import ABCMeta, abstractmethod

from typing import Union, List, Optional

import warnings

import dask
import dask.array as da
from dask.array.linalg import tsqr
import numpy as np
import time

from pandas_plink import read_plink1_bin, read_plink
from pathlib import Path

from lmdec.array.matrix_ops import subspace_to_SVD
from lmdec.array.scaled import ScaledArray
from lmdec.decomp.init_methods import sub_svd_init, rnormal_start, eigengap_init
from lmdec.array.metrics import q_value_converge, subspace_dist, rmse_k
from lmdec.array.transform import acc_format_svd
from lmdec.array.random import array_constant_partition, cumulative_partition
from lmdec.array.types import ArrayType, LargeArrayType


class _IterAlgo(metaclass=ABCMeta):
    """Base Class for All Iteration Based Methods that fit the following form:

    Required Functions:
        Initialization Method, I, with parameters (data, *args, **kwargs)
        Update Solution, u, with parameters (data, x_i, *args, **kwargs)
        Update parameters, p, with parameters (data, x_i, acc_i, *args, **kwargs)
        Calculate Solution Quality, s, with parameters (data, x_i, *args, **kwargs)
        Check Solution, c, with parameters (data, x_i, *args, **kwargs)
        Finalization Method, f, with parameters (data, x_i, *args, **kwargs)


    Algorithm Overview

        x_0 <- I(data, *args, **kwargs)

        for i in {0, 1, 2, ..., max_iter}:
            acc_i <- s(data, x_i, *args, **kwargs)

            if c(x_i):
                break out of loop

            args, kwargs <- p(data, x_i, acc_i, *args, *kwargs)
            x_i+1 <- u(data, x_i, *args, *kwargs)

        x_f <- f(data, x_i, *args, **kwargs)
    """

    def __init__(self,
                 max_iter: Optional[int] = None,
                 scale: Optional[bool] = None,
                 center: Optional[bool] = None,
                 factor: Optional[str] = None,
                 scoring_method: Optional[Union[List[str], str]] = None,
                 p: Optional[Union[int, float]] = None,
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
        if p is None:
            self.p = 1
        else:
            if p != 1:
                raise NotImplementedError('p other than 1 is not supported yet')
            self.p = p
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
        self.history = {'times': {'start': None, 'stop': None, 'iter': [], 'step': [], 'acc': []},
                        'acc': {'q-vals': [], 'rmse': [], 'v-subspace': []},
                        'iter': {'U': [], 'S': [], 'V': []}}
        self.num_iter = None
        self.scaled_array = None

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
        return self.history['times']['stop'] - self.history['times']['start']

    @abstractmethod
    def _initialization(self, data, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _solution_step(self, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _parameter_step(self, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _solution_accuracy(self, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _finalization(self, x, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_array(array: Optional[LargeArrayType] = None, path_to_files: Optional[Union[str, Path]] = None,
                  bed: Optional[Union[str, Path]] = None, bim: Optional[Union[str, Path]] = None,
                  fam: Optional[Union[str, Path]] = None, mask_nan: bool = True) -> LargeArrayType:
        """Gathers array from possible methods

        Requires one of the following parameter configures to be satisfied to load an array:
            - array is specified and no other parameters are specified
            - path_to_files is specified and no other parameters are specified
            - bim AND fam AND bed are specified and no other parameters are specified

        Parameters
        ----------
        array : LargeArrayType, optional
                Union[Dask Array, Numpy Array, Sparse Scipy Array, Finite Sparse Array]
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
        mask_nan : bool
            Whether to mask nan values

        Returns
        -------
        s_array : LargeArrayType
            Loaded or converted Dask Array Type
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

        if mask_nan:
            return da.ma.masked_invalid(array)
        else:
            return array

    def svd(self, array: Optional[LargeArrayType] = None, path_to_files: Optional[Union[str, Path]] = None,
            bed: Optional[Union[str, Path]] = None, bim: Optional[Union[str, Path]] = None,
            fam: Optional[Union[str, Path]] = None, *args, **kwargs):

        self.history['times']['start'] = time.time()

        data = self.get_array(array, path_to_files, bed, bim, fam, **kwargs)
        x_k = self._initialization(data, *args, **kwargs)

        converged = False
        for i in range(1, self.max_iter + 1):
            self.num_iter = i
            iter_start = time.time()
            acc_list = self._solution_accuracy(x_k, *args, **kwargs)
            self.history['times']['acc'].append(time.time() - iter_start)

            for method, acc, tol in zip(self.scoring_method, acc_list, self.tol):
                self.history['acc'][method].append(acc)
                if acc <= tol:
                    converged = True

            if converged:
                break

            self._parameter_step(x_k, *args, **kwargs)

            step_start = time.time()
            x_k = self._solution_step(x_k, *args, **kwargs)
            self.history['times']['step'].append(time.time() - step_start)

            iter_end = time.time()
            self.history['times']['iter'].append(iter_end - iter_start)

            if time.time() - self.history['times']['start'] > self.time_limit:
                break

        result = self._finalization(x_k, *args, **kwargs)
        self.history['times']['stop'] = time.time()

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
                 factor: Optional[str] = 'n',
                 scoring_method='q-vals',
                 p=1,
                 tol=.1,
                 buffer=10,
                 sub_svd_start: bool = True,
                 init_row_sampling_factor: int = 5,
                 time_limit=1000):

        super().__init__(max_iter=max_iter,
                         scale=scale,
                         center=center,
                         factor=factor,
                         scoring_method=scoring_method,
                         p=p,
                         tol=tol,
                         time_limit=time_limit)

        if int(k) <= 0:
            raise ValueError('k must be a postitive integer')
        if int(buffer) <= 0:
            raise ValueError('buffer must be a positive integer')
        self.k = int(k)
        self.buffer = int(buffer)

        self.sub_svd_start = sub_svd_start
        if init_row_sampling_factor <= 0:
            raise ValueError('init_row_sampling_factor must be a positive value')
        self.init_row_sampling_factor = init_row_sampling_factor
        self.compute = True

    def _initialization(self, data, *args, **kwargs):
        vec_t = self.k + self.buffer

        if vec_t >= min(data.shape):
            raise ValueError('Cannot find more than min(n,p) singular values of array function.'
                             'Currently k = {}, buffer = {}. k + b >= min(n,p)'.format(self.k, self.buffer))

        if not isinstance(data, ScaledArray):
            self.scaled_array = ScaledArray(scale=self._scale, center=self._center, factor=self._factor)
            self.scaled_array.fit(data)
        else:
            self.scaled_array = data

        if self.sub_svd_start:
            x = sub_svd_init(self.scaled_array,
                             k=vec_t,
                             warm_start_row_factor=self.init_row_sampling_factor,
                             log=0)
        else:
            x = rnormal_start(self.scaled_array._array, vec_t, log=0)

        self.scaled_array.fit_x(x)

        return x

    def _solution_step(self, x, *args, **kwargs):
        q, _ = tsqr(x)
        x = self.scaled_array.sym_mat_mult(x=q)  # x <- AA'q
        return x.persist()

    def _parameter_step(self, x, *args, **kwargs):
        pass

    def _solution_accuracy(self, x, *args, **kwargs):
        U_k, S_k, V_k = subspace_to_SVD(x, self.scaled_array, sqrt_s=True, k=self.k, full_v=True, log=0)

        U_k, S_k, V_k = dask.persist(U_k, S_k, V_k)
        acc_list = []
        for method in self.scoring_method:
            if method == 'q-vals':
                try:
                    prev_S_k = self.history['iter']['S'][-1]
                    acc = q_value_converge(S_k, prev_S_k)
                except IndexError:
                    acc = float('INF')
                self.history['iter']['S'].append(S_k.compute())
            elif method == 'rmse':
                acc = rmse_k(self.scaled_array, U_k, S_k ** 2)
            else:  # method == 'v-subspace'
                try:
                    prev_V_k = self.history['iter']['V'][-1]
                    acc = subspace_dist(V_k.T, prev_V_k.T, S_k)
                except IndexError:
                    acc = float('INF')
                self.history['iter']['V'].append(V_k.compute())
            acc_list.append(acc)
        return acc_list

    def _finalization(self, x, *args, **kwargs):
        return subspace_to_SVD(x, self.scaled_array, sqrt_s=True, k=self.k, full_v=True, log=0)


class EigenPowerMethod(PowerMethod):

    def __init__(self, max_iter=50, k=5, max_buffer=50,
                 scoring_method='q-vals', p=1, tol=.1, init_row_sampling_factor=5):
        """
        Super with key word arguments.

        :param max_iter:
        :param k:
        :param max_buffer:
        :param scoring_method:
        :param p:
        :param init_row_sampling_factor:
        """
        super().__init__(max_iter=max_iter,
                         k=k,
                         scoring_method=scoring_method,
                         p=p,
                         tol=tol)
        self.init_row_sampling_factor = init_row_sampling_factor
        self.max_buffer = max_buffer

    def _initialization(self, data, *args, **kwargs):
        x = eigengap_init(data, k=self.k, b_max=self.max_buffer, warm_start_row_factor=self.init_row_sampling_factor,
                          tol=self.tol, log=0)

        m, n = x.shape

        self.buff_opt = n - self.k

        return x


class _vPowerMethod(PowerMethod):

    def __init__(self, v_start: ArrayType, k=None, buffer=None, max_iter=None, scoring_method=None, p=None, tol=None):
        super().__init__(max_iter=max_iter,
                         scoring_method=scoring_method,
                         p=p,
                         tol=tol)
        self.v_start = v_start.rechunk('auto')
        self.k = k
        self.buffer = buffer

    def _initialization(self, data, *args, **kwargs):
        self.scaled_array = data
        self.scaled_array.fit_x(self.v_start)
        xt = self.scaled_array.dot(self.v_start)
        return xt

    def _finalization(self, x, *args, **kwargs):
        return self.scaled_array.T.dot(x)


class SuccessiveStochasticPowerMethod2(PowerMethod):
    """ Implementation of Power Method that uses more and more of the matrix

       Suppose A exists in R(n by p)

       If we have reason to believe that that rows of A come from a small number of samples.

       Or, in other words a Truncated SVD with k << n explains much of the variance in the n rows of A.

       Therefore, it would be reasonable to assume that with just n' < n rows of A we can accurately learn the same
       factors that we would learn with all n rows.

       Algorithm:

           A <- Shuffle(A)
           I_rows = [0:n1, 0:n2, 0:n3, ..., 0:n]
           U0, S0, V0 <- Direct Truncated SVD of Small Subset Rows of A'

           convergence <- False
           until convergence:
               i = i + 1
               Ui, Si, Vi <- PowerMethod(A[I_rows[i], :], V_init=V{i-1})

               convergence = ||Si - S{i-1}||_2

           return Ui

       However, since shuffling an array in expensive, we will replace

           A <- Shuffle(A)
           I_rows = [0:n1, 0:n2, 0:n3, ..., 0:n]

       With:

           A <- A
           I_rows = shuffle([0:n1, n1:n2, n2:n3, ..., ni:n])

       However, through testing. It seems that

           Si are depressed from S_full by a factor.
           This factor seems to be of the form

               S_full ~= S_i * (a*sqrt(f_i -1) + 1) where f_i refers to the fraction of the array being used!
       """

    def __init__(self, k: int = 10,
                 max_sub_iter: int = 50,
                 scale: bool = True,
                 center: bool = True,
                 factor: Optional[str] = 'n',
                 scoring_method='q-vals',
                 p=1,
                 f=.1,
                 tol=.1,
                 buffer=10,
                 sub_svd_start: bool = True,
                 init_row_sampling_factor: int = 5,
                 max_sub_time=1000):
        super().__init__(k=k, max_iter=max_sub_iter, scale=scale, center=center, factor=factor,
                         scoring_method=scoring_method, p=p, tol=tol, buffer=buffer, sub_svd_start=sub_svd_start,
                         init_row_sampling_factor=init_row_sampling_factor, time_limit=max_sub_time)
        self.f = f
        self.history['acc'] = []
        self.history['times'] = {'start': None, 'stop': None, 'iter': []}

    def svd(self, array: Optional[LargeArrayType] = None, path_to_files: Optional[Union[str, Path]] = None,
            bed: Optional[Union[str, Path]] = None, bim: Optional[Union[str, Path]] = None,
            fam: Optional[Union[str, Path]] = None, *args, **kwargs):

        array = self.get_array(array, path_to_files, bed, bim, fam)

        self.history['times']['start'] = time.time()

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
            x = rnormal_start(sub_array._array, k=vec_t, log=0)

        for i, part in enumerate(partitions):
            _PM = _vPowerMethod(v_start=x, k=self.k, buffer=self.buffer, max_iter=self.max_iter,
                                scoring_method=self.scoring_method, p=self.p, tol=self.tol)

            x = _PM.svd(self.scaled_array[part, :], mask_nan=False)

            if 'v-subspace' in self.scoring_method:
                self.history['iter']['V'].append(_PM.history['iter']['V'][-1])
            self.history['iter']['S'].append(_PM.history['iter']['S'][-1])
            self.history['acc'].append(_PM.history['acc'])
            self.history['times']['iter'].append(_PM.history['times'])

        x = self.scaled_array.dot(x)
        U, S, V = subspace_to_SVD(x, self.scaled_array, k=self.k, full_v=True, log=0)
        self.history['times']['stop'] = time.time()
        return U, S, V
