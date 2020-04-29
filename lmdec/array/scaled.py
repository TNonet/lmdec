import warnings
from typing import Optional, Union, List, Tuple, Iterable
from functools import wraps

from copy import copy
import numpy as np
from dask import array as da

from lmdec.array.matrix_ops import diag_dot


def reshape_degenerate_2d_array(f):
    @wraps(f)
    def wrapped(self, x) -> Union[da.core.Array, np.ndarray]:
        reshape = False

        if len(x.shape) == 2 and x.shape[1] == 1:
            reshape = True
            x = np.squeeze(x)

        r = f(self, x)

        if reshape:
            r = r[:, np.newaxis]

        return r

    return wrapped


def combine_stds(itr: Iterable[Tuple[int, float, float]]):
    """Combines sample size, sample mean, sample std of N sets into one std

    Parameters
    ----------
    itr : Variable length list of 3-tuples
         [(sample size 1, sample mean 1, sample std 1),
          (sample size 2, sample mean 2, sample std 2),
          (     ....    ,      ....    ,     ...     ),
          (sample size N, sample mean N, sample std N)]

    Returns
    -------
    std : float
        combined std of N sets
    """
    copy_itr = tuple(itr)
    set_mu = combine_means(copy_itr)

    sum_weight_std = 0
    sum_sample_size = 0
    for n, mu, std in copy_itr:
        sum_weight_std += n*(std**2 + (mu - set_mu)**2)
        sum_sample_size += n

    return np.sqrt(sum_weight_std/sum_sample_size)


def combine_means(itr: Iterable[Union[Tuple[int, float], Tuple[int, float, float]]]) -> float:
    """Combines sample size, sample mean, of N sets into mean of all N sets

    Parameters
    ----------
    itr : Variable length list of 2-tuples
        [(sample size 1, sample mean 1),
         (sample size 2, sample mean 2),
         (     ....    ,      ....    ),
         (sample size N, sample mean N)]

    Returns
    -------
    u : float
        combined mean of N sets
    """
    sum_sample_size = 0
    sum_sample_mean = 0
    for n, mu, *_ in itr:
        sum_sample_size += n
        sum_sample_mean += n*mu

    return sum_sample_mean/sum_sample_size


class ArrayMoment:
    """Helper Class for storing, calculating, and interpolating column means and standard deviations
    """

    def __init__(self, a, batch_calc):
        self._array = a
        self._batch_calc = batch_calc

        # Values
        self._axis = 0
        self._std_tol = 1e-6

        # Caches
        self._center_batches = None
        self._scale_batches = None
        self._size_batches = None
        self._center_vector = None
        self._scale_vector = None
        self._sym_scale_vector = None
        self._vector_width = None
        self._scale_matrix = None
        self._sym_scale_matrix = None

    def fit_x(self, x):
        """Pre-computes scale matrix/vector based on the second dimension of x.

        If x is a 1-d or degenerate 2-d (N, 1) array.

        Parameters
        ----------
        x : array_like (?, K) or (?,)

        Returns
        -------
        None
        """
        if len(x.shape) > 2:
            raise ValueError('Cannot fit on a {}-d array.'.format(len(x.shape)))

        if len(x.shape) == 1:
            return
        else:
            _, k = x.shape
            self._vector_width = k

        x_h_placeholder = da.ones((self._array.shape[1], k))  # (P, k). k = 1
        self._scale_matrix = diag_dot(self.scale_vector, x_h_placeholder, return_diag=True)
        self._sym_scale_matrix = diag_dot(self.sym_scale_vector, x_h_placeholder, return_diag=True)

    def _std_inverter(self, std):
        """
        Parameters
        ----------
        std : array_like, shape  (P,)
            vector of standard deviations of the P rows of self._array

        Returns
        -------
        inv_std : array_like, shape (P,)
            vector of 1/std
        """

        try:
            std = std.compute()
        except AttributeError:
            pass

        degenerate_snp_columns = np.where(std <= self._std_tol)
        if len(degenerate_snp_columns[0]) > 0:
            warnings.warn('SNP Columns {} have low standard deviation.'
                          ' Setting STD of columns to 1'.format(degenerate_snp_columns))
            std[degenerate_snp_columns[0]] = 1

        return da.array(1 / std)

    def fit(self) -> None:
        """Uses batch_calc to compute mean and std

        Let Ai =  A[batch_calc[i] :]

        Then A = [A1'; A2'; A3'; ... AN']'

        Then;
            u_batchs = [ColumnMean(A1), ColumnMean(A2), ... ColumnMean(AN)]
            s_batchs = [ColumnStd(A1), ColumnStf(A2), ... ColumnStd(AN)]

        center_vector = combine_mean(u_batchs)
        scale_vector = combine_std(s_batchs)

        We can also find the center/scale_vector of sub batches by using subsections of u/s_batches

        Returns
        -------
        None
        """
        if self._array is not None:
            if self._batch_calc is None:
                self._center_vector = self._array.mean(axis=self._axis)
                self._scale_vector = self._std_inverter(self._array.std(axis=self._axis))
                self._sym_scale_vector = self._scale_vector**2
                return
            else:
                self._center_batches = []
                self._scale_batches = []
                self._size_batches = []
                for sub_index in self._batch_calc:
                    sub_array = self._array[sub_index, :]
                    num_rows = sub_array.shape[0]
                    sub_array_mean = sub_array.mean(axis=self._axis)
                    sub_array_std = sub_array.std(axis=self._axis)
                    self._center_batches.append(sub_array_mean)
                    self._scale_batches.append(sub_array_std)
                    self._size_batches.append(num_rows)

        self._center_vector = combine_means(zip(self._size_batches, self._center_batches))
        self._scale_vector = combine_stds(zip(self._size_batches, self._center_batches, self._scale_batches))
        self._sym_scale_vector = self._scale_vector**2

    @property
    def center_vector(self):
        if self._center_vector is None:
            raise ValueError('Must fit on array')
        return self._center_vector

    @property
    def scale_vector(self):
        if self._scale_vector is None:
            raise ValueError("Must fit on array")
        else:
            return self._scale_vector

    @property
    def vector_width(self):
        if self._vector_width is None:
            raise ValueError('Must fit on x')
        else:
            return self._vector_width

    @property
    def sym_scale_vector(self):
        if self._sym_scale_vector is None:
            raise ValueError("Must fit on array")
        else:
            return self._sym_scale_vector

    @property
    def scale_matrix(self):
        if self._scale_matrix is None:
            raise ValueError('Must fit on x for scale_matrix')
        else:
            return self._scale_matrix

    @property
    def sym_scale_matrix(self):
        if self._sym_scale_matrix is None:
            raise ValueError('Must fit on x for sym_scale_matrix')
        else:
            return self._sym_scale_matrix

    @property
    def vector_width(self):
        if self._vector_width is None:
            raise ValueError("Must fit on x for vector_width")
        else:
            return self._vector_width

    def __getitem__(self, item):
        """Returns a new ArrayMoment

        Parameters
        ----------
        item : index_like

        Returns
        -------
        """
        if isinstance(item, tuple) and len(item) > 2:
            raise IndexError("Cannot index with more than two dimensions.")

        try:
            indx = [i for i in range(len(self._batch_calc)) if self._batch_calc[i] in item]
            new_array_moment = ArrayMoment(self._array[indx, :], item)
            new_array_moment._center_batches = self._center_batches[indx]
            new_array_moment._scale_batches = self._scale_batches[indx]
            new_array_moment._size_batches = self._size_batches[indx]
            new_array_moment.fit()

        except (ValueError, TypeError):
            new_array_moment = ArrayMoment(self._array[item], None)
            new_array_moment.fit()

        return new_array_moment


class ScaledArray:
    """
    Class for efficiently scaled and centered matrix multiplications
        Columns of A have mean 0 if center = True
        Columns of A have std 0 if scale = True

    In addition give easy and efficient access to:
        Access to genetic relatedness matrix (GRM) through sym_mat_mult
            f*AA'x
        Access to A'

    We use the standard GRM definitions as:
        AA'
        where A is a {n \times p} matrix of standardized genotypes for
            n individuals
            p single nucleotide polymorphisms

        Therefore, AA' is a {n \times n}
    """

    def __init__(self, scale: bool = True, center: bool = True, factor: Optional[str] = None,
                 batch_calc: Optional[List[slice]] = None):
        self.scale = scale
        self.center = center
        self.batch_calc = batch_calc
        if factor not in [None, 'n', 'p']:
            raise ValueError('factor, {}, must be in [None, "n", "p"]'.format(factor))
        else:
            self.factor = factor
        self._axis = 0
        self._std_tol = 1e-6
        self._array = None
        self._array_moment = None
        self._factor_value = None
        self._t_cache = None
        self._t_flag = False  # Implying the array is not a transposition of the original fit

    def fit(self, a, x=None):
        """Finds mean and standard deviation of the columns of array (when specified)

        Parameters
        ----------
        a : array_like, shape (N, P)
        x : array_like, shape  (N, K), or (N, ) optional

        Notes
        --------
        Let A exist in {0,1,2}^(N times P)

        mu = mean of A for each SNP
            mu exists in R^P

        sig = standard deviation of A for each SNP
            sig exists in R^P
        """
        if self._t_flag:
            raise ValueError('Cannot fit on a Transposed Array. Use <instance>.T to recover base array')

        if len(a.shape) != 2:
            raise ValueError('Cannot fit non-2D array.')

        self._array = da.array(a)

        self._array_moment = ArrayMoment(self._array, self.batch_calc)

        self._array_moment.fit()

        if x is not None:
            self._array_moment.fit_x(x)

        if self.factor:
            n, p = a.shape
            self._factor_value = n if self.factor == 'n' else p

    def fit_x(self, x):
        self._array_moment.fit_x(x)

    @reshape_degenerate_2d_array
    def sym_mat_mult(self, x: Union[da.core.Array, np.ndarray]) -> Union[da.core.Array, np.ndarray]:
        """Performs Symmetrical Matrix Multiplication of array and x

        Parameters
        ----------
        x : array_like, shape (N, ) or (N, K)

        Returns
        -------
        y : array_like, shape (N, ) or (N, K)

        Notes
        -----

        x_k+1 <- AA'x_k

        Let B be the A with the SNP wise mean and standard deviations removed
        Therefore,

            B = (A - U)D Where

                ⌈:   :   :   ...  : ⌉
            U = |u1, u2, u3, ..., uP| = u'1, where 1 is {1 \times n}
                ⌊:   :   :   ...  : ⌋

            where u = [u1, u2, u3, ..., uP], where u is a {p \time 1}
                  ui is mean of ith column of A


            D = diag(d1, d2, d3, ...., dP)


        if not scaled or centered:
            y = AA'x_k

        if only centered:
            x_k_h = (A - U)'x_k
                  = A'x_k - U'x_k
                  = A'x_k - u'1x_k
               y  = (A - U)x_k_h
                  = Ax_k_h - Ux_k_h
                  = Ax_k_h - 1'ux_k_h

        looking at:
            u'1x
                   u'     1      x
                (p x 1)(1 x n)(n x k)
                       |------------|
                   u'  [sum(x1), sum(x2), ..., sum(xk)]

                [u1*sum(x1), u1*sum(x2), ... u1*sum(xk);
                 u2*sum(x1), u2*sum(x2), ... u2*sum(xk);
                 ...
                 up*sum(x1), up*sum(x2), ... up*sum(xk)]
            1'ux
                   1'     u      x
                (n x 1)(1 x p)(p x k)
                       |------------|
                   1'  [<u,x1>, <u, x2>, ..., <u,xk>]

                   [<u,x1>, <u, x2>, ..., <u,xk>;
                    <u,x1>, <u, x2>, ..., <u,xk>;
                   ...
                    <u,x1>, <u, x2>, ..., <u,xk>]


        if only scaled:
            y = ADDA'x_k

            Let D2 = DD (PreCompute D2)

            y = AD2A'x_k
        if centered and scaled
            x_k_h = B'x_k
                  = D(A - U)'x_k
                  = D(A'x_k - U'x_k)
                  = D(A'x_k - u'1x_k)
               y  = Bx_k_h
                  = (A - U)Dx_k_h
                  = ADx_k_h - UDx_k_h
                  = ADx_k_h - 1'uDx_k_h

            We can move D around:
            x_k_h = DD(A'x_k - u'1x_k)
                y = Ax_k_h - 1'ux_k_h
        """
        if self._t_flag:
            raise NotImplementedError('sym_mat_mult is only implemented for non-transposed arrays')

        x_h = self._array.T.dot(x)
        if self.center:
            # Computes mu1'x_k_h
            # -= is not implemented full for dask arrays
            x_h = self._center_x(x=x_h, dx=x, transpose=True)

        if self.scale:
            # Computes Dx_k
            x_h = self._scale_x(x=x_h, sym=True)

        y = self._array.dot(x_h)
        if self.center:
            y = self._center_x(x=y, dx=x_h, transpose=False)

        if self.factor:
            y /= self._factor_value

        return y

    @property
    def T(self) -> "ScaledArray":
        if self._t_cache is None:
            self._t_cache = self._transpose()
            self._t_cache._t_cache = self
        return self._t_cache

    @property
    def array(self) -> Union[da.core.Array, np.ndarray]:
        """Returns scaled and centered array

        Returns
        -------
        a : array_like, shape (N, P)

        Notes:
        -----
        A in R(N times P)
        B = (A - mu)D
        """
        n, p = self.shape
        if n * p >= 1e9:
            warnings.warn('Array is Large, {}'.format(self.shape))
        new_array = self._array
        if self.center:
            new_array = new_array - self.center_vector
        if self.scale:
            new_array = diag_dot(self.scale_vector, new_array.T).T

        if isinstance(new_array._meta, np.ma.core.MaskedArray):
            new_array = da.ma.filled(new_array)

        if self._t_flag:
            # These are explicitly switched due the
            return new_array.T
        else:
            return new_array

    def rechunk(self, rechunk_info):
        new_scaled_array = copy(self)
        try:
            new_scaled_array._array = new_scaled_array._array.rechunk(rechunk_info)
        except AttributeError:
            pass
        return new_scaled_array

    def _transpose(self) -> "ScaledArray":
        t_scaled_array = copy(self)
        t_scaled_array._t_flag = not self._t_flag
        return t_scaled_array

    def _scale_x(self, x, sym: bool = False) -> Union[da.core.Array, np.ndarray]:
        try:
            if len(x.shape) == 2 and self._array_moment.vector_width == x.shape[1]:
                scale_matrix = self._array_moment.sym_scale_matrix if sym else self._array_moment.scale_matrix
                return np.multiply(scale_matrix, x)
        except ValueError:
            pass
        scale_vector = self._array_moment.sym_scale_vector if sym else self._array_moment.scale_vector
        x_h = diag_dot(scale_vector, x, return_diag=False)
        return x_h

    def _center_x(self, x, dx, transpose: bool = False) -> Union[da.core.Array, np.ndarray]:
        if transpose:
            # Computes mu1'x_k_h
            return x - np.squeeze(np.outer(self._array_moment.center_vector, dx.sum(axis=0)))
        else:
            return x - self._array_moment.center_vector.dot(dx)

    @reshape_degenerate_2d_array
    def dot(self, x: Union[da.core.Array, np.ndarray]) -> Union[da.core.Array, np.ndarray]:
        if not self._t_flag and self.scale:
            x = self._scale_x(x, sym=False)

        if self._t_flag:
            y = self._array.T.dot(x)
        else:
            y = self._array.dot(x)

        if self.center:
            y = self._center_x(x=y, dx=x, transpose=self._t_flag)

        if self._t_flag and self.scale:
            y = self._scale_x(y, sym=False)

        return y

    @property
    def center_vector(self):
        return self._array_moment.center_vector

    @property
    def factor_value(self):
        if self._factor_value is None:
            return self.factor
        else:
            return self._factor_value

    @property
    def scale_vector(self):
        return self._array_moment.scale_vector

    @property
    def shape(self):
        if self._t_flag:
            return tuple(reversed(self._array.shape))
        else:
            return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def chunks(self):
        if self._t_flag:
            return self._array.T.chunks
        else:
            return self._array.chunks

    def __getitem__(self, item):
        """Creates and Returns a new ScaledArray, with the appropriate underlying array, with the proper state

        Parameters
        ----------
        item : index like or tuple of index_like

        Returns
        -------
        sub_array : ScaledArray

        Notes
        -----

        A = [[a11, a12, a13, ..., a1p],
             [a21, a22, a23, ..., a2p],
             [ . ,  . ,  . , ...,  . ],
             [an1, an2, an3, ..., amp]]

        A[row_index_item, column_index_item]

        """
        if isinstance(item, tuple) and len(item) > 2:
            raise IndexError('too many indices for array')

        new_scaled_array = ScaledArray(self.scale, self.center, self.factor)
        if self._t_flag:
            new_scaled_array.fit(self._array.T[item])
        else:
            new_scaled_array.fit(self._array[item])

        return new_scaled_array
