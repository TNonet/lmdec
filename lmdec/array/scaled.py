import warnings
from typing import Optional, Union
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


class ArrayMoment:
    """Helper Class for storing, calculating, and interpolating column means and standard deviations
    """

    def __init__(self, a, std_dist='normal', warn: bool = True):
        self._array = a

        if std_dist not in ['binom', 'normal']:
            raise ValueError(f"std_dist must be either `binom` or `normal`")
        self._std_dist = std_dist
        self._warn = warn
        # Values
        self._axis = 0
        self._std_tol = 1e-6

        # Caches
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
            if self._warn:
                warnings.warn('SNP Columns {} have low standard deviation.'
                              ' Setting STD of columns to 1'.format(degenerate_snp_columns))
            std[degenerate_snp_columns[0]] = 1

        return da.array(1 / std)

    def fit(self) -> None:
        """
        Returns
        -------
        None
        """
        self._center_vector = self._array.mean(axis=self._axis)
        if self._std_dist == 'normal':
            self._scale_vector = self._std_inverter(self._array.std(axis=self._axis))
        else:
            p = self._center_vector / 2
            self._scale_vector = self._std_inverter(da.sqrt(2 * p * (1 - p)))
        self._sym_scale_vector = self._scale_vector ** 2

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

        new_array_moment = ArrayMoment(self._array[item])
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
                 std_dist: Optional[str] = None, warn: bool = True):
        self.scale = scale
        self.center = center
        if factor not in [None, 'n', 'p']:
            raise ValueError('factor, {}, must be in [None, "n", "p"]'.format(factor))
        else:
            self.factor = factor
        if std_dist not in [None, 'normal', 'binom']:
            raise ValueError(f"std_dist must be in [None, 'normal', 'binom']")
        elif std_dist is None:
            std_dist = 'normal'

        self._std_dist = std_dist
        self._warn = warn
        self._axis = 0
        self._std_tol = 1e-6
        self._array = None
        self._array_moment = None
        self._factor_value = None
        self._t_cache = None
        self._t_flag = False  # Implying the array is not a transposition of the original fit

    @classmethod
    def fromScaledArray(cls, array: da.core.Array, scaled_array: "ScaledArray",
                        factor: Optional[str] = None) -> "ScaledArray":
        """ Creates a ScaledArray with the same scale and center vectors as `scaled_array` but with the underlying array
        of `array`.

        Parameters
        ----------
        array : array_like, shape (?, P)
            array to base a ScaledArray on top of.
        scaled_array : ScaledArray, fit on array of shape (?, P)
            ScaledArray to collect scale and center vectors from.
        factor : str, optional
            if None: new ScaledArray will have a factor_value of 1.
            if 'n' factor value will come from array.shape[0]
            if 'p' factor value will come from array.shape[1]

        Returns
        -------
        scaled_array : ScaledArray
            ScaledArray with underlying array as `array` and underlying moments from `array_moment`
        """
        if (scaled_array.scale and array.shape[1] != len(scaled_array.scale_vector))\
                or (scaled_array.center and array.shape[1] != len(scaled_array.center_vector)):
            array_moment_shape = len(scaled_array.scale_vector) if scaled_array.scale is not None\
                else len(scaled_array.center_vector)
            raise ValueError(f'expected array of shape {array.shape} to have matching second dimension '
                             f'with array_moment shape, {array_moment_shape}')

        sa = ScaledArray(scale=scaled_array.scale, center=scaled_array.center, factor=factor,
                         std_dist=scaled_array._std_dist, warn=scaled_array._warn)
        sa._array = array
        sa._array_moment = ArrayMoment(a=array, std_dist=scaled_array._std_dist, warn=scaled_array._warn)
        if sa.factor:
            n, p = array.shape
            sa._factor_value = n if sa.factor == 'n' else p
        sa._array_moment._center_vector = scaled_array.center_vector
        sa._array_moment._scale_vector = scaled_array.scale_vector
        return sa

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

        self._array_moment = ArrayMoment(self._array, std_dist=self._std_dist, warn=self._warn)

        self._array_moment.fit()

        if x is not None:
            self._array_moment.fit_x(x)

        if self.factor:
            n, p = a.shape
            self._factor_value = n if self.factor == 'n' else p

    def fit_x(self, x):
        self._array_moment.fit_x(x)

    @reshape_degenerate_2d_array
    def sym_mat_mult(self, x: Union[da.core.Array, np.ndarray]) -> da.core.Array:
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
        if self._warn and n * p >= 1e9:
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

    def _scale_x(self, x, sym: bool = False) -> da.core.Array:
        """ Scales the product of a matrix multiplication instead of the matrix itself

        Let A be a matrix of shape (n by p) with non zero column stds, D of shape (p,).

        Matrix B could be constructed as follows with zero column std.
            B = A*Inv(Diag(D))
        However, this is inefficient if only the matrix product of B, with a matrix x is needed.
        Instead `_scale_x` implements:
            Ax*Inv(Diag(D))
            ^^
            x being passed in already computed as Ax
        with efficient broadcasting.

        Parameters
        ----------
        x : array_like
            Usually the product of Ax that needs to be scaled
        sym : bool
            Flag whether we are scaling twice in the case of AA'x
            The square of the column standard deviations must be removed

        Returns
        -------
        x_scaled : array_like

        """
        try:
            # self._array_moment.vector_width is not set until ScaledArray is fit_x.
            if len(x.shape) == 2 and self._array_moment.vector_width == x.shape[1]:
                scale_matrix = self._array_moment.sym_scale_matrix if sym else self._array_moment.scale_matrix
                return da.multiply(scale_matrix, x)
        except ValueError:
            pass
        scale_vector = self._array_moment.sym_scale_vector if sym else self._array_moment.scale_vector
        x_scaled = diag_dot(scale_vector, x, return_diag=False)
        return x_scaled

    def _center_x(self, x, dx, transpose: bool = False) -> da.core.Array:
        """ Centers the product of matrix multiplication instead of center the matrix

        Let A be a matrix of shape (n by p) with non zero column means, U of shape (p,).

        Matrix B could be constructed as follows with zero column mean.
            B = A - 1'U where 1 is a 1 vector. And 1'U is an outer product of shape (n by p)
        However, this is inefficient if only the matrix product of B, with a matrix x is needed.
        Instead `_center_x` implements:

            Ax - Ux
             ^    ^- dx being passed in,
             |
             x being passed in
        with efficient broadcasting.


        Parameters
        ----------
        x : array_like
            Usually the product of Ax that needs to be center
        dx : array_like
            Usually the original x before being multiplied by A
        transpose : bool
            Flag whether to indicate if A'x or Ax. Adjusts dimensions

        Returns
        -------
        x_centered: array_like
        """
        if transpose:
            # Computes mu1'x_k_h
            return x - da.squeeze(da.outer(self._array_moment.center_vector, dx.sum(axis=0)))
        else:
            return x - self._array_moment.center_vector.dot(dx)

    @reshape_degenerate_2d_array
    def dot(self, x: Union[da.core.Array, np.ndarray]) -> da.core.Array:
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

        new_scaled_array = ScaledArray(scale=self.scale,
                                       center=self.center,
                                       factor=self.factor,
                                       std_dist=self._std_dist,
                                       warn=self._warn)
        if self._t_flag:
            new_scaled_array.fit(self._array.T[item])
        else:
            new_scaled_array.fit(self._array[item])

        return new_scaled_array
