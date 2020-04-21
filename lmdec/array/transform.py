from typing import Tuple, Union

import numpy as np

from .types import ArrayType, LargeArrayType


def acc_format_svd(u: ArrayType,
                   s: ArrayType,
                   array_shape: Tuple[int, int],
                   square: bool = True,
                   log: int = 0) -> Tuple[ArrayType, ArrayType]:
    """
    Handles variable shapes and formats for U, S and returns
    proper formatting U, S for error calculations.

    Singular vectors are assumed ground truth and will adjust singular values

    :param u: Left Singular Vectors
    :param s: Singular Values or square root of Singluar Values
    :param array_shape:
    :param square: Indictor of state of s.
    :param log:

    :return:
    """
    if log:
        raise Exception("Does not return Logging")

    m, n = array_shape

    if len(u.shape) == 1:
        # Single singular vector -> expects 1 singular value
        m_vector = u.shape[1]
        k_vector = 1
    elif len(u.shape) == 2:
        m_vector, k_vector = u.shape
    else:
        raise Exception('u is of the wrong shape, {}. \n Must be of shape: ({}, k) or ({},)'.format(
            u.shape, m, m))

    s_copy = s.copy()
    if square:
        s_copy = np.square(s_copy)

    if len(s_copy.shape) == 1:
        k_values = s_copy.shape[0]
        s_copy: ArrayType = np.diag(s_copy)
    elif len(s_copy.shape) == 2:
        k_values1, k_values2 = s_copy.shape
        if k_values1 != k_values2:
            raise Exception('s is the wrong shape, {}. \n Must be of shape ({}, {}) or ({},)'.format(
                s_copy.shape,
                k_vector,
                k_vector,
                k_vector))
        k_values = k_values1
        s_copy: ArrayType = np.diag(np.diag(s_copy))
    else:
        raise Exception('s is the wrong shape, {}. \n Must be of shape ({}, {}) or ({},)'.format(
            s_copy.shape,
            k_vector,
            k_vector,
            k_vector))

    if m == m_vector:
        pass
    elif m != m_vector and m == k_vector:
        # U might be coming in as Transpose U
        u = u.T
        m_vector, k_vector = k_vector, m_vector
    else:
        raise Exception('u or s is of the wrong shape')

    if k_values != k_vector:
        raise Exception('u or s is of the wrong shape')

    return u, s_copy


def svd_to_pca(array: LargeArrayType,
               u: ArrayType,
               s: ArrayType,
               log: int = 1) -> Union[Tuple[ArrayType, ArrayType, dict],
                                      Tuple[ArrayType, ArrayType]]:
    """Converts Left Singular Vectors and Singular Values to Principal Components and EigenValues

    Let A be a matrix of size (n x p)

    Let lamabda_i, v_i be the i_th eigenpair of A

    """
    raise NotImplementedError


def pca_to_svd(array: LargeArrayType,
               u: ArrayType,
               s: ArrayType,
               log: int = 1) -> Union[Tuple[ArrayType, ArrayType, ArrayType, dict],
                                      Tuple[ArrayType, ArrayType, ArrayType]]:
    raise NotImplementedError