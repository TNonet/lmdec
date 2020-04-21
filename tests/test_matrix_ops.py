import pytest
import numpy as np
import dask.array as da

from lmdec.array.matrix_ops import diag_dot, subspace_to_SVD
from lmdec.array.metrics import subspace_dist

decimals = 6
num_runs = 10


def test_diag_dot_math():
    for K in range(2, 10):
        d = np.random.randn(K)
        d_diag = np.diag(d)
        for N in range(1, 10):
            x = np.random.randn(K, N)

            result = d_diag.dot(x)

            np.testing.assert_array_equal(diag_dot(d, x), result)
            np.testing.assert_array_equal(diag_dot(d[:, np.newaxis], x), result)

            np.testing.assert_array_equal(diag_dot(d, np.squeeze(x)), np.squeeze(result))
            np.testing.assert_array_equal(diag_dot(d[:, np.newaxis], np.squeeze(x)), np.squeeze(result))


def test_diag_dot_error():
    with pytest.raises(ValueError):
        d = np.random.rand(10)
        x = np.random.randn(10, 10, 10)
        diag_dot(d, x)

    with pytest.raises(ValueError):
        d = np.random.rand(10)
        x = np.random.randn(8, 10)
        diag_dot(d, x)

    with pytest.raises(ValueError):
        d = np.random.rand(10)
        x = np.random.randn(8)
        diag_dot(d, x)

    with pytest.raises(ValueError):
        d = np.random.rand(10, 10, 10)
        x = np.random.randn(10, 10)
        diag_dot(d, x)


def test_subspace_to_SVD_case1():
    """
    A = USV

    U = I
    S = np.range(N, 0, -1)
    V = I

    A = np.diag(np.range(N, 0, -1))

    subspace_to_SVD should recover I and S from sections of A
    """
    for N in range(2, 10):
        A = da.diag(da.arange(N, 0, -1))
        for j in range(2, N + 1):
            subspace = A[:, 0:j]
            for sqrt in [True, False]:
                U, S, V = subspace_to_SVD(subspace, A, sqrt_s=sqrt)

                np.testing.assert_array_almost_equal(U, (subspace != 0).astype(int))
                if sqrt:
                    np.testing.assert_array_almost_equal(S, da.sqrt(da.arange(N, N - j, -1)))
                else:
                    np.testing.assert_array_almost_equal(S, da.arange(N, N - j, -1))


def test_subspace_to_SVD_case2():
    """
    A = USV

    U = [e1 e2, ..., ek] \
        [0,  0, ...,  0] | N by K
        [1,  1, ...,  1] /


    S = np.range(N, 0, -1)
    V = I

    A = np.diag(np.range(N, 0, -1))

    subspace_to_SVD should recover I and S from sections of A
    """
    for N in range(2, 10):
        for K in range(2, N + 1):
            U = np.zeros((N, K))
            U[N - 1, :] = np.ones(K)
            U[:K, :K] = np.eye(K)
            V = da.eye(K)

            U = da.array(U)
            U_q, _ = da.linalg.qr(U)

            S = da.arange(K, 0, -1)

            A = U.dot(da.diag(S))
            for j in range(K, N + 1):
                subspace = A[:, 0:j]
                U_s, S_s, V_s = subspace_to_SVD(subspace, A, full_v=True)
                np.testing.assert_almost_equal(subspace_dist(V_s, V, S), 0, decimal=decimals)
                _, l, _ = da.linalg.svd(U_q.dot(U_s.T))
                np.testing.assert_almost_equal(l[:K].compute(), np.ones(K))
                np.testing.assert_almost_equal(l[K:].compute(), np.zeros(N - K))


def test_subspace_to_SVD_case3():
    """
    A = N(0, 1, size = (N,P))

    USV = SVD(A)

    A = np.diag(np.range(N, 0, -1))

    """
    for N in range(2, 10):
        for P in range(2, 10):
            A = da.random.random(size=(N, P))
            U, S, V = da.linalg.svd(A)
            for _ in range(num_runs):
                subspace_shuffle = da.random.permutation(U.T).T
                for j in range(2, P):
                    subspace = subspace_shuffle[:, 0:j]
                    U_s, S_s, V_s = subspace_to_SVD(subspace, A, full_v=True)
                    np.testing.assert_almost_equal(subspace_dist(V_s, V, S_s), 0, decimal=decimals)
