import pytest
import numpy as np
from math import sqrt

from lmdec.array.metrics import subspace_dist, q_value_converge, rmse_k

num_run = 10
p = np.linspace(0, 10, 10)
norm_range = [0, -1, 1, float('inf'), 2, *p]
decimals = 6


def test_v_subspace_reflexive():
    """
    V1 = Anything
    V2 = V1
    S = Anything

    subspace_dist(V1, V2, S) == 0:
    """
    for _ in range(num_run):
        for N in range(1, 1002, 100):
            for K in range(1, min(52, N), 5):
                for P in norm_range:
                    V1, _ = np.linalg.qr(np.random.rand(N, K))
                    V2 = V1.copy()
                    S = np.random.randn(K) + 10

                    np.testing.assert_almost_equal(subspace_dist(V1, V2, S, P), 0, decimal=decimals)


def test_v_subspace_shuffle_columns():
    """
    V1 = Identity
    V2 = ColumnShuffle(Identity)
    S = Anything

    subspace_dist(V1, V2, S) == 0:
    """

    for N in range(2, 10):
        I = np.eye(N)

        I_shuffle = np.random.permutation(I.T).T

        S = np.random.randn(N) + 10
        for P in norm_range:
            np.testing.assert_almost_equal(subspace_dist(I, I_shuffle, S, P), 0, decimal=decimals)


def test_v_subspace_case2():
    """
    Same V with degenerate Singular Values
    """
    V1 = np.array([[1, 0],
                   [0, 1]])
    V2 = np.array([[1, 0],
                   [0, 1]])

    V2, _ = np.linalg.qr(V2)

    S = np.array([1, 0])
    for P in norm_range:
        if P == -1:
            np.testing.assert_almost_equal(subspace_dist(V1, V2, S, P), 1, decimal=decimals)
        else:
            np.testing.assert_almost_equal(subspace_dist(V1, V2, S, P), 0, decimal=decimals)


def test_v_subspace_case3():
    """
    Same V with non-degenerate Singular Values
    """
    V1 = np.array([[1, 0],
                   [0, 1]])
    V2 = np.array([[1, sqrt(2)/2],
                   [0, sqrt(2)/2]])

    V2, _ = np.linalg.qr(V2)

    S = np.array([1, 1])

    np.testing.assert_almost_equal(subspace_dist(V1, V2, S), 0, decimal=decimals)


def test_v_subspace_case4():
    """
    Different V with non-degenerate Singular Values
    """
    V1 = np.array([[1, 0],
                   [0, 1],
                   [0, 0]])
    V2 = np.array([[1, 0],
                   [0, 0],
                   [0, 1]])

    V2, _ = np.linalg.qr(V2)

    for p in np.logspace(0, 10):
        for a in np.linspace(0, 1):
            S = np.array([1, a])

            np.testing.assert_almost_equal(subspace_dist(V1, V2, S, power=p), (a ** p) / (1 ** p + a ** p),
                                           decimal=decimals)


def test_v_subspace_case5():
    a = np.random.randn(100, 50)

    U, S, V = np.linalg.svd(a, full_matrices=False)

    with pytest.raises(ValueError):
        subspace_dist(U, a.dot(V), S)


def test_q_vals_reflexive():
    for K in range(1, 10):
        si = np.random.rand(K)
        sj = si.copy()
        for norm in norm_range:
            for scale in [True, False]:

                assert q_value_converge(si, sj, norm=norm, scale=scale) == 0


def test_q_vals_case1():
    for a in np.linspace(0, 1):
        si = np.array([1, a])
        sj = np.array([1, 1])

        assert q_value_converge(si, sj, norm=2, scale=False) == 1-a

        assert q_value_converge(si, sj, norm=2, scale=True) == 2*(1-a) / (np.sqrt(2) + np.sqrt(1+a**2))

        assert q_value_converge(si, sj, norm=1, scale=False) == 1 - a
        np.testing.assert_almost_equal(q_value_converge(si, sj, norm=1, scale=True), 2*(1 - a)/(3 + a))

def test_q_vals_case2():
    with pytest.raises(ValueError):
        q_value_converge(np.array([1, 1, 1]), np.array([1, 1]))

    with pytest.raises(ValueError):
        q_value_converge(np.array([1, 1, 1]), np.array([1]))


def test_rmse_reflexive():
    for N in range(2, 10):
        for P in range(2, 10):
            array = np.random.rand(N, P)
            U, S, V = np.linalg.svd(array.dot(array.T))
            np.testing.assert_almost_equal(rmse_k(array, U, S), 0)


def test_rmse_case1():
    for s in np.linspace(0.1, 1):
        for N in range(2, 10):
            array = np.eye(N)
            for K in range(1, N):
                U = array[:, 0:K]
                S = s*np.ones(K)
                result = (1-s)/np.sqrt(N*N)
                np.testing.assert_almost_equal(rmse_k(array, U, S), result)


def test_rmse_case2():
    n, p = 10, 7
    a = np.random.random((n, p))
    aat = a.dot(a.T)

    U, S, _ = np.linalg.svd(aat, full_matrices=False)
    V, _, _ = np.linalg.svd(a.T.dot(U), full_matrices=False)
    np.testing.assert_almost_equal(rmse_k(a, U, S), 0)
    np.testing.assert_almost_equal(rmse_k(a.T, V, S[0:p]), 0)
