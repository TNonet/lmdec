import numpy as np
import dask.array as da

from lmdec.decomp.init_methods import v_init, sub_svd_init
from lmdec.array.scaled import ScaledArray
from lmdec.array.matrix_ops import svd_to_trunc_svd
from lmdec.array.metrics import subspace_dist


def test_v_init():
    N, P = 100, 40
    a = da.array(np.random.randn(N, P))

    U, S, V = da.linalg.svd(a)

    np.testing.assert_almost_equal(subspace_dist(U, v_init(a, V), S), 0)
    np.testing.assert_almost_equal(subspace_dist(U, v_init(a, V), S), 0)


def test_sub_svd_init():
    N, P = 100, 40
    k = 10
    a = da.array(np.random.randn(N, P))
    sa = ScaledArray(False, False, None)
    sa.fit(a)
    U, S, V = da.linalg.svd(a)
    Uk, Sk = svd_to_trunc_svd(u=U, s=S, k=k)

    U1 = sub_svd_init(sa, k=k, warm_start_row_factor=10, log=0)

    np.testing.assert_almost_equal(subspace_dist(U1, Uk, Sk), 0)


def test_sub_svd_init_warm_start_row_factor():
    N, P = 100, 40
    k = 10
    a = da.array(np.random.randn(N, P))
    sa = ScaledArray(False, False, None)
    sa.fit(a)
    U, S, V = da.linalg.svd(a)
    Uk, Sk = svd_to_trunc_svd(u=U, s=S, k=k)

    previous_error = 1
    for i in range(1, 11, 2):
        U1 = sub_svd_init(sa, k=k, warm_start_row_factor=i, log=0)

        assert subspace_dist(U1, Uk, Sk) <= previous_error
        previous_error = subspace_dist(U1, Uk, Sk)