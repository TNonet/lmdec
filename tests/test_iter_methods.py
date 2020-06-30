import pytest

import numpy as np
from lmdec.decomp import PowerMethod, SuccessiveBatchedPowerMethod
from lmdec.array.metrics import subspace_dist
from lmdec.array.matrix_ops import svd_to_trunc_svd
from lmdec.decomp.utils import make_snp_array

import dask.array as da


def test_PowerMethod_case1():
    n = 100
    p = 80
    array = np.random.rand(100, 80)
    mu = array.mean(axis=0)
    std = np.diag(1/array.std(axis=0))
    scaled_centered_array = (array-mu).dot(std)
    U, S, V = np.linalg.svd(scaled_centered_array, full_matrices=False)  # Ground Truth
    array = make_snp_array(da.array(array), mean=True, std=True,
                           std_method='norm', mask_nan=False, dtype='float64')
    for k in range(1, 10):
        U_k, S_k, V_k = U[:, :k], S[:k], V[:k, :]

        PM = PowerMethod(k=k, tol=1e-9, scoring_method='rmse', max_iter=100, sub_svd_start=False,
                         init_row_sampling_factor=1, factor=None, lmbd=0)
        U_k_PM, S_k_PM, V_k_PM = PM.svd(array)

        np.testing.assert_array_almost_equal(S_k, S_k_PM)
        assert V_k.shape == V_k_PM.shape == (k, p)
        assert U_k.shape == U_k_PM.shape == (n, k)
        np.testing.assert_almost_equal(subspace_dist(V_k, V_k_PM, S_k_PM), 0)
        np.testing.assert_almost_equal(subspace_dist(U_k, U_k_PM, S_k_PM), 0)


def test_PowerMethod_case2():
    array = np.random.rand(100, 100)
    mu = array.mean(axis=0)
    std = np.diag(1/array.std(axis=0))
    scaled_centered_array = (array-mu).dot(std)
    array = make_snp_array(da.array(array), mean=True, std=True,
                           std_method='norm', mask_nan=False, dtype='float64')
    U, S, V = np.linalg.svd(scaled_centered_array.dot(scaled_centered_array.T), full_matrices=False)  # Ground Truth
    _, _, V = np.linalg.svd(scaled_centered_array.T.dot(scaled_centered_array), full_matrices=False)

    S = np.sqrt(S)
    k = 10
    U_k, S_k, V_k = U[:, :k], S[:k], V[:k, :]
    previous_S_error = float('inf')
    previous_U_error = float('inf')
    previous_V_error = float('inf')
    for t in np.logspace(0, -12, 20):

        PM = PowerMethod(k=k, tol=t, scoring_method='q-vals', max_iter=100, factor=None, lmbd=0)
        U_k_PM, S_k_PM, V_k_PM = PM.svd(array)

        assert subspace_dist(U_k, U_k_PM, S_k) <= previous_U_error
        assert subspace_dist(V_k, V_k_PM, S_k) <= previous_V_error
        assert np.linalg.norm(S_k-S_k_PM) <= previous_S_error
        previous_S_error = np.linalg.norm(S_k-S_k_PM)
        previous_U_error = subspace_dist(U_k, U_k_PM, S_k)
        previous_V_error = subspace_dist(V_k, V_k_PM, S_k)

    assert subspace_dist(U_k, U_k_PM, S_k) <= 1e-9
    assert subspace_dist(V_k, V_k_PM, S_k) <= 1e-9
    assert np.linalg.norm(S_k - S_k_PM) <= 1e-12


def test_PowerMethod_subsvd_finds_eigenvectors():
    N = 1000
    k = 10
    s_orig = np.array([1.01**i for i in range(1, N+1)])
    array = da.diag(s_orig)

    PM = PowerMethod(tol=1e-16, factor=None, lmbd=.1, max_iter=100)

    U_PM, S_PM, V_PM = PM.svd(array)

    np.testing.assert_array_almost_equal(s_orig[-k:][::-1], S_PM, decimal=0)  # Max Q-Val is 21,000

    v = np.zeros_like(V_PM).compute()

    for j in range(k):
        v[j, N - k + j] = 1

    np.testing.assert_almost_equal(subspace_dist(V_PM, v, S_PM), 0, decimal=5)


def test_PowerMethod_subsvd_finds_eigenvectors_failure():
    N = 1000
    k = 10
    s_orig = np.array([1.01**i for i in range(1, N+1)])
    array = da.diag(s_orig)

    PM = PowerMethod(tol=1e-16, lmbd=0, max_iter=100)

    U_PM, S_PM, V_PM = PM.svd(array)

    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(s_orig[-k:][::-1], S_PM, decimal=0)


def test_PowerMethod_all_tols_agree():
    n = 100
    p = 80
    k = 10
    array = np.random.rand(n, p)

    PM = PowerMethod(k=k, tol=1e-9, scoring_method='q-vals', max_iter=100, lmbd=0)
    U_q, S_q, V_q = PM.svd(array)

    PM = PowerMethod(k=k, tol=1e-4, scoring_method='rmse', max_iter=100, lmbd=0)
    U_r, S_r, V_r = PM.svd(array)

    PM = PowerMethod(k=k, tol=1e-9, scoring_method='v-subspace', max_iter=100, lmbd=0)
    U_v, S_v, V_v = PM.svd(array)

    np.testing.assert_array_almost_equal(S_q, S_r)
    np.testing.assert_array_almost_equal(S_q, S_v)

    np.testing.assert_almost_equal(subspace_dist(U_q, U_r, S_q), 0)
    np.testing.assert_almost_equal(subspace_dist(U_q, U_v, S_q), 0)

    np.testing.assert_almost_equal(subspace_dist(V_q, V_r, S_q), 0)
    np.testing.assert_almost_equal(subspace_dist(V_q, V_v, S_q), 0)


def test_PowerMethod_factor():
    n = 100
    p = 80
    array = np.random.rand(n, p)
    sym_array = array.dot(array.T)

    for f in ['n', 'p', None]:
        if f == 'n':
            factor = n
        elif f == 'p':
            factor = p
        else:
            factor = 1

        U, S, V = np.linalg.svd(sym_array/factor, full_matrices=False)
        S = np.sqrt(S)
        k = 10
        U_k, S_k, V_k = U[:, :k], S[:k], V[:k, :]

        array = make_snp_array(da.array(array), mean=False, std=False,
                               std_method='norm', mask_nan=False, dtype='float64')
        PM = PowerMethod(k=k, tol=1e-9, scoring_method='q-vals', max_iter=100, factor=f, lmbd=0)

        U_k_PM, S_k_PM, V_k_PM = PM.svd(array)

        np.testing.assert_array_almost_equal(S_k, S_k_PM)
        assert U_k.shape == U_k_PM.shape == (n, k)
        np.testing.assert_almost_equal(subspace_dist(U_k, U_k_PM, S_k_PM), 0)


def test_PowerMethod_scale_center():
    array = np.random.rand(100, 70)
    mu = array.mean(axis=0)
    std = np.diag(1 / array.std(axis=0))
    k = 10
    for scale in [True, False]:
        for center in [True, False]:
            new_array = array
            if center:
                new_array = new_array - mu
            if scale:
                new_array = new_array.dot(std)

            U, S, _ = np.linalg.svd(new_array.dot(new_array.T), full_matrices=False)  # Ground Truth
            _, _, V = np.linalg.svd(new_array.T.dot(new_array), full_matrices=False)  # Ground Truth
            S = np.sqrt(S)
            U_k, S_k, V_k = U[:, :k], S[:k], V[:k, :]

            snp_array = make_snp_array(da.array(array), std=scale, mean=center, std_method='norm', dtype='float64')

            np.testing.assert_array_almost_equal(new_array, snp_array)
            PM = PowerMethod(k=k, tol=1e-12, scoring_method='q-vals', max_iter=100,
                             factor=None, lmbd=0)
            U_q, S_q, V_q = PM.svd(snp_array)

            assert subspace_dist(U_k, U_q, S_k) <= 1e-8
            assert subspace_dist(V_k, V_q, S_k) <= 1e-8
            assert np.linalg.norm(S_k - S_q) <= 1e-9


def test_PowerMethod_multi_tol():
    array = np.random.rand(100, 70)
    k = 10
    tol_groups = [[1e-1, 1e-1, 1e-1],
                  [1e-16, 1e-4, 1-16],
                  [1e-6, 1e-16, 1e-16],
                  [1e-16, 1e-16, 1e-6]]

    for tols in tol_groups:
        PM = PowerMethod(k=k, tol=tols, scoring_method=['q-vals', 'rmse', 'v-subspace'], factor=None, lmbd=0)

        _, _, _ = PM.svd(array)

        assert (min(PM.history.acc['q-vals']) <= tols[0]
                or min(PM.history.acc['rmse']) <= tols[1]
                or min(PM.history.acc['v-subspace']) <= tols[2])

    fail_test_tols = [1e-6, 1e-16, 1e-16]

    PM = PowerMethod(k=k, tol=fail_test_tols, scoring_method=['q-vals', 'rmse', 'v-subspace'], factor=None, lmbd=0)

    _, _, _ = PM.svd(array)

    assert (min(PM.history.acc['q-vals']) <= fail_test_tols[0]
            and min(PM.history.acc['rmse']) >= fail_test_tols[1]
            and min(PM.history.acc['v-subspace']) >= fail_test_tols[2])


def test_PowerMethod_k():
    with pytest.raises(ValueError):
        _ = PowerMethod(k=0)

    with pytest.raises(ValueError):
        _ = PowerMethod(k=-1)

    PM = PowerMethod(k=100)
    array = np.random.rand(50, 50)
    with pytest.raises(ValueError):
        _, _, _ = PM.svd(array)


def test_PowerMethod_max_iter():
    with pytest.raises(ValueError):
        _ = PowerMethod(max_iter=0)

    PM = PowerMethod(max_iter=1)
    array = np.random.rand(100, 100)

    _, _, _ = PM.svd(array)

    assert PM.num_iter == 1
    assert len(PM.history.iter['S']) == 1


def test_PowerMethod_time_limit():
    with pytest.raises(ValueError):
        _ = PowerMethod(time_limit=0)

    PM = PowerMethod(time_limit=10, max_iter=int(1e10), tol=1e-20)
    array = np.random.rand(100, 100)

    _, _, _ = PM.svd(array)

    assert 9 <= PM.time <= 11


@pytest.mark.skip('Have yet to implement stochastic power methods efficiently within dask.')
def test_PowerMethod_p():
    with pytest.raises(ValueError):
        _ = PowerMethod(p=-1)

    with pytest.raises(ValueError):
        _ = PowerMethod(p=1)


def test_PowerMethod_buffer():
    with pytest.raises(ValueError):
        _ = PowerMethod(buffer=-1)

    array = np.random.rand(1000, 1000)
    U, S, V = np.linalg.svd(array)
    S[0:15] = 1e3 + S[0:15]

    array = U.dot(np.diag(S).dot(V))

    for method in ['q-vals', 'rmse', 'v-subspace']:
        PM1 = PowerMethod(buffer=2, max_iter=3, tol=1e-16, scoring_method=method)
        PM2 = PowerMethod(buffer=30, max_iter=3, tol=1e-16, scoring_method=method)

        _, _, _ = PM1.svd(array)
        _, _, _ = PM2.svd(array)

        if method == 'v-subspace':
            # V-Subspace for this convergese quickly due small size of test case
            assert np.abs(PM1.history.acc[method][-1] - PM2.history.acc[method][-1]) < 1e-4
        else:
            assert PM1.history.acc[method][-1] > PM2.history.acc[method][-1]


def test_PowerMethod_sub_svd_start():
    array = np.random.rand(100, 100)

    PM1 = PowerMethod(sub_svd_start=True, tol=1e-12, lmbd=0)
    _, S1, _ = PM1.svd(array)

    PM2 = PowerMethod(sub_svd_start=False, tol=1e-12, lmbd=0)
    _, S2, _ = PM2.svd(array)

    np.testing.assert_array_almost_equal(S1, S2)


def test_PowerMethod_init_row_sampling_factor():
    with pytest.raises(ValueError):
        _ = PowerMethod(init_row_sampling_factor=0)

    PM1 = PowerMethod(init_row_sampling_factor=4)
    array = np.random.rand(100, 100)
    _, _, _ = PM1.svd(array)


def test_PowerMethod_bad_arrays():
    PM = PowerMethod()

    with pytest.raises(ValueError):
        PM.svd(np.random.rand(100))

    with pytest.raises(ValueError):
        PM.svd(np.random.rand(100, 100, 100))


def test_PowerMethod_nan_arrays():
    array = np.random.randn(100, 100)
    for bad_type in [float('nan')]:
        array[0, 0] = bad_type
        for start in [True, False]:
            PM = PowerMethod(sub_svd_start=start, max_iter=2)
            with pytest.raises(np.linalg.LinAlgError):
                _, _, _ = PM.svd(da.array(array))

            clean_array = make_snp_array(da.array(array), mask_nan=True, std_method='norm', dtype='float64')
            _, _, _ = PM.svd(clean_array)


@pytest.mark.skip
def test_PowerMethod_nan_arrays_fills():
    array = np.random.randint(0, 3, size=(100, 100)).astype(float)
    array[0, 0] = 10000
    median = round(np.median(array))
    mean = round(np.mean(array))
    k = 10

    for method in ['mean', 'median', 10]:

        PM = PowerMethod(factor=None, scale=False, center=False, tol=1e-16, lmbd=0)
        if method == 'mean':
            filled_value = mean
        elif method == 'median':
            filled_value = median
        else:
            filled_value = method
        array[0, 1] = filled_value

        U, S, V = np.linalg.svd(array, full_matrices=False)
        U_k, S_k, V_k = svd_to_trunc_svd(U, S, V, k=k)

        array[0, 1] = float('nan')
        U, S, V = PM.svd(da.array(array), mask_fill=method, mask_nan=True)
        assert PM.array.array[0, 1] == filled_value
        np.testing.assert_array_almost_equal(S, S_k)


def test_PowerMethod_reset():
    for start in [True, False]:

        PM = PowerMethod(sub_svd_start=start)

        _, _, _ = PM.svd(da.array(np.random.randn(100, 50)))
        _, _, _ = PM.svd(da.array(np.random.randn(110, 60)))


@pytest.mark.skip
def test_PowerMethod_transpose_array_shape():
    N, P, K = 100, 200, 10
    array = da.array(np.random.rand(N, P))

    PM = PowerMethod(k=K)
    U_PM, S_PM, V_PM = PM.svd(array)

    assert U_PM.shape == (N, K)
    assert S_PM.shape == (K,)
    assert V_PM.shape == (K, P)

    U_PM, S_PM, V_PM = PM.svd(array)

    assert U_PM.shape == (P, K)
    assert S_PM.shape == (K,)
    assert V_PM.shape == (K, N)


@pytest.mark.skip
def test_PowerMethod_transpose_array():
    array = da.array(np.random.rand(100, 200))
    k = 10
    U, S, V = da.linalg.svd(array)

    U_k, S_k, V_k = svd_to_trunc_svd(U, S, V, k=k)

    PM = PowerMethod(tol=1e-12, k=k, factor=None, scale=False, center=False, lmbd=0)

    U_PM, S_PM, V_PM = PM.svd(array)

    np.testing.assert_array_almost_equal(S_k, S_PM)
    np.testing.assert_almost_equal(subspace_dist(V_k, V_PM, S_PM), 0)
    assert U_k.shape == U_PM.shape

    U_PM, S_PM, V_PM = PM.svd(array, transpose=True)

    np.testing.assert_almost_equal(subspace_dist(U_PM.T, V_k, S_PM), 0)
    assert V_PM.shape != U_PM.shape


@pytest.mark.skip
def test_PowerMethod_persist():
    N = 10000
    x = da.random.random((N, N))

    PM = PowerMethod(max_iter=1)

    _, _, _ = PM.svd(x, persist=False)
    _, _, _ = PM.svd(x, persist=True)
    for dt in ['float64', 'float32', 'int8', 'uint8', 'int64', 'uint64']:
        _, _, _ = PM.svd(x, persist=True, dtype=dt)


def test_PowerMethod_std_method():
    N = 1000
    P = 100
    k = 10
    array = da.random.randint(0, 3, size=(N, P))
    for method in ['norm', 'binom']:
        new_array = make_snp_array(array, std_method=method)
        PM = PowerMethod(k=k, scoring_method='q-vals', tol=1e-13, factor=None)
        U_PM, S_PM, V_PM = PM.svd(array=new_array)

        mean = array.mean(axis=0)
        if method == 'norm':
            std = array.std(axis=0)
        else:
            p = mean/2
            std = da.sqrt(2*p*(1-p))

        x = (array - mean).dot(np.diag(1/std))

        U, S, V = da.linalg.svd(x)
        U_k, S_k, V_k = svd_to_trunc_svd(U, S, V, k=k)
        np.testing.assert_almost_equal(subspace_dist(U_PM, U_k, S_k), 0, decimal=3)
        np.testing.assert_almost_equal(subspace_dist(V_PM, V_k, S_k), 0, decimal=3)
        np.testing.assert_array_almost_equal(S_k, S_PM, decimal=2)


@pytest.mark.xfail
def test_PowerMethod_project():
    N, P = 1000, 1000
    k = 10
    svd_array = da.random.random(size=(N, P)).persist()
    proj_array = da.random.random(size=(10, P)).persist()
    mu = da.mean(svd_array, axis=0).persist()
    std = da.diag(1/da.std(svd_array, axis=0)).persist()

    for scale in [True, False]:
        for center in [True, False]:
            svd_array1 = svd_array
            proj_array1 = proj_array
            if center:
                svd_array1 = svd_array1 - mu
                proj_array1 = proj_array1 - mu
            if scale:
                svd_array1 = svd_array1.dot(std)
                proj_array1 = proj_array1.dot(std)

            U, S, V = da.linalg.svd(svd_array1)
            U_k, S_k, V_k = svd_to_trunc_svd(U, S, V, k=k)

            PM = PowerMethod(k=k, scale=scale, center=center, factor=None, tol=1e-12)
            U_PM, S_PM, V_PM = PM.svd(array=svd_array)

            np.testing.assert_array_almost_equal(PM.project(proj_array, onto=V_k.T), proj_array1.dot(V_k.T))


def test_PowerMethod_return_history():
    max_iter = 5
    array = da.random.random(size=(1000, 1000))
    PM = PowerMethod(scoring_method=['rmse', 'q-vals', 'v-subspace'], max_iter=max_iter, tol=[1e-16, 1e-16, 1e-16])
    history = PM.svd(array, return_history=True)

    assert len(history.acc['q-vals']) == max_iter
    assert len(history.acc['rmse']) == max_iter
    assert len(history.acc['v-subspace']) == max_iter

    assert len(history.iter['U']) == 0
    assert len(history.iter['S']) == max_iter
    assert len(history.iter['V']) == max_iter
    assert len(history.iter['last_value']) == 3

    assert history.time['start'] < history.time['stop']
    assert len(history.time['step']) == max_iter
    assert len(history.time['acc']) == max_iter

    np.testing.assert_array_almost_equal(np.array(history.time['iter']),
                                         np.array(history.time['acc']) + np.array(history.time['step']), decimal=3)


def test_SSPM_case1():
    N, P, k = 100000, 100, 10
    array = np.zeros(shape=(N, P))
    s_orig = np.linspace(1, 2, P)
    array[:P, :] = np.diag(np.linspace(1, 2, P))
    array[N - 1, :] = 1

    U, S, V = np.linalg.svd(array, full_matrices=False)
    U_k, S_k, V_k = svd_to_trunc_svd(U, S, V, k=k)

    SSPM = SuccessiveBatchedPowerMethod(k=k,
                                        sub_svd_start=True,
                                        tol=1e-12,
                                        factor=None)
    U_PM, S_PM, V_PM = SSPM.svd(array)

    np.testing.assert_almost_equal(subspace_dist(U_PM, U_k, S_k), 0)
    np.testing.assert_almost_equal(subspace_dist(V_PM, V_k, S_k), 0)
    np.testing.assert_array_almost_equal(S_k, S_PM)

    for sub_S in SSPM.history.iter['S'][:-1]:
        np.testing.assert_array_almost_equal(s_orig[::-1][:k], sub_S)


def test_SSPM_case2():
    N = 10000
    k = 10
    s_orig = np.ones(N)
    array = da.diag(s_orig)

    SSPM = SuccessiveBatchedPowerMethod(k=k,
                                        sub_svd_start=True,
                                        tol=1e-12,
                                        factor=None)

    _, _, _ = SSPM.svd(array)

    for sub_S in SSPM.history.iter['S']:
        np.testing.assert_array_almost_equal(s_orig[:k], sub_S)


def test_SSPM_case3():
    N = 500
    k = 10
    f = .1
    s_orig = np.array([1.01 ** i for i in range(1, N + 1)])
    array = da.diag(s_orig)

    SSPM = SuccessiveBatchedPowerMethod(k=k, sub_svd_start=True,
                                        tol=[1e-14, 1e-14],
                                        f=f,
                                        scoring_method=['q-vals', 'v-subspace'],
                                        factor=None)

    U_PM, S_PM, V_PM = SSPM.svd(array)

    np.testing.assert_array_almost_equal(s_orig[-k:][::-1], S_PM)

    for i, (sub_S, sub_V) in enumerate(zip(SSPM.history.iter['S'], SSPM.history.iter['V'])):
        s = sorted(s_orig[:int(f*(i+1)*N)], reverse=True)[:k]
        np.testing.assert_array_almost_equal(s, sub_S)

        v = np.zeros_like(sub_V)
        for j in range(k):
            v[j, int(f*(i+1)*N) - k + j] = 1

        np.testing.assert_almost_equal(subspace_dist(v, sub_V, sub_S), 0)


def test_SSPM_return_history():
    max_iter = 5
    f = .2
    array = da.random.random(size=(1000, 1000))
    SSPM = SuccessiveBatchedPowerMethod(scoring_method=['rmse', 'q-vals', 'v-subspace'], f=f, max_sub_iter=max_iter,
                                        tol=[1e-16, 1e-16, 1e-16])
    history = SSPM.svd(array, return_history=True)

    sub_iters = int(1/f)
    for i in range(sub_iters):
        assert len(history.time['iter']) == sub_iters - 1
        assert len(history.iter['S']) == sub_iters - 1
        assert len(history.iter['sub_svd']) == sub_iters
        assert len(history.acc['sub_svd_acc']) == sub_iters - 1
