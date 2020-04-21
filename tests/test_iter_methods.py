import pytest

import numpy as np
from lmdec.decomp import PowerMethod
from lmdec.array.metrics import subspace_dist


def test_PowerMethod_case1():
    n = 100
    p = 80
    array = np.random.rand(100, 80)
    mu = array.mean(axis=0)
    std = np.diag(1/array.std(axis=0))
    scaled_centered_array = (array-mu).dot(std)
    U, S, V = np.linalg.svd(scaled_centered_array, full_matrices=False)  # Ground Truth
    for k in range(1, 10):
        U_k, S_k, V_k = U[:, :k], S[:k], V[:k, :]

        PM = PowerMethod(k=k, tol=1e-9, scoring_method='rmse', max_iter=100, init_row_sampling_factor=1, factor=None)
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
    U, S, V = np.linalg.svd(scaled_centered_array.dot(scaled_centered_array.T), full_matrices=False)  # Ground Truth
    _, _, V = np.linalg.svd(scaled_centered_array.T.dot(scaled_centered_array), full_matrices=False)
    S = np.sqrt(S)
    k = 10
    U_k, S_k, V_k = U[:, :k], S[:k], V[:k, :]
    previous_S_error = float('inf')
    previous_U_error = float('inf')
    previous_V_error = float('inf')
    for t in np.logspace(0, -12, 20):

        PM = PowerMethod(k=k, tol=t, scoring_method='q-vals', max_iter=100, factor=None)
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


def test_PowerMethod_all_tols_agree():
    n = 100
    p = 80
    k = 10
    array = np.random.rand(n, p)

    PM = PowerMethod(k=k, tol=1e-9, scoring_method='q-vals', max_iter=100)
    U_q, S_q, V_q = PM.svd(array)

    PM = PowerMethod(k=k, tol=1e-4, scoring_method='rmse', max_iter=100)
    U_r, S_r, V_r = PM.svd(array)

    PM = PowerMethod(k=k, tol=1e-9, scoring_method='v-subspace', max_iter=100)
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

        PM = PowerMethod(k=k, tol=1e-9, scoring_method='q-vals', max_iter=100, factor=f, scale=False, center=False)

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

            PM = PowerMethod(k=k, tol=1e-12, scoring_method='q-vals', max_iter=100, scale=scale, center=center,
                             factor=None)
            U_q, S_q, V_q = PM.svd(array)

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
        PM = PowerMethod(k=k, tol=tols, scoring_method=['q-vals', 'rmse', 'v-subspace'], factor=None)

        _, _, _ = PM.svd(array)

        assert (min(PM.history['acc']['q-vals']) <= tols[0]
                or min(PM.history['acc']['rmse']) <= tols[1]
                or min(PM.history['acc']['v-subspace']) <= tols[2])

    fail_test_tols = [1e-6, 1e-16, 1e-16]

    PM = PowerMethod(k=k, tol=fail_test_tols, scoring_method=['q-vals', 'rmse', 'v-subspace'], factor=None)

    _, _, _ = PM.svd(array)

    assert (min(PM.history['acc']['q-vals']) <= fail_test_tols[0]
            and min(PM.history['acc']['rmse']) >= fail_test_tols[1]
            and min(PM.history['acc']['v-subspace']) >= fail_test_tols[2])


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
    assert len(PM.history['iter']['S']) == 1


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

    PM1 = PowerMethod(buffer=10, max_iter=2, tol=1e-12, scoring_method='rmse')
    PM2 = PowerMethod(buffer=20, max_iter=2, tol=1e-12, scoring_method='rmse')

    array = np.random.rand(1000, 1000)
    U, S, V = np.linalg.svd(array)
    S[0:15] = 1e3+S[0:15]

    array = U.dot(np.diag(S).dot(V))

    _, _, _ = PM1.svd(array)
    _, _, _ = PM2.svd(array)

    PM1.history['acc']['rmse'][-1] > PM2.history['acc']['rmse'][-1]


def test_PowerMethod_sub_svd_start():
    array = np.random.rand(100, 100)

    PM1 = PowerMethod(sub_svd_start=True, tol=1e-12)
    _, S1, _ = PM1.svd(array)

    PM2 = PowerMethod(sub_svd_start=False, tol=1e-12)
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

