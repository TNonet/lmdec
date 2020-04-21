import numpy as np
import pytest
import dask.array as da
from lmdec.array.scaled import ScaledArray, combine_means, combine_stds, ArrayMoment
from lmdec.array.random import array_constant_partition

num_run = 10


def test_ScaledArray_array():
    N = 50
    P = 40
    array = np.random.rand(N, P) + 1
    std = np.diag(1 / np.std(array, axis=0))
    mu = np.mean(array, axis=0)
    for s in [True, False]:
        for c in [True, False]:
            for f in [None, 'n', 'p']:
                sarray = ScaledArray(scale=s, center=c, factor=f)
                sarray.fit(da.array(array))

                if c:
                    array = array - mu
                if s:
                    array = array.dot(std)

                result = array.dot(array.T)

                if f == 'n':
                    result /= N
                elif f == 'p':
                    result /= P

                np.testing.assert_array_almost_equal(sarray.array, result)


def test_ScaledArray_sym_mat_mult():
    for N in range(2, 5):
        for P in range(2, 5):
            array = np.random.rand(N, P) + 1
            std = np.diag(1/np.std(array, axis=0))
            mu = np.mean(array, axis=0)
            for factor in [None, 'n', 'p']:
                if factor is None:
                    f = 1
                elif factor == 'n':
                    f = N
                else:
                    f = P
                for K in range(1, 5):
                    for squeeze in [True, False]:
                        x = np.random.rand(N, K)
                        if squeeze:
                            x = np.squeeze(x)

                            for fit_x in [x, None]:

                                # With No Scale or Center
                                # x = A'Ax
                                result = array.dot(array.T.dot(x))/f
                                assert result.shape == x.shape
                                sarray = ScaledArray(scale=False, center=False, factor=factor)
                                sarray.fit(da.array(array), x=fit_x)
                                np.testing.assert_array_equal(result, sarray.sym_mat_mult(x))

                                # With Scale but No Center
                                # B = AD
                                b_array = array.dot(std)
                                result = b_array.dot(b_array.T.dot(x))/f
                                assert result.shape == x.shape
                                sarray = ScaledArray(scale=True, center=False, factor=factor)
                                sarray.fit(da.array(array), x=fit_x)
                                np.testing.assert_array_almost_equal(result, sarray.sym_mat_mult(x))

                                # With Center but No Scale:
                                # B = (A - U)
                                b_array = array - mu
                                result = b_array.dot(b_array.T.dot(x))/f
                                sarray = ScaledArray(scale=False, center=True, factor=factor)
                                sarray.fit(da.array(array), x=fit_x)
                                np.testing.assert_array_almost_equal(result, sarray.sym_mat_mult(x))

                                # With Center and  Scale:
                                # (A - U)'D'D(A - U)x
                                result = (array - mu).dot(std).dot(std).dot((array - mu).T.dot(x))/f
                                sarray = ScaledArray(scale=True, center=True, factor=factor)
                                sarray.fit(da.array(array), x=fit_x)
                                np.testing.assert_array_almost_equal(result, sarray.sym_mat_mult(x))


def test_ScaledArray_different_x():
    for N in range(2, 5):
        for P in range(2, 5):
            array = np.random.rand(N, P) + 1
            std = np.diag(1/np.std(array, axis=0))
            mu = np.mean(array, axis=0)
            for K in range(3, 5):
                x = np.random.rand(N, K)
                for fit_x in [x, None]:
                    for dx in [-1, 0, 1]:
                        x2_K = x.shape[1] + dx
                        x2 = x[:, 0:x2_K]

                        # With No Scale or Center
                        # x = A'Ax
                        result = array.dot(array.T.dot(x2))
                        sarray = ScaledArray(scale=False, center=False)
                        sarray.fit(da.array(array), x=fit_x)
                        np.testing.assert_array_equal(result, sarray.sym_mat_mult(x2))

                        # With Scale but No Center
                        # B = AD
                        b_array = array.dot(std)
                        result = b_array.dot(b_array.T.dot(x2))
                        sarray = ScaledArray(scale=True, center=False)
                        sarray.fit(da.array(array), x=fit_x)
                        np.testing.assert_array_almost_equal(result, sarray.sym_mat_mult(x2))

                        # With Center but No Scale:
                        # B = (A - U)
                        b_array = array - mu
                        result = b_array.dot(b_array.T.dot(x2))
                        sarray = ScaledArray(scale=False, center=True)
                        sarray.fit(da.array(array), x=fit_x)
                        np.testing.assert_array_almost_equal(result, sarray.sym_mat_mult(x2))

                        # With Center and  Scale:
                        # (A - U)'D'D(A - U)x
                        result = (array - mu).dot(std).dot(std).dot((array - mu).T.dot(x2))
                        sarray = ScaledArray(scale=True, center=True)
                        sarray.fit(da.array(array), x=fit_x)
                        np.testing.assert_array_almost_equal(result, sarray.sym_mat_mult(x2))


def test_ScaledArray_dot():
    for N in range(2, 5):
        for P in range(2, 5):
            array = np.random.rand(N, P) + 1
            std = np.diag(1/np.std(array, axis=0))
            mu = np.mean(array, axis=0)
            for K in range(1, 5):
                for squeeze in [True, False]:
                    x = np.random.rand(P, K)
                    if squeeze:
                        x = np.squeeze(x)
                    for fit_x in [x, None]:
                        # With No Scale or Center
                        # x = A'Ax
                        result = array.dot(x)
                        sarray = ScaledArray(scale=False, center=False)
                        sarray.fit(da.array(array), x=fit_x)
                        np.testing.assert_array_almost_equal(result, sarray.dot(x))

                        # With Scale but No Center
                        # B = AD
                        b_array = array.dot(std)
                        result = b_array.dot(x)
                        sarray = ScaledArray(scale=True, center=False)
                        sarray.fit(da.array(array), x=fit_x)
                        np.testing.assert_array_almost_equal(result, sarray.dot(x))

                        # With Center but No Scale:
                        # B = (A - U)
                        b_array = array - mu
                        result = b_array.dot(x)
                        sarray = ScaledArray(scale=False, center=True)
                        sarray.fit(da.array(array), x=fit_x)
                        np.testing.assert_array_almost_equal(result, sarray.dot(x))

                        # With Center and  Scale:
                        # (A - U)'D'D(A - U)x
                        b_array = (array - mu).dot(std)
                        result = b_array.dot(x)
                        sarray = ScaledArray(scale=True, center=True)
                        sarray.fit(da.array(array), x=fit_x)
                        np.testing.assert_array_almost_equal(result, sarray.dot(x))


def test_ScaledArray_T_dot():
    for N in range(2, 5):
        for P in range(2, 5):
            array = np.random.rand(N, P) + 1
            std = np.diag(1/np.std(array, axis=0))
            mu = np.mean(array, axis=0)
            for K in range(1, 5):
                for squeeze in [True, False]:
                    x = np.random.rand(N, K)
                    if squeeze:
                        x = np.squeeze(x)
                    for fit_x in [x, None]:
                        # With No Scale or Center
                        # x = A'Ax
                        result = array.T.dot(x)
                        sarray = ScaledArray(scale=False, center=False)
                        sarray.fit(da.array(array), x=fit_x)
                        np.testing.assert_array_almost_equal(result, sarray.T.dot(x))

                        # With Scale but No Center
                        # B = AD
                        b_array = array.dot(std).T
                        result = b_array.dot(x)
                        sarray = ScaledArray(scale=True, center=False)
                        sarray.fit(da.array(array), x=fit_x)
                        np.testing.assert_array_almost_equal(result, sarray.T.dot(x))

                        # With Center but No Scale:
                        # B = (A - U)
                        b_array = (array - mu).T
                        result = b_array.dot(x)
                        sarray = ScaledArray(scale=False, center=True)
                        sarray.fit(da.array(array), x=fit_x)
                        np.testing.assert_array_almost_equal(result, sarray.T.dot(x))

                        # With Center and  Scale:
                        # (A - U)'D'D(A - U)x
                        b_array = (array - mu).dot(std).T
                        result = b_array.dot(x)
                        sarray = ScaledArray(scale=True, center=True)
                        sarray.fit(da.array(array), x=fit_x)
                        np.testing.assert_array_almost_equal(result, sarray.T.dot(x))


def test_array_tranpose_tranpose():
    array = da.array(np.random.rand(7, 10))
    x = da.array(np.random.rand(10, 5))
    sarray = ScaledArray(scale=True, center=True)
    sarray.fit(da.array(array))

    s_array_T_T = sarray.T.T
    assert id(sarray._array) == id(s_array_T_T._array)

    assert id(sarray) == id(s_array_T_T)

    np.testing.assert_array_equal(sarray.dot(x), s_array_T_T.dot(x))


def test_array_id():
    array = da.array(np.random.rand(10, 7))
    x = da.array(np.random.rand(10,5))
    sarray = ScaledArray(scale=True, center=True)
    sarray.fit(da.array(array), x=x)
    sarray_T = sarray.T
    assert id(sarray._array) == id(sarray_T._array)
    assert id(sarray.center_vector) == id(sarray_T.center_vector)
    assert id(sarray._array_moment.scale_matrix) == id(sarray_T._array_moment.scale_matrix)
    assert id(sarray._array_moment.sym_scale_matrix) == id(sarray_T._array_moment.sym_scale_matrix)


def test_array_shapes():
    N, K = 10, 7
    array = da.array(np.random.rand(N, K))
    sarray = ScaledArray()
    sarray.fit(array)

    assert sarray.shape == (N, K)
    assert sarray.T.shape == (K, N)


def test_bad_array():
    array = da.array(np.random.rand(3, 3, 3))
    sarray = ScaledArray()
    with pytest.raises(ValueError):
        sarray.fit(array)


def test_bad_x():
    array = da.array(np.random.rand(3, 3))
    x = da.array(np.random.rand(3,3,3))
    sarray = ScaledArray()
    with pytest.raises(ValueError):
        sarray.fit(array, x)


def test__getitem__base_array_case1():
    array = da.array(np.random.rand(3, 3))
    sarray = ScaledArray()
    sarray.fit(array)
    np.testing.assert_array_equal(sarray[:, :]._array, sarray._array)


def test_ScaledArray_array():
    for N in range(3, 5):
        for P in range(3, 5):
            array = np.random.rand(N, P) + 1
            std = np.diag(1 / np.std(array, axis=0))
            mu = np.mean(array, axis=0)

            sarray = ScaledArray(scale=False, center=False)
            sarray.fit(da.array(array))
            np.testing.assert_array_almost_equal(array, sarray.array)
            np.testing.assert_array_almost_equal(array.T, sarray.T.array)

            # With Scale but No Center
            # B = AD
            b_array = array.dot(std)
            sarray = ScaledArray(scale=True, center=False)
            sarray.fit(da.array(array))
            np.testing.assert_array_almost_equal(b_array, sarray.array)
            np.testing.assert_array_almost_equal(b_array.T, sarray.T.array)

            # With Center but No Scale:
            # B = (A - U)
            b_array = array - mu
            sarray = ScaledArray(scale=False, center=True)
            sarray.fit(da.array(array))
            np.testing.assert_array_almost_equal(b_array, sarray.array)
            np.testing.assert_array_almost_equal(b_array.T, sarray.T.array)

            # With Center and  Scale:
            # (A - U)'D'D(A - U)x
            b_array = (array - mu).dot(std)
            sarray = ScaledArray(scale=True, center=True)
            sarray.fit(da.array(array))
            np.testing.assert_array_almost_equal(b_array, sarray.array)
            np.testing.assert_array_almost_equal(b_array.T, sarray.T.array)


def test__getitem__mult_case():
    for N in range(3, 5):
        for P in range(3, 5):
            array = np.random.rand(N, P) + 1
            for K in range(1, 5):
                for sub_N in range(2, N):
                    sub_array = array[0:sub_N, :]
                    std = np.diag(1 / np.std(sub_array, axis=0))
                    mu = np.mean(sub_array, axis=0)
                    for squeeze in [True, False]:
                        x = np.random.rand(P, K)
                        if squeeze:
                            x = np.squeeze(x)
                        for fit_x in [x, None]:
                            # With No Scale or Center
                            # x = A'Ax
                            result = sub_array.dot(x)
                            sarray = ScaledArray(scale=False, center=False)
                            sarray.fit(da.array(array), x=fit_x)
                            np.testing.assert_array_almost_equal(result, sarray[0:sub_N, :].dot(x))

                            # With Scale but No Center
                            # B = AD
                            b_array = sub_array.dot(std)
                            result = b_array.dot(x)
                            sarray = ScaledArray(scale=True, center=False)
                            sarray.fit(da.array(array), x=fit_x)
                            np.testing.assert_array_almost_equal(result, sarray[0:sub_N, :].dot(x))

                            # With Center but No Scale:
                            # B = (A - U)
                            b_array = sub_array - mu
                            result = b_array.dot(x)
                            sarray = ScaledArray(scale=False, center=True)
                            sarray.fit(da.array(array), x=fit_x)
                            np.testing.assert_array_almost_equal(result, sarray[0:sub_N, :].dot(x))

                            # With Center and  Scale:
                            # (A - U)'D'D(A - U)x
                            b_array = (sub_array - mu).dot(std)
                            result = b_array.dot(x)
                            sarray = ScaledArray(scale=True, center=True)
                            sarray.fit(da.array(array), x=fit_x)
                            np.testing.assert_array_almost_equal(result, sarray[0:sub_N, :].dot(x))


def test__getitem__base_array_case2():
    for N in range(2, 5):
        for P in range(2, 5):
            array = da.array(np.random.rand(N , P))
            sarray = ScaledArray()
            sarray.fit(array)
            for i in range(2, N):
                for j in range(2, P):
                    np.testing.assert_array_equal(sarray[0:i, 0:j]._array, array[0:i, 0:j])


def test__getitem_scaled_array_shape():
    for N in range(2, 5):
        for P in range(2, 5):
            array = da.array(np.random.rand(N, P))
            sarray = ScaledArray()
            sarray.fit(array)
            for i in range(2, N):
                for j in range(2, P):
                    assert sarray[0:i, 0:j].shape == array[0:i, 0:j].shape

            for size in range(2, 5):
                i = np.sort(np.random.choice(np.arange(N), size=size))

                assert sarray[i, :].shape == array[i, :].shape


def test__getitem_T_subset_array():
    a = da.random.random(size=(100, 200))
    sa = ScaledArray()
    sa.fit(a)

    np.testing.assert_array_equal(sa.T[0:10, :].shape, a.T[0:10, :].shape)
    np.testing.assert_array_equal(sa.T[:, 0:8].shape, a.T[:, 0:8].shape)
    np.testing.assert_array_equal(sa.T[0:10, 0:8].shape, a.T[0:10, 0:8].shape)

    np.testing.assert_array_equal(sa[0:10, :].shape, a[0:10, :].shape)
    np.testing.assert_array_equal(sa.T[:, 0:8].shape, a.T[:, 0:8].shape)
    np.testing.assert_array_equal(sa.T[0:10, 0:8].shape, a.T[0:10, 0:8].shape)

    np.testing.assert_array_equal(sa[0:50, 0:40].T[0:20, 0:30].shape, a[0:50, 0:40].T[0:20, 0:30].shape)


def test__getitem_T_subset_chunks():
    a = da.random.random(size=(100, 200))
    sa = ScaledArray()
    sa.fit(a)

    np.testing.assert_array_equal(sa.T[0:10, :].chunks, a.T[0:10, :].chunks)
    np.testing.assert_array_equal(sa.T[:, 0:8].chunks, a.T[:, 0:8].chunks)
    np.testing.assert_array_equal(sa.T[0:10, 0:8].chunks, a.T[0:10, 0:8].chunks)

    np.testing.assert_array_equal(sa[0:10, :].chunks, a[0:10, :].chunks)
    np.testing.assert_array_equal(sa.T[:, 0:8].chunks, a.T[:, 0:8].chunks)
    np.testing.assert_array_equal(sa.T[0:10, 0:8].chunks, a.T[0:10, 0:8].chunks)

    np.testing.assert_array_equal(sa[0:50, 0:40].T[0:20, 0:30].chunks, a[0:50, 0:40].T[0:20, 0:30].chunks)


def gen_sets():
    num_sets = np.random.randint(2, 10)
    sets = []
    for i in range(num_sets):
        set_size = np.random.randint(1, 10)
        sub_set = np.random.rand(set_size)
        sets.append(sub_set)

    combined_set = [item for sublist in sets for item in sublist]

    return sets, combined_set


def test_combine_stds():
    for _ in range(num_run):
        sets, combined_set = gen_sets()

        std = np.std(combined_set)

        set_len = [len(sub_set) for sub_set in sets]
        sets_mean = [np.mean(sub_set) for sub_set in sets]
        sets_std = [np.std(sub_set) for sub_set in sets]

        itr = list(zip(set_len, sets_mean, sets_std))
        combined_std = combine_stds(itr)
        np.testing.assert_almost_equal(combined_std, std)


def test_combnine_means():
    for _ in range(num_run):
        sets, combined_set = gen_sets()

        sets_info = []
        for set in sets:
            sets_info.append((len(set), np.mean(set)))

        mean = np.mean(combined_set)

        combined_mean = combine_means(sets_info)

        np.testing.assert_almost_equal(mean, combined_mean)


def test_array_moment_case1():
    for N in range(2, 5):
        for P in range(2, 5):
            array = np.random.rand(N, P)
            for array_type in [da.array, np.array]:
                temp_array = array_type(array)
                for K in range(1, 10):
                    vector = np.random.rand(N, K)
                    for vector_type in [da.array, np.array]:
                        temp_vector = vector_type(vector)
                        for fit_x in [temp_vector, None]:
                            a = ArrayMoment(temp_array, batch_calc=None)
                            a.fit()
                            if fit_x is not None:
                                a.fit_x(fit_x)

                                assert a.vector_width == K
                                assert a.sym_scale_matrix.shape == (P, K)
                                assert a.scale_matrix.shape == (P, K)
                            else:
                                with pytest.raises(ValueError):
                                    a.vector_width
                                with pytest.raises(ValueError):
                                    a.sym_scale_matrix
                                with pytest.raises(ValueError):
                                    a.scale_matrix

                            np.testing.assert_array_almost_equal(a.center_vector, temp_array.mean(axis=0))
                            np.testing.assert_array_almost_equal(a.scale_vector, 1/temp_array.std(axis=0))
                            np.testing.assert_array_almost_equal(a.sym_scale_vector, 1/temp_array.std(axis=0)**2)


def test_sub_array_moments_no_batch_calc():
    for N in range(3, 5):
        for P in range(3, 5):
            array = np.random.rand(N, P)

            a = ArrayMoment(array, None)

            for sub_N in range(2, N):
                for sub_P in range(2, P):
                    sub_array = array[0:sub_N, 0:sub_P]
                    np.testing.assert_almost_equal(a[0:sub_N, 0:sub_P].sym_scale_vector, 1/sub_array.std(axis=0)**2)
                    np.testing.assert_almost_equal(a[0:sub_N, 0:sub_P].scale_vector, 1/sub_array.std(axis=0))
                    np.testing.assert_almost_equal(a[0:sub_N, 0:sub_P].center_vector, sub_array.mean(axis=0))


def test_sub_array_moments_use_batch_calc():

    array = np.random.rand(10, 10)

    batches = array_constant_partition(array.shape, f= .2, min_size = 2)

    a = ArrayMoment(array, batches)

    for i in range(5):
        for j in range(i):
            sub_batches = batches[i:j]
            np.testing.assert_array_almost_equal(a[sub_batches, :].center_vector,
                                                 array[sub_batches, :].mean(axis=0))
            np.testing.assert_array_almost_equal(a[sub_batches, :].scale_vector,
                                                 1/array[sub_batches, :].std(axis=0))
            np.testing.assert_array_almost_equal(a[sub_batches, :].sym_scale_vector,
                                                 1/array[sub_batches, :].std(axis=0)**2)

