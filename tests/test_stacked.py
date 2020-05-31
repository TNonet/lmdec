import pytest
import numpy as np
from dask import array as da
import time

from lmdec.array.stacked import StackedArray

from functools import reduce


def test_non_stacked():
    with pytest.raises(ValueError):
        StackedArray(da.random.random(size=(10, 10)))


def test_bad_stacked_types():
    with pytest.raises(ValueError):
        StackedArray(['4', da.random.random(size=(10, 10))])


def test_bad_stacked_shapes():
    with pytest.raises(ValueError):
        StackedArray([da.random.random(size=(2, )), da.random.random(size=(3, ))])


def test_underlying_arrays():
    arrays = [da.random.random(size=(4,)) for _ in range(3)]

    sa = StackedArray(arrays)

    id_list = [id(x) for x in sa.arrays]

    for array in arrays:
        assert id(array) in id_list


def test_reduce_arrays_size2():
    x1 = da.random.random(size=(4,))
    x2 = da.random.random(size=(4,))

    x = x1+x2

    for tree_reduce in [True, False]:
        sa = StackedArray([x1, x2], tree_reduce=tree_reduce)

        np.testing.assert_array_equal(sa, x)


def test_reduce_arrays_size3():
    x1 = da.random.random(size=(4,))
    x2 = da.random.random(size=(4,))
    x3 = da.random.random(size=(4,))
    x = x1+x2+x3

    for tree_reduce in [True, False]:
        sa = StackedArray([x1, x2, x3], tree_reduce=tree_reduce)

        np.testing.assert_array_almost_equal(sa, x, decimal=14)


def test_reduce_arrays_sizeN():
    for N in range(3, 15):
        arrays = [da.random.random(size=(4,)) for _ in range(N)]

        array = reduce(da.add, arrays)

        for tree_reduce in [True, False]:
            sa = StackedArray(arrays, tree_reduce=tree_reduce)

            np.testing.assert_array_almost_equal(sa, array, decimal=12)


def test_T():
    N, P = 10, 7
    sa = StackedArray([da.random.random(size=(N, P)) for _ in range(7)])

    np.testing.assert_array_almost_equal(sa.T, sa.array.T)
    np.testing.assert_array_almost_equal(sa, sa.T.T)


def test_dot():
    N, P = 10, 7
    sa = StackedArray([da.random.random(size=(N, P)) for _ in range(7)])
    y = da.random.random(P)

    np.testing.assert_array_almost_equal(sa.dot(y), sa.array.dot(y), decimal=12)


def test_persist():
    def delay_array(x):
        # slow operation takes about 2 seconds on my computer
        d = da.mean(da.random.random(1e9))
        return d + x - d

    arrays = [delay_array(da.random.random(size=(4,))) for _ in range(2)]

    sa = StackedArray(arrays)
    sa_persist = sa.persist()

    start = time.time()
    sa_persist.mean().compute()
    persist_took = time.time() - start

    start_mean = time.time()
    sa.mean().compute()
    mean_took = time.time() - start_mean
    assert persist_took <= mean_took / 1000


def test_mean():
    sa = StackedArray([da.random.random(size=(10, 10, 10)) for _ in range(7)])

    for axis in [None, 0, 1, 2, (0, 1), -1, (1, 2), (0, 1, 2)]:
        np.testing.assert_array_almost_equal(sa.mean(axis=axis), sa.array.mean(axis=axis))


def test___getitem__():
    sa = StackedArray([da.random.random(size=(10, 10)) for _ in range(7)])

    np.testing.assert_array_almost_equal(sa[0, 0], sa.array[0, 0],  decimal=12)
    np.testing.assert_array_almost_equal(sa[0, :], sa.array[0, :], decimal=12)
    np.testing.assert_array_almost_equal(sa[:, 0], sa.array[:, 0], decimal=12)
    np.testing.assert_array_almost_equal(sa[:, :], sa.array[:, :], decimal=12)


def test_rechunk():
    sa = StackedArray([da.random.random(size=(4, 4)) for _ in range(7)])

    for chunk_shape in [(1, 1), (2, 2), (4, 4)]:
        sa1 = sa.rechunk(chunk_shape)

        assert sa1.chunks == sa1.array.chunks


def test_reshape():
    sa = StackedArray([da.random.random(size=(4, 4)) for _ in range(7)])

    for new_shape in [(16,), (2, 8), (4, 4), (8, 2)]:

        sa1 = sa.reshape(new_shape)

        assert sa1.shape == new_shape
        np.testing.assert_array_almost_equal(sa1.mean(), sa.mean(), decimal=12)
        np.testing.assert_array_almost_equal(sa1.std(), sa.std(), decimal=12)


def test_fallback_methods():
    sa = StackedArray([da.random.random(size=(4, 4)) for x in range(7)])

    assert sa.shape == sa.array.shape
    assert sa.chunks == sa.array.chunks

    for axis in [None, 0, 1]:
        np.testing.assert_array_almost_equal(sa.std(axis=axis), sa.array.std(axis=axis), decimal=12)
        np.testing.assert_array_almost_equal(sa.mean(axis=axis), sa.array.mean(axis=axis), decimal=12)
        np.testing.assert_array_almost_equal(sa.max(axis=axis), sa.array.max(axis=axis), decimal=12)
        np.testing.assert_array_almost_equal(sa.min(axis=axis), sa.array.min(axis=axis),  decimal=12)


def test_StackedArray_of_StackedArrays():
    sa_arrays = [StackedArray([da.random.random(size=(4, 4)) for x in range(2)]) for _ in range(2)]
    StackedArray(sa_arrays)
