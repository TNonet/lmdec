import pytest
import numpy as np
from dask import array as da
import time

from hypothesis import given, note, assume
from hypothesis.extra import numpy as npst

from lmdec.array.stacked import StackedArray

from functools import reduce

import utils_tests


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


@given(shape=npst.array_shapes(min_dims=2, max_dims=2))
def test_T_constant_shape_2D(shape):
    N, P = shape
    sa = StackedArray([da.random.random(size=(N, P)) for _ in range(7)])

    np.testing.assert_array_almost_equal(sa.T, sa.array.T)
    np.testing.assert_array_almost_equal(sa, sa.T.T)


@given(shape=npst.array_shapes(min_dims=2, max_dims=2))
def test_T_2D_1D(shape):
    N, P = shape
    assume(N > 1)
    assume(P > 1)
    sa = StackedArray([da.random.random(size=(N, P)), da.random.random(size=(N, 1)),
                       da.random.random(size=(P,)), da.random.random(size=(1, P))])

    np.testing.assert_array_almost_equal(sa.T, sa.array.T)
    np.testing.assert_array_almost_equal(sa, sa.T.T)


@given(shape=npst.array_shapes(min_dims=2, max_dims=2))
def test_T_dot_2D_1D(shape):
    N, P = shape
    assume(N > 1)
    assume(P > 1)
    sa = StackedArray([da.random.random(size=(N, P)), da.random.random(size=(N, 1)),
                       da.random.random(size=(P,)), da.random.random(size=(1, P))])

    n, p = sa.shape
    for size in [(n, 2), (n,)]:
        y = da.random.random(size=size)
        np.testing.assert_array_equal(sa.T.array.dot(y), sa.array.T.dot(y))
        np.testing.assert_array_almost_equal(sa.T.dot(y), sa.array.T.dot(y))


@given(shape=npst.array_shapes(min_dims=2, max_dims=2))
def test_dot_constant_shape_2D(shape):
    N, P = shape
    sa = StackedArray([da.random.random(size=(N, P)) for _ in range(7)])
    y = da.random.random(P)

    np.testing.assert_array_almost_equal(sa.dot(y), sa.array.dot(y), decimal=12)

    y = da.random.random((P, 2))

    np.testing.assert_array_almost_equal(sa.dot(y), sa.array.dot(y), decimal=12)


@given(shape=npst.array_shapes(min_dims=2, max_dims=2))
def test_dot_2D_1D(shape):
    N, P = shape
    assume(N > 1)
    assume(P > 1)
    sa = StackedArray([da.random.random(size=(N, P)), da.random.random(size=(N, 1)),
                       da.random.random(size=(P,)), da.random.random(size=(1, P))])

    for size in [(P, 2), (P,)]:
        y = da.random.random(size=size)
    np.testing.assert_array_almost_equal(sa.dot(y), sa.array.dot(y), decimal=12)



@given(shapes=utils_tests.get_boardcastable_arrays_shapes(base_min_dims=2, base_max_dims=2,
                                                          broadcast_min_dims=1, broadcast_max_dims=2))
def test_dot_non_consistent_shape(shapes):
    assume(all(max(shape) > 1 for shape in shapes))
    sa = StackedArray([da.random.random(shape) for shape in shapes])
    N, P = sa.shape
    y = da.random.random(P)
    note(f"shapes: {shapes}, y shape: {y.shape}")
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
    assert persist_took <= mean_took / 10


def test_mean_consistent_shape():
    sa = StackedArray([da.random.random(size=(10, 10, 10)) for _ in range(7)])

    for axis in [None, 0, 1, 2, (0, 1), -1, (1, 2), (0, 1, 2)]:
        np.testing.assert_array_almost_equal(sa.mean(axis=axis), sa.array.mean(axis=axis))


@given(shapes=utils_tests.get_boardcastable_arrays_shapes())
def test_mean_non_consistent_shape(shapes):
    sa = StackedArray([da.random.random(shape) for shape in shapes])
    for axis in [None, -1, *list(list(range(x)) for x in range(len(sa.shape)))]:
        note(f"shapes: {shapes}, axis: {axis}")
        np.testing.assert_array_almost_equal(sa.mean(axis=axis), sa.array.mean(axis=axis))


def test___getitem__():
    sa = StackedArray([da.random.random(size=(10, 10)) for _ in range(7)])

    np.testing.assert_array_almost_equal(sa[0, 0], sa.array[0, 0],  decimal=12)
    np.testing.assert_array_almost_equal(sa[0, :], sa.array[0, :], decimal=12)
    np.testing.assert_array_almost_equal(sa[:, 0], sa.array[:, 0], decimal=12)
    np.testing.assert_array_almost_equal(sa[:, :], sa.array[:, :], decimal=12)


@given(shapes_indicies=utils_tests.get_boardcastable_arrays_shapes_and_indices())
def test__getitem__non_consistent_shape(shapes_indicies):
    shapes, indices, shape = shapes_indicies
    sa = StackedArray([da.random.random(shape) for shape in shapes])
    note(f"shapes: {shapes}, indices: {indices}")
    np.testing.assert_array_equal(sa.array.shape, shape)
    if np.product(sa.array[indices].shape) > 0:
        np.testing.assert_array_almost_equal(sa[indices], sa.array[indices], decimal=12)


def test_rechunk():
    sa = StackedArray([da.random.random(size=(4, 4)) for _ in range(7)])

    for chunk_shape in [(1, 1), (2, 2), (4, 4)]:
        sa1 = sa.rechunk(chunk_shape)

        assert sa1.chunks == sa1.array.chunks


@pytest.mark.xfail
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
    sa_arrays = [StackedArray([da.random.random(size=(4, 4)) for _ in range(2)]) for _ in range(2)]
    StackedArray(sa_arrays)
