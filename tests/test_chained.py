import pytest
import numpy as np
from dask import array as da

from hypothesis import given, note, assume, settings

from lmdec.array.chained import ChainedArray

from functools import reduce

import utils_tests


def test_non_stacked():
    with pytest.raises(ValueError):
        ChainedArray(da.random.random(size=(10, 10)))


def test_bad_stacked_types():
    with pytest.raises(ValueError):
        ChainedArray(['4', da.random.random(size=(10, 10))])


def test_bad_stacked_shapes():
    with pytest.raises(ValueError):
        ChainedArray([da.random.random(size=(2, )), da.random.random(size=(3, ))])


def test_underlying_arrays():
    arrays = [da.random.random(size=(4,)) for _ in range(3)]

    sa = ChainedArray(arrays)

    id_list = [id(x) for x in sa.arrays]

    for array in arrays:
        assert id(array) in id_list


def test_reduce_arrays_size2():
    x1 = da.random.random(size=(2, 3))
    x2 = da.random.random(size=(3, 4))

    x = x1@x2

    for tree_reduce in [True, False]:
        sa = ChainedArray([x1, x2], tree_reduce=tree_reduce)

        np.testing.assert_array_equal(sa, x)


def test_reduce_arrays_size3():
    x1 = da.random.random(size=(2, 3))
    x2 = da.random.random(size=(3, 4))
    x3 = da.random.random(size=(4, 5))
    x = x1@x2@x3

    for tree_reduce in [True, False]:
        sa = ChainedArray([x1, x2, x3], tree_reduce=tree_reduce)

        np.testing.assert_array_almost_equal(sa, x, decimal=14)


def test_reduce_arrays_sizeN():
    for N in range(3, 15):
        arrays = [da.random.random(size=(i, i+1)) for i in range(2, N)]

        array = reduce(da.dot, arrays)

        for tree_reduce in [True, False]:
            sa = ChainedArray(arrays, tree_reduce=tree_reduce)

            np.testing.assert_array_almost_equal(sa, array, decimal=8)


@given(shapes=utils_tests.get_chainable_array_shapes())
def test_T(shapes):
    assume(1 not in shapes[0])
    sa = ChainedArray([da.random.random(size=shape) for shape in shapes])

    np.testing.assert_array_almost_equal(sa.T, sa.array.T)
    np.testing.assert_array_almost_equal(sa, sa.T.T)


@given(shapes=utils_tests.get_chainable_array_shapes(base_min_dims=2, chain_min_dims=2))
def test_dot_constant_shape_2D(shapes):
    assume(1 not in shapes[0])
    sa = ChainedArray([da.random.random(size=shape) for shape in shapes])
    n, p = sa.shape
    y = da.random.random(p)

    np.testing.assert_array_almost_equal(sa.dot(y), sa.array.dot(y), decimal=10)


@given(shapes=utils_tests.get_chainable_array_shapes(base_min_dims=1, chain_min_dims=2))
def test_dot_constant_shape_1D_2D(shapes):
    assume(1 not in shapes[0])
    sa = ChainedArray([da.random.random(size=shape) for shape in shapes])
    n, p = sa.shape
    y = da.random.random(p)

    np.testing.assert_array_almost_equal(sa.dot(y), sa.array.dot(y), decimal=10)


@given(shapes=utils_tests.get_chainable_array_shapes(base_min_dims=1, chain_min_dims=2))
def test_dot_constant_shape_1D_2D(shapes):
    assume(1 not in shapes[0])
    sa = ChainedArray([da.random.random(size=shape) for shape in shapes])
    n, p = sa.shape
    y = da.random.random(p)

    np.testing.assert_array_almost_equal(sa.dot(y), sa.array.dot(y), decimal=10)


@given(shapes=utils_tests.get_chainable_array_shapes(base_min_dims=1, chain_min_dims=2))
def test_dot_transpose_dot(shapes):
    assume(1 not in shapes[0])
    sa = ChainedArray([da.random.random(size=shape) for shape in shapes])
    n, p = sa.shape

    for size in [(n, 2), (n, )]:
        y = da.random.random(size)
        np.testing.assert_array_almost_equal(sa.T.dot(y), sa.array.T.dot(y), decimal=10)


@given(shapes=utils_tests.get_chainable_array_shapes(base_min_dims=2, chain_min_dims=2))
@settings(deadline=None)
def test___getitem__shape_2D(shapes):
    sa = ChainedArray([da.random.random(size=shape) for shape in shapes])
    np.testing.assert_array_almost_equal(sa[0:1, 0:1], sa.array[0:1, 0:1])
    np.testing.assert_array_almost_equal(sa[0:1, :], sa.array[0:1, :])
    np.testing.assert_array_almost_equal(sa[:, 0:1], sa.array[:, 0:1])
    np.testing.assert_array_almost_equal(sa[:, :], sa.array[:, :])


@given(shapes_indicies=utils_tests.get_chainable_arrays_shapes_and_indices(max_chain=3))
@settings(deadline=None, max_examples=10)
def test__getitem__non_consistent_shape(shapes_indicies):
    shapes, indices, shape = shapes_indicies
    sa = ChainedArray([da.random.random(shape) for shape in shapes])
    note(f"shapes: {shapes}, indices: {indices}")

    np.testing.assert_array_equal(sa.array.shape, shape)
    np.testing.assert_array_almost_equal(sa.array[indices], sa[indices])
