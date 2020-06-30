from hypothesis import given

import numpy as np

from functools import reduce

import utils_hypothesis


@given(shapes=utils_hypothesis.get_boardcastable_arrays_shapes())
def test_get_boardcastable_arrays_shapes(shapes):
    arrays = [np.ones(shape) for shape in shapes]
    reduce(np.add, arrays)


@given(shapes_indicies=utils_hypothesis.get_boardcastable_arrays_shapes_and_indices())
def test_get_boardcastable_arrays_shapes_and_indices(shapes_indicies):
    shapes, indices, shape = shapes_indicies

    arrays = [np.ones(shape) for shape in shapes]
    array = reduce(np.add, arrays)
    _ = array[indices]

    np.testing.assert_array_equal(shape, array.shape)


@given(shapes=utils_hypothesis.get_chainable_array_shapes())
def test_get_chainable_array_shapes(shapes):
    arrays = [np.ones(shape) if len(shape) == 2 else np.diag(np.ones(shape)) for shape in shapes]
    reduce(np.dot, arrays)


@given(shapes_indicies=utils_hypothesis.get_chainable_arrays_shapes_and_indices())
def test_get_chainable_array_shapes_and_indicies(shapes_indicies):
    shapes, indices, shape = shapes_indicies
    arrays = [np.ones(shape) if len(shape) == 2 else np.diag(np.ones(shape)) for shape in shapes]
    array = reduce(np.dot, arrays)

    np.testing.assert_array_equal(array.shape, shape)
    array[indices]


@given(n_index_axis=utils_hypothesis.get_vector_index_axis())
def test_get_vector_index_axis(n_index_axis):
    n, index, axis = n_index_axis

    a = np.diag(np.ones(n))
    np.testing.assert_array_equal(a[index, :], a[:, index].T)
