from hypothesis import given, assume, settings
from hypothesis.strategies import integers, booleans, floats
from hypothesis.extra import numpy as npst

import numpy as np
from dask import array as da

from lmdec.array.stacked import StackedArray
from lmdec.decomp import utils


@given(npst.array_shapes(min_dims=2, max_dims=2), integers(min_value=2, max_value=10), booleans())
def test_COO_data_by_axis(shape, value, axis):
    ones_arr = np.ones(shape)

    n, p = shape
    if axis:
        data = np.random.randint(1, value, size=(p, ))
    else:
        data = np.random.randint(1, value, size=(n, 1))

    arr = ones_arr*data

    coords = np.where(arr == value - 1)

    data = utils.axis_wise_COO_data_by_axis(data, coords, axis)

    np.testing.assert_equal(np.linalg.norm(data - value + 1), 0)


@given(npst.array_shapes(min_dims=2, max_dims=2), booleans())
def test_get_array_moments_non_nans(shape, axis):
    axis = int(axis)
    assume(shape[0] > 1 and shape[1] > 1)

    arr = da.random.random(size=shape)

    mean, std = utils.get_array_moments(arr, True, True, 'norm', axis=axis)

    np.testing.assert_array_almost_equal(mean, arr.mean(axis=axis))
    np.testing.assert_array_almost_equal(std, arr.std(axis=axis))

    mean, std = utils.get_array_moments(arr, True, True, 'binom', axis=axis)

    u = arr.mean(axis=axis)/2
    np.testing.assert_array_almost_equal(mean, 2*u)
    np.testing.assert_array_almost_equal(std, np.sqrt(2*u*(1-u)))


@given(npst.array_shapes(min_dims=2, max_dims=2), booleans(), floats(0, 1))
@settings(deadline=None)
def test_mask_imputation(shape, axis, clip_value):
    assume(shape[0] > 1 and shape[1] > 1)
    axis = int(axis)

    n, p = shape

    arr = da.random.random(size=shape)
    arr_result = arr.copy()

    arr[arr < clip_value] = float('nan')
    arr_result[arr_result < clip_value] = 1

    if axis:
        values = da.ones(n)
    else:
        values = da.ones(p)

    try:
        filled_arr, mask_arr = utils.mask_imputation(arr, mask_values=values, mask_axis=axis)
    except ValueError:
        assert np.count_nonzero(np.isnan(arr)) == 0
    else:
        assume(mask_arr.compute().data.size > 0)

        combined_arr = StackedArray((filled_arr, mask_arr))

        np.testing.assert_array_equal(arr_result, combined_arr.array)


@given(npst.array_shapes(min_dims=2, max_dims=2), floats(0, 1))
@settings(deadline=None)
def test_make_snp_array_case_binom(shape, threshold):
    assume(shape[0] > 1 and shape[1] > 1)  # Assumes not degenerate 2d Array

    arr = da.random.random(size=shape)
    arr[arr > threshold] = float('nan')

    assume(da.mean(da.mean(da.isnan(arr), axis=0) < 1) == 1)
    # Asserts that every tested arr has at least 1 non-nan value in each column

    snp_array = utils.make_snp_array(arr, mean=True, std=True, std_method='binom', dtype='float')

    mean = snp_array.mean(axis=0)
    np.testing.assert_array_almost_equal(1 + mean, np.ones(shape[1]))

    # TODO: What does it mean to have unit binomial std if the mean is 0?
    # u = mean/2
    # np.testing.assert_array_almost_equal(da.sqrt(u*(1-u)), np.ones(shape[1]))


@given(npst.array_shapes(min_dims=2, max_dims=2), floats(0, 1))
@settings(deadline=None)
def test_make_snp_array_case_normal(shape, threshold):
    assume(shape[0] > 1 and shape[1] > 1)  # Assumes not degenerate 2d Array

    arr = da.random.random(size=shape)
    arr[arr > threshold] = float('nan')

    assume(da.mean(da.nanstd(arr, axis=0) > 0) == 1)
    # Asserts that every tested arr has a non-zero std for each column

    snp_array = utils.make_snp_array(arr, mean=True, std=True, std_method='norm', dtype='float')

    np.testing.assert_array_almost_equal(1 + snp_array.mean(axis=0), np.ones(shape[1]))

    # TODO: What should the STD of the full array be with filled in values?
    # np.testing.assert_array_almost_equal(snp_array.std(axis=0), np.ones(shape[1]))


@given(npst.array_shapes(min_dims=2, max_dims=2), integers(2, 10), booleans())
@settings(deadline=None)
def test_make_snp_array_case_normal(shape, max_value, mask_nans):
    assume(shape[0] > 1 and shape[1] > 1)  # Assumes not degenerate 2d Array

    arr = da.random.randint(0, max_value, size=shape)
    if mask_nans:
        arr[arr == max_value-1] = float('nan')

    assume(da.mean(da.nanstd(arr, axis=0) > 0) == 1)
    # Asserts that every tested arr has a non-zero std for each column

    snp_array = utils.make_snp_array(arr, mean=True, std=True, std_method='norm', mask_nan=mask_nans, dtype='int8')

    np.testing.assert_array_almost_equal(1 + snp_array.mean(axis=0), np.ones(shape[1]))

    # TODO: What should the STD of the full array be with filled in values?
    # np.testing.assert_array_almost_equal(snp_array.std(axis=0), np.ones(shape[1]))


def test_mask_snp_array_casse1():
    array = np.random.rand(100, 80)
    mu = array.mean(axis=0)
    std = np.diag(1 / array.std(axis=0))
    scaled_centered_array = (array - mu).dot(std)
    array = utils.make_snp_array(da.array(array), mean=True, std=True, std_method='norm',
                                 mask_nan=False, dtype='float64')

    np.testing.assert_array_almost_equal(scaled_centered_array, array)