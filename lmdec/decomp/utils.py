from collections import namedtuple
from typing import Optional, Union, Tuple

import numpy as np
import sparse
from dask import array as da
from dask import persist, compute

from lmdec.array.chained import ChainedArray
from lmdec.array.stacked import StackedArray

PowerMethodSummary = namedtuple('PowerMethodSummary', ['time', 'acc', 'iter'])


def fill_array(array: da.core.Array, fill_value: int = 0) -> da.core.Array:
    """ Fills a masked array with 'mask_fill' value.

    Parameters
    ----------
    array : masked dask.array.core.Array
    fill_value : int

    Returns
    -------
    filled_array : dask.array.core.Array
        Filled array
    """
    array = da.ma.masked_invalid(array)
    da.ma.set_fill_value(array, fill_value)
    return da.ma.filled(array)


def get_array_moments(array: da.core.Array, mean: bool = True, std: bool = True, std_method: str = 'binom',
                      axis: int = 0) -> Tuple[Optional[da.core.Array], Optional[da.core.Array]]:
    """ Computes specified array_moments

    Parameters
    ----------
    array : array_like, shape (N, P)
        Array that moments will be computed from
    mean : bool
        Flag whether to compute mean of "array" along "axis"
    std : bool
        Flag whether to compute std of "array" along "axis"
    std_method : str
        Method used to compute standard deviation.

        Possible methods are:
            'norm' ===> Normal Distribution Standard Deviation. See np.std
            'binom' ====> Binomial Standard Deviation
                            sqrt(2*p*(1-p)), where p = "mean"/2
    axis : int
        Axis to compute mean and std along.

    Returns
    -------
    array_mean : da.core.array, optional
        If "mean" is false, returns None
        Otherwise returns the array mean
    array_std: da.core.array, optional
        If "std" is false, returns None
        Otherwise returns the array std
    """
    array_mean = None
    array_std = None

    if mean:
        array_mean = da.nanmean(array, axis=axis)

    if std:
        if std_method == 'binom':
            u = array_mean if mean else da.nanmean(array, axis=axis)
            u /= 2
            array_std = da.sqrt(2*u*(1-u))
        elif std_method == 'norm':
            array_std = da.nanstd(array, axis=axis)
        else:
            raise NotImplementedError(f'std_method, {std_method}, is not implemented ')

    array_mean, array_std = persist(array_mean, array_std)

    return array_mean, array_std


def axis_wise_COO_data_by_axis(values: np.ndarray, coords: Tuple[np.ndarray, np.ndarray],
                               axis: int = 1) -> np.ndarray:
    """Gathers appropriate elements of "values" for a COO sparse array by axis

    If an array is missing values to be imputed by either row or column summaries (mean for example).

    The representative coordinates of the missing data must be know for two reasons.
        The corresponding row or column index is important to know which value is used to replace a missing value

    This method takes the summary values, "values", and the missing value coordinates "coords" and the axis used to
    calculate the summary values

    Parameters
    ----------
    values : array_like, shape (P, )
        Values that correspond either to the columns or rows of an array

    coords : tuple of array_like, each of shape (NNZ, )
        Coordinates that will be filled by selected elements of "values"

    axis : int
        Axis used to select which dimension of coords to use

    Returns
    -------
    data : array_like
        Respecitve Values used to replace the Missing value referenced by coords

        if array[i,j] is missing then i exists in coords[0] at ix and j exists in coords[1] at ix as well.
        The value to replace array[i,j] = data[ix]

    """
    return values[coords[axis].tolist()]


def mask_imputation(array: da.core.Array, mask_values: Optional[da.core.Array] = None, fill_value: int = 0,
                    mask_method: str = 'mean', mask_axis: int = 0) -> Tuple[da.core.Array, sparse._coo.core.COO]:
    """ Creates the mask that will fill "array" and the filled_array that has the missing values of "array" filled.

    If A is array and has missing values

    A = [[1, 2, 3],
         [?, 4, 5],
         [3, 4, ?]]

    Then mask is a Sparse COO array that has teh following entries

    mask = [[-, -, -],
            [a, -, -],
            [-, -, b]] Where "-" refers 0, but is replaced with "-" to show that value is not stored

    Then the filled array is:

    A_filled = [[1, 2, 3],
                [f, 4, 5],
                [3, 4, f]] Where "f" refers to a common fill value specified as "fill_value"


    Parameters
    ----------
    array : array_like, shape (N, P)
        Array that a copy of will be filled, and if needed mask values will be computed from
    mask_values : array_like, shape (P,) optional
        Values to fill mask with, if already computed
    fill_value : int
        Value that will be used to fill NaN values in array
    mask_method : str
        Method used to compute mask_values. Only used if mask_values is not specified
    mask_axis : int
        Axis in which values will be computed from.

        axis = 0 ===> column summary of values
        axis = 1 ===> row summary of values

    Returns
    -------
    filled_array: dask array, shape (N, P)
        copy of "array" with nan_values filled, if specified
    mask : dask array, shape (N, P)
        sparse dask array with mask values where "array" is NaN.

    """
    if not isinstance(array._meta, np.ndarray):
        raise ValueError(f'expected meta, {type(np.ndarray)},  but got {type(array._meta)}')

    if mask_values is None:
        if mask_method == 'mean':
            mask_values = da.nanmean(array, axis=mask_axis).compute()
        else:
            raise NotImplementedError(f'mask_method, {mask_method}, is not implemented ')
    else:
        try:
            mask_values = mask_values.compute()
        except AttributeError:
            pass

    coords = compute(*da.where(da.isnan(array)))
    if not len(coords[0]):
        raise ValueError('expected array to have maskable values, but got none.')

    data = axis_wise_COO_data_by_axis(mask_values, coords, axis=1 - mask_axis)

    mask = sparse.COO(coords=np.vstack(coords), data=data, shape=array.shape, has_duplicates=False, cache=True)
    mask = da.from_array(mask, chunks=array.shape)

    filled_array = fill_array(array, fill_value=fill_value).persist()

    return filled_array, mask


def make_snp_array(array: da.core.Array, mean: bool = True, std: bool = True, std_method: str = 'binom',
                   dtype='int8', rechunk: Union[bool, str, dict] = 'auto', mask_method: str = 'mean',
                   mask_nan: bool = True) -> da.core.Array:
    """Creates a SNP Array from "array" that;
        1. Has zero column mean, if 'mean' is True
        2. Has unit standard deviation according to 'std_method', if 'std' is True
        3. Has type of type "dtype".
        4. Is rechunked according to 'rechunk'
        5. Has NaN values replaced with imputed values following 'mask_method' for each column if 'mask_nan' is True

    Parameters
    ----------
    array : da.core.Array, shape (N, P)
        base array to be masked, centered, and scaled (if specified)

    mean : bool
        Flag whether to use center 'array'

    std : bool
        Flag whether to scale  'array'

    std_method : str
        Specification for how to scale 'std'

    dtype : str
        Numpy datatype that array will be cast into once NaN's are removed if specified by 'mask_nan'

    rechunk : bool, str, dict
        Underlying array is rechunked once cast into 'dtype' an NaN values are removed if specified by 'mask_nan'

        See https://docs.dask.org/en/latest/array-api.html#dask.array.rechunk

    mask_method : str
        Method for imputing values to replace NaN values.

        if 'mask_method' == 'mean':
            NaN values are replaced with column means
            (see https://numpy.org/doc/1.18/reference/generated/numpy.nanmean.html)

    mask_nan : bool
        Flag whether to mask NaN values (or to check if NaN values exist)

    Returns
    -------
    array : da.core.Array

    Notes
    -----
    It is assumed that NaN values are sparse, if they exist.
    If this is not the case, the performance will be quite slow.

    NaN values are filled with 0 in array
    NaN value locations are recorded as coords for a COO sparse array.


    SNP = ((array + mask) - U)D
            ^^^^^^^^^^^^    |  |
            Masked Array    |  |
            ^^^^^^^^^^^^^^^^^  |
            Centered Array     |
            ^^^^^^^^^^^^^^^^^^^^
            Scaled Array

    """
    mean_array, std_array = get_array_moments(array, mean=mean, std=std, std_method=std_method, axis=0)

    mask_valid = False
    if mask_nan:
        try:
            if mask_method == 'mean' and mean:
                array, mask_array = mask_imputation(array, mean_array, fill_value=0, mask_axis=0)
            else:
                array, mask_array = mask_imputation(array, mask_method=mask_method, fill_value=0, mask_axis=0)
            mask_valid = True
        except ValueError:
            pass

    array = array.astype(dtype)

    if rechunk:
        if isinstance(rechunk, dict):
            array = array.rechunk(**rechunk)
        else:
            array = array.rechunk(rechunk)

    if mask_nan and mask_valid:
        array = StackedArray((array, mask_array))

    if mean:
        array = StackedArray((array, -mean_array))

    if std:
        array = ChainedArray((array, 1/std_array))

    return array
