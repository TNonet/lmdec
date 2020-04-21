from typing import Tuple, List
import numpy as np
import warnings


def array_split(array_shape: Tuple[int, int], f: float, axis: int = 0, seed: int = 42,
                log: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a partition of 2D array along a specified axis such that the partitions have relative size p and (1-p).

    Parameters
    ----------
    array_shape : 2-Tuple of integers
        Represents the shape of array. Can be found from array.shape
    f : float in (0, 1)
        The relative percentages of the partitions:
            partition_1 has p% (.1 -> 10%)
            partition_2 has (1-p)%
    axis : 0 or 1
        The specified axis to split on.
        If axis == 0:
            partition is on the rows
        If axis == 1:
            partition is on the columns
    seed : int
        Standard seed usage in numpy
    log: int
        See log details in README.

    Returns
    -------
    index_set : np.ndarray of items in the partition
    not_index_set : np.ndarray of the items not in the partition
    """
    if log:
        raise ValueError('Function is not logged.')

    n, p = _array_shape_check(array_shape)

    if axis == 0:
        pass
    elif axis == 1:
        return array_split((p, n), f=f, axis=0, seed=seed)
    else:
        raise ValueError('Can only split on axis = 0 or 1. axis = {}'.format(axis))

    sample_size = int(f * n)

    if f > 1:
        raise ValueError('Cannot split more than unity. p = {}'.format(f))
    elif f < 0:
        raise ValueError('Cannot split non-positive fraction. p = {}'.format(f))

    if sample_size == n:
        warnings.warn('Degenerate Split. Decrease p. p = {} ~= 1'.format(f))
    elif sample_size == 0:
        warnings.warn('Degenerate Split. Decrease p. p = {} ~= 0'.format(f))

    index = np.arange(0, n)
    np.random.seed(seed=seed)
    np.random.shuffle(index)
    index_set = np.sort(index[0:sample_size])
    not_index_set = np.sort(index[sample_size:])

    return index_set, not_index_set


def _array_shape_check(array_shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(array_shape) == 2:
        m, n = array_shape
    elif len(array_shape) == 1:
        m = array_shape[0]
        n = 1
    else:
        raise ValueError('Can only split 2-D arrays. Shape = {}'.format(array_shape))

    return m, n


def cumulative_partition(part_list: List[slice]) -> List[slice]:
    cum_part_list = []
    start = part_list[0].start
    step = part_list[0].step
    previous_end = part_list[0].start

    for part in part_list:
        if part.step != step:
            raise ValueError("Cannot find cumulative partition of non uniform step slices.")
        if previous_end == part.start:
            cum_part_list.append((slice(start, part.stop, step)))
            previous_end = part.stop

    return cum_part_list


def array_geometric_partition(array_shape: Tuple[int, int], f: float, min_size: int = 5, axis: int = 0,
                              log: int = 0) -> Tuple[List[slice], List[int]]:
    """Creates a geometric partition of 2D array along a specified axis such that the partitions
    are ordered have relative sizes that decrease geometrically.

    Effectively:
        part1, part2, part3, ... <- array_geometric_partition(...)
        |parts_i+1| ~= |f*parts_i|

        Partitions are ordered and the array is not shuffled.
        Therefore stacking partitions in order will fully select the origional array

        if axis == 0:
            A[[*part1, *part2, ...], :] == A[:, :]
        if axis == 1:
            A[:, [*part1, *part2, ...]] = A[:, :]

    Parameters
    ----------
    array_shape : 2-Tuple of integers
        Represents the shape of array. Can be found from array.shape
    f : float in (0, 1)
        The relative percentages array found in each of the partitions:
            partition_1 has (f)% elements
            partition_2 has (f**2)% elements
            ...
            partition_i has (f**i)% elements
    min_size : int >= 0
        The minimum size of a partition
        If |partition_i| <= min_size:
            partition_i is increased to min_size
    axis : 0 or 1
        The specified axis to split on.
        If axis == 0:
            partition is on the rows
        If axis == 1:
            partition is on the columns
    log: int
        See log details in README.

    Returns
    -------
    slices : List of slices
        slices[i] refers to partition_i
    partition_sizes : List of ints
        partition_sizes[i] refers to size of partition_i
    """
    if log:
        raise ValueError('Function is not logged.')

    n, p = _array_shape_check(array_shape)

    if axis == 0:
        pass
    elif axis == 1:
        return array_geometric_partition((p, n), f=f, min_size=min_size, axis=0)
    else:
        raise ValueError('Can only partition on axis = 0 or 1. axis = {}'.format(axis))

    if f <= 0:
        raise ValueError('Cannot partition with non-positive geometric series.')
    elif f >= 1:
        raise ValueError('Cannot partition with divergent geometric series.')

    if min_size < 1:
        raise ValueError('Min size must be a positive integer.')

    n_left = n
    partition_sizes = []
    while n_left * f >= min_size:
        part = int(n_left * f)
        partition_sizes.append(part)
        n_left -= part

    while n_left >= min_size:
        partition_sizes.append(min_size)
        n_left -= min_size

    if n_left:
        partition_sizes.append(n_left)

    partition_sizes = sorted(partition_sizes, reverse=True)

    slices = []
    offset = 0

    for end in partition_sizes:
        slices.append(slice(offset, end + offset))
        offset += end

    return slices, partition_sizes


def array_constant_partition(array_shape: Tuple[int, int],
                             f: float,
                             min_size: int = 5,
                             axis: int = 0,
                             log: int = 0) -> List[slice]:
    """Creates a geometric partition of 2D array along a specified axis such that the partitions
    are ordered have relative sizes that are constant.

    Effectively:
        part1, part2, part3, ... <- array_constant_partition(...)
        |parts_i+1| ~= |parts_i|

        Partitions are ordered and the array is not shuffled.
        Therefore stacking partitions in order will fully select the origional array

        if axis == 0:
            A[[*part1, *part2, ...], :] == A[:, :]
        if axis == 1:
            A[:, [*part1, *part2, ...]] = A[:, :]

    Parameters
    ----------
    array_shape : 2-Tuple of integers
        Represents the shape of array. Can be found from array.shape
    f : float in (0, 1)
        The relative percentages array found in each of the partitions:
            partition_1 has (f)% elements
            partition_2 has (f)% elements
            ...
            partition_i has (f)% elements
    min_size : int >= 0
        The minimum size of a partition
        If |partition_i| <= min_size:
            partition_i is increased to min_size
    axis : 0 or 1
        The specified axis to split on.
        If axis == 0:
            partition is on the rows
        If axis == 1:
            partition is on the columns
    log: int
        See log details in README.

    Returns
    -------
    slices : List of slices
        slices[i] refers to partition_i
    """
    if log:
        raise ValueError('Function is not logged.')

    n, p = _array_shape_check(array_shape)

    if axis == 0:
        pass
    elif axis == 1:
        return array_constant_partition((p, n), f=f, min_size=min_size, axis=0)
    else:
        raise ValueError('Can only partition on axis = 0 or 1. axis = {}'.format(axis))

    if f <= 0:
        raise ValueError('Cannot partition with non-positive fraction.')
    elif f > .5:
        raise ValueError('Cannot partition with fraction more than 0.5.')

    if min_size < 1:
        raise ValueError('Min size must be a positive integer.')

    num_partitions = int(1 / f)

    if num_partitions == 1:
        warnings.warn('Only 1 Partition was created. int(1/p) == 1')

    partition_size = n // num_partitions
    if partition_size < min_size:
        partition_size = min_size

    slices = []
    n_left = n
    start = 0
    while n_left >= partition_size:
        slices.append(slice(start, start+partition_size))
        start += partition_size
        n_left -= partition_size

    if n_left > 0 and n_left >= min_size:
        slices.append(slice(start, n))
    elif min_size > n_left > 0:
        slices[-1] = slice(slices[-1].start, n)

    return slices
