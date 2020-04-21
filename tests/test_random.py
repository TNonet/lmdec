import numpy as np
import pytest

from lmdec.array.random import array_split, _array_shape_check, array_constant_partition, \
    array_geometric_partition, cumulative_partition


def test_array_split():
    n, p = 1000, 800
    a = np.random.rand(n, p)

    for f in [0.01, .1, .25, .5, .9, .99]:
        I, not_I = array_split(a.shape, f=f)
        np.testing.assert_array_almost_equal(np.mean(a, axis=0), np.mean(a[[*I, *not_I], :], axis=0))
        assert len(I) == int(f*n)
        assert len(not_I) == n - int(f*n)

    for f in [0.01, .1, .25, .5, .9, .99]:
        I, not_I = array_split(a.shape, f=f, axis=1)
        np.testing.assert_array_almost_equal(np.mean(a, axis=1), np.mean(a[:, [*I, *not_I]], axis=1))
        assert len(I) == int(f*p)
        assert len(not_I) == p - int(f*p)


def test_array_split_bad_args():
    with pytest.raises(ValueError):
        array_split((10, 10), f=-.1)

    with pytest.raises(ValueError):
        array_split((10, 10), f=1.1)

    with pytest.raises(ValueError):
        array_split((10, 10), f=.1, axis=2)


def test__array_shape_check():
    with pytest.raises(ValueError):
        _array_shape_check((10, 10, 10))

    m, n = _array_shape_check((10,))
    assert m == 10
    assert n == 1


def test_array_constant_partition_sizes():
    a = np.random.rand(10, 20)
    parts = array_constant_partition(a.shape, f=.1, min_size=1)
    assert len(parts) == 10

    parts = array_constant_partition(a.shape, f=.1, min_size=5)
    assert len(parts) == 2

    parts = array_constant_partition(a.shape, f=.1, axis=1, min_size=1)
    assert len(parts) == 10

    parts = array_constant_partition(a.shape, f=.1, axis=1, min_size=5)
    assert len(parts) == 4


def test_array_constant_partition_bad_size():
    array_constant_partition((10, 10), f=.5)
    with pytest.raises(ValueError):
        array_constant_partition((10, 10), f=.51)

    with pytest.raises(ValueError):
        array_constant_partition((10,10,10), f=.51)


def test_array_constant_partition_array():
    n, p = 1000, 2000
    a = np.random.rand(1000, 2000)

    for f in [0.01, .1, .25, .5]:
        parts = array_constant_partition(a.shape, f=f)
        for i, part in enumerate(parts):
            np.testing.assert_array_equal(a[int(i*f*n):int((i+1)*(f*n))], a[part, :])

        parts = array_constant_partition(a.shape, f=f, axis=1)
        for i, part in enumerate(parts):
            np.testing.assert_array_equal(a[:, int(i * f * p):int((i + 1) * (f * p))], a[:, part])


def test_array_geometric_partition():
    a = np.random.rand(16, 32)
    parts, sizes = array_geometric_partition(a.shape, f=.5, min_size=1)
    assert len(parts) == 5

    parts, sizes = array_geometric_partition(a.shape, f=.5, min_size=1, axis=1)
    assert len(parts) == 6
    #
    parts, sizes = array_geometric_partition(a.shape, f=.25, min_size=4, axis=0)
    assert len(parts) == 4
    #
    parts, sizes = array_geometric_partition(a.shape, f=.25, min_size=4, axis=1)
    assert len(parts) == 7


def test_array_geometric_partition_bad_size():
    with pytest.raises(ValueError):
        array_geometric_partition((10, 10, 10), f=.1)

    with pytest.raises(ValueError):
        array_geometric_partition((10, 10), f=-.1)

    with pytest.raises(ValueError):
        array_geometric_partition((10, 10), f=1.1)


def test_cumulative_partition():
    n, p = 1000, 1000
    a = np.random.randn(n, p)

    for f in [.001, .1, .25, .5]:
        parts = array_constant_partition(a.shape, f=f, min_size=1)
        cum_parts = cumulative_partition(parts)

        for i, cum_part in enumerate(cum_parts):
            a_actual = a[:int((i+1)*n*f), :]
            a_part_stack = np.vstack([a[p, :] for p in parts[:i+1]])
            a_cum = a[cum_part, :]

            np.testing.assert_array_equal(a_actual, a_part_stack)
            np.testing.assert_array_equal(a_actual, a_cum)

        parts = array_constant_partition(a.shape, f=f, min_size=1, axis=1)
        cum_parts = cumulative_partition(parts)

        for i, cum_part in enumerate(cum_parts):
            a_actual = a[:, 0:int((i + 1) * p * f)]
            a_part_stack = np.hstack([a[:, p] for p in parts[:i+1]])
            a_cum = a[:, cum_part]

            np.testing.assert_array_equal(a_actual, a_part_stack)
            np.testing.assert_array_equal(a_actual, a_cum)
