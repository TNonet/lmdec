from typing import Union, Iterable, List

import numpy as np
import dask

import dask.array as da

from lmdec.array.matrix_ops import expand_arrays
from lmdec.array.reduce import tree_reduction, linear_reduction
from lmdec.array.utils import issparse


class StackedArray(dask.array.core.Array):
    """ Stacked Dask Array

    A stack of parallel nd-array comprised of many numpy arrays arranged in a grid.

    While the stack can be combined into one array, this is delayed to optimize matrix operations.

    Parameters
    ----------
    arrays : Iterable of dask.array.core.Array
        Underlying dask arrays that make up StackedArray

        A = B + C + D + E

        A = StackedArray([B, C, D, E])
                         ^^^^^^^^^
                         arrays
    tree_reduce: bool
        Flag whether to use a tree reduction or a linear reduction

        tree reduction:
            BigO faster than linear reduction (reduce in log time). However, they can require more memory.

            Example:
                              A
                              ^
                       X ====>^<==== Y
                       ^             ^
                   B =>^<= C     D =>^<= E

        linear reduction:
            BigO slower than tree reduction (reduce in linear time). However, can use less memory.

            Example:
                            A
                            ^
                        B =>^<= X
                                ^
                            C =>^<= Y
                                    ^
                                D =>^<= E

    Notes
    -----
    StackedArray overrides methods/properties:
        T:
            returns a new StackedArray of the Transposes of underlying

            sa = StackedArray([A, B, C, D])
            sa.T <==> StackedArray([A.T, B.T, C.T, D.T])

        dot:
            computes sum of the dot products of the underlying arrays

            sa = StackedArray([A, B, C, D])
            sa.dot(x) <==> A.dot(x) + B.dot(x) + C.dot(x) + D.dot(x)

        persist:
            returns a new StackedArray of the persisted underlying arrays

            sa = StackedArray([A, B, C, D])
            sa.persist() <===> StackedArray([A.persist(), B.persist(), C.persist(), D.persist()])

        mean:
            computes the mean of the mean of underlying arrays

            sa = StackedArray([A, B, C, D])
            sa.mean() <==== mean(A.mean(), B.mean(), C.mean(), D.mean())

        __getitem__:
            returns a new StackedArray of the subarrays requesting in __getitem__ of each underlying array

            sa = StackedArray([A, B, C, D])
            sa[item] <==== StackedArray([A[item], B[item], C[item], D[item])

        rechunk:
            returns a new StackedArray of each underlying array rechunked

            sa = StackedArray([A, B, C, D])
            sa.rechunk(...) <==== StackedArray([A.rechunk(...), B.rechunk(...), C.rechunk(...), D.rechunk(...))

        reshape:
            returns a new StackedArray of each underlying array reshaped

            sa = StackedArray([A, B, C, D])
            sa.reshape(...) <==== StackedArray([A.reshape(...), B.reshape(...), C.reshape(...), D.reshape(...))

    All other operations are preformed on the combined array where the underlying arrays are reduced according to the
    flag `tree_reduce`

    To gain access to the underlying array for these operations use method `array`

    >>>B, C, D, E = (dask.array.random.random(shape=(10,10)) for _ in range(4))
    ...A = StackedArray([B, C, D, E])
    ...assert type(A) == lmdec.array.stacked.StackedArray
    ...assert type(A.array) == dask.array.core.Array

    A.dot(x) and A.array.dot(x) preform very different operations as StackedArray re-implements dot.
    A.std(x) and A.array.std(x) preform the exact same operation as StackedArray does not re-implement dot.

    BroadCasting
        The Broadcasting Rule:
            "In order to broadcast, the size of the trailing axes for both arrays in an operation must either be the
            same size or one of them must be one." - https://numpy.org/devdocs/user/theory.broadcasting.html
    """

    def __new__(cls, arrays: Iterable[dask.array.core.Array], tree_reduce: bool = True):
        if isinstance(arrays, dask.array.core.Array):
            raise ValueError(f"expected Iterable of dask.array.core.Array. Got dask.array.core.Array")

        arrays = list(arrays)

        if any(not isinstance(x, (dask.array.core.Array, np.ndarray)) for x in arrays):
            raise ValueError(f"expected Iterable of dask.array.core.Array, "
                             f"but got types {set(type(x) for x in arrays)}")

        if tree_reduce:
            reduce = tree_reduction
        else:
            reduce = linear_reduction

        try:
            array = reduce(arrays, da.add)
        except ValueError:
            raise ValueError(f'expected arrays to have broadcast-able shapes, got {set([a.shape for a in arrays])}')

        self = super(StackedArray, cls).__new__(cls, array.dask, array.name, array.chunks, array.dtype,
                                                array._meta, array.shape)
        self.tree_reduce = tree_reduce
        self.reduce = reduce
        self._arrays = arrays
        self.array = array

        return self

    @property
    def arrays(self) -> List[da.core.Array]:
        return self._arrays

    @property
    def T(self) -> "StackedArray":
        if self.ndim != 2:
            raise NotImplementedError
        transposed_arrays = []
        n, p = self.shape
        for array in self.arrays:
            if array.shape == (n, p):
                out = array.T
            elif array.shape in [(p,), (1, p)]:
                out = array.reshape(p, 1)
            else:
                out = array.reshape(1, n)
            transposed_arrays.append(out)
        return StackedArray(transposed_arrays, tree_reduce=self.tree_reduce)

    @property
    def width(self):
        return len(self._arrays)

    def dot(self, x: Union[dask.array.core.Array, np.ndarray]) -> da.core.Array:
        """ See np.dot

        Notes
        -----
        See np.dot?

        Dot product of `self` and x. Specifically,

            - If both `self` and `x` are 1-D arrays, it is inner product of vectors
            (without complex conjugation).

            - If both `self` and `x` are 2-D arrays, it is matrix multiplication,
            but using :func:`matmul` or ``a @ b`` is preferred.

            - If either `a` or `b` is 0-D (scalar), it is equivalent to :func:`multiply`
              and using ``numpy.multiply(a, b)`` or ``a * b`` is preferred.

        The next two cases are not Implemented:

            - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
              the last axis of `a` and `b`.

            - If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
              sum product over the last axis of `a` and the second-to-last axis of `b`::

                dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
        """
        if self.ndim != 2 or x.ndim > 2:
            raise NotImplementedError
        n, p = self.shape
        dot_sums = []
        for array in self.arrays:
            if array.shape == self.shape:
                out = array.dot(x)
            elif array.shape == (n, 1):
                row = x.sum(axis=0)
                out = array * row
                if x.ndim == 1:
                    out = np.squeeze(out)
            elif array.shape == (1, p):
                out = np.squeeze(array).dot(x)
            elif array.shape == (p,):
                out = array.dot(x)
            dot_sums.append(out)
        return self.reduce(dot_sums, da.add).rechunk('auto')

    def persist(self) -> "StackedArray":
        return StackedArray((array.persist() for array in self.arrays), tree_reduce=self.tree_reduce)

    def mean(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None) -> da.core.Array:
        if out is not None:
            raise NotImplementedError(f'`out` argument is not supported for {StackedArray.__name__}')
        means = (da.mean(array, axis=axis, dtype=dtype, keepdims=keepdims, split_every=split_every, out=None)
                 for array in expand_arrays(self.arrays))
        return self.reduce(means, da.add)

    # def std(self, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None):
    #     if out is not None:
    #         raise NotImplementedError(f'`out` argument is not supported for {StackedArray.__name__}')
    #     stds = (da.std(array, axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof, split_every=split_every, out=None)
    #             for array in self.arrays)
    #     means = [da.mean(array, axis=axis, dtype=dtype, keepdims=keepdims, split_every=split_every, out=None)
    #              for array in self.arrays]
    #     mean = self.reduce((mu for mu in means), da.add)
    #
    #     sum_weight_std = da.zeros_like(mean)
    #     for mu, std in zip(means, stds):
    #         sum_weight_std += (std ** 2 + (mu - mean) ** 2)
    #
    #     return da.sqrt(sum_weight_std)

    def get_subarray(self, item) -> da.core.Array:
        return self._arrays[item]

    def __getitem__(self, item) -> "StackedArray":
        if not isinstance(item, tuple):
            return StackedArray((array[item] for array in self.arrays), tree_reduce=self.tree_reduce)
        else:
            if len(item) > self.ndim:
                raise IndexError('Too many indices for array')
            sub_arrays = []
            for array in self.arrays:
                sub_array_item = []
                for i, s in zip(item[-len(array.shape):], array.shape):
                    if s > 1:
                        sub_array_item.append(i)
                    elif isinstance(i, slice):
                        sub_array_item.append(slice(None, None, None))
                    else:
                        sub_array_item.append(0)
                sub_array_item = tuple(sub_array_item)
                sub_arrays.append(array[sub_array_item])
            return StackedArray(sub_arrays, tree_reduce=self.tree_reduce)

    def __repr__(self):
        spaces = '\n' + ' ' * (len(StackedArray.__name__) + 1)
        stacked_arrays = spaces.join([str(array) for array in self.arrays])
        return f"{StackedArray.__name__}({stacked_arrays})"

    def rechunk(self, chunks='auto', threshold=None, block_size_limit=None) -> "StackedArray":
        arrays = []
        for array in self.arrays:
            if array.ndim == 1 or max(array.shape) == np.product(array.shape):
                arrays.append(array)
            elif issparse(array):
                arrays.append(array)
            else:
                arrays.append(array.rechunk(chunks, threshold, block_size_limit))
        return StackedArray(arrays, tree_reduce=self.tree_reduce)

    def reshape(self, *shape) -> "StackedArray":
        raise NotImplementedError
