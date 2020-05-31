from typing import List, Callable, Generator, Union, Iterable

import numpy as np
import dask

import dask.array as da


def tree_reduction(items: Union[List[dask.array.core.Array],
                                Generator[dask.array.core.Array, None, None]],
                   binary_operator: Callable[[dask.array.core.Array, dask.array.core.Array],
                                             dask.array.core.Array]) -> dask.array.core.Array:
    items = list(items)

    while len(items) > 1:
        reduced_items = []
        for i in range(0, len(items) - 1, 2):
            item = binary_operator(items.pop(), items.pop())
            reduced_items.append(item)
        items = reduced_items + items

    return items[0]


def linear_reduction(items: Union[List[dask.array.core.Array],
                                  Generator[dask.array.core.Array, None, None]],
                     binary_operator: Callable[[dask.array.core.Array, dask.array.core.Array],
                                               dask.array.core.Array]) -> dask.array.core.Array:
    if isinstance(items, list):
        items = iter(items)

    reduced = next(items)
    for item in items:
        reduced = binary_operator(reduced, item)
    return reduced


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

    >>>B, C, D, E = (dask.random.random(shape=(10,10)) for _ in range(4))
    ...A = StackedArray([B, C, D, E])
    ...assert type(A) == lmdec.array.stacked.StackedArray
    ...assert type(A.array) == dask.array.core.Array

    A.dot(x) and A.array.dot(x) preform very different operations as StackedArray re-implements dot.
    A.std(x) and A.array.std(x) preform the exact same operation as StackedArray does not re-implement dot.
    """

    def __new__(cls, arrays: Iterable[dask.array.core.Array], tree_reduce: bool = True):
        if isinstance(arrays, dask.array.core.Array):
            raise ValueError(f"expected Iterable of dask.array.core.Array. Got dask.array.core.Array")

        arrays = list(arrays)

        if any(not isinstance(x, (dask.array.core.Array, np.ndarray)) for x in arrays):
            raise ValueError(f"expected Iterable of dask.array.core.Array.")

        if len({x.shape for x in arrays}) != 1:
            raise ValueError(f"expected constant shape of arrays, got {set([x.shape for x in arrays])}")

        if tree_reduce:
            reduce = tree_reduction
        else:
            reduce = linear_reduction

        array = reduce(arrays, da.add)

        self = super(StackedArray, cls).__new__(cls, array.dask, array.name, array.chunks, array.dtype,
                                                array._meta, array.shape)
        self.tree_reduce = tree_reduce
        self.reduce = reduce
        self._arrays = arrays
        self.array = array

        return self

    @property
    def arrays(self):
        yield from self._arrays

    @property
    def T(self):
        return StackedArray((array.T for array in self.arrays), tree_reduce=self.tree_reduce)

    @property
    def width(self):
        return len(self._arrays)

    def dot(self, x):
        return self.reduce((array.dot(x) for array in self.arrays), da.add)

    def persist(self):
        return StackedArray((array.persist() for array in self.arrays), tree_reduce=self.tree_reduce)

    def mean(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        if out is not None:
            raise NotImplementedError(f'`out` argument is not supported for {StackedArray.__name__}')
        means = (da.mean(array, axis=axis, dtype=dtype, keepdims=keepdims, split_every=split_every, out=None)
                 for array in self.arrays)
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

    def get_subarray(self, item):
        return self._arrays[item]

    def __getitem__(self, item):
        return StackedArray((array[item] for array in self.arrays), tree_reduce=self.tree_reduce)

    def __repr__(self):
        spaces = '\n' + ' ' * (len(StackedArray.__name__) + 1)
        stacked_arrays = spaces.join([str(array) for array in self.arrays])
        return f"{StackedArray.__name__}({stacked_arrays})"

    def rechunk(self, chunks='auto', threshold=None, block_size_limit=None):
        return StackedArray((array.rechunk(chunks, threshold, block_size_limit) for array in self.arrays),
                            tree_reduce=self.tree_reduce)

    def reshape(self, *shape):
        arrays = (array.reshape(*shape) for array in self.arrays)
        return StackedArray((array.reshape(*shape) for array in self.arrays), tree_reduce=self.tree_reduce)
