from typing import Iterable, Generator, Union

import numpy as np
import dask
import dask.array as da

from lmdec.array.matrix_ops import vector_to_sparse
from lmdec.array.reduce import tree_reduction, linear_reduction


class ChainedArray(dask.array.core.Array):
    """ Chained Dask Array

    A chain of parallel {1,2}d-array comprised of many numpy arrays arranged in a grid.

    While the chain can be combined into one array, this is delayed to optimize matrix operations.

    Parameters
    ----------
    arrays : Iterable of dask.array.core.Array
        Underlying dask arrays that make up ChainedArray

        A = B @ C @ D @ E, where @ denotes a matrix multiplication

        A = ChainedArray([B, C, D, E])
                         ^^^^^^^^^
                         arrays

        Any vector will be interpreted as the diagonal of a diagonal matrix

    tree_reduce: bool
        **NOTE** Reduction algorithm is only used in __new__ to create ChainedArray.array.
        Most operations for a ChainedArray are linear due to its lack of distributive property

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
    ChainedArray overrides methods/properties:
        T:
            returns a new ChainedArray of the Transposes of underlying

            sa = ChainedArray([A, B, C, D])
            sa.T <==> ChainedArray([D.T, C.T, B.T, A.T])

        dot:
            computes sum of the dot products of the underlying arrays

            sa = ChainedArray([A, B, C, D])
            sa.dot(x) <==> A.dot(B.dot(C.dot(D.dot(x))))

        persist:
            returns a new ChainedArray of the persisted underlying arrays

            sa = ChainedArray([A, B, C, D])
            sa.persist() <===> ChainedArray([A.persist(), B.persist(), C.persist(), D.persist()])

        mean:
            TODO: Implement mean Operation

        __getitem__:
            returns a new ChainedArray of the subarrays requesting in __getitem__ of each underlying array

            sa = ChainedArray([A, B, C, D])
            sa[item] <==== ChainedArray([A[item[0], :], B, C, D[:, item[0])

        rechunk:
            returns a new ChainedArray of each underlying array rechunked

            sa = ChainedArray([A, B, C, D])
            sa.rechunk(...) <==== ChainedArray([A.rechunk(...), B.rechunk(...), C.rechunk(...), D.rechunk(...))


    All other operations are preformed on the combined array where the underlying arrays are reduced according to the
    flag `tree_reduce`

    To gain access to the underlying array for these operations use method `array`

    >>>B, C, D, E = (dask.array.random.random(shape=(10,10)) for _ in range(4))
    ...A = ChainedArray([B, C, D, E])
    ...assert type(A) == lmdec.array.stacked.ChainedArray
    ...assert type(A.array) == dask.array.core.Array

    A.dot(x) and A.array.dot(x) preform very different operations as ChainedArray re-implements dot.
    """

    def __new__(cls, arrays: Iterable[dask.array.core.Array], tree_reduce: bool = True):
        # TODO: Use minimize operation reduction with dynamic programming.

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

        prev_col = None
        for array in arrays:
            if array.ndim > 2:
                raise NotImplementedError(f'expected 1D or 2D arrays, got {array.ndim}D array')
            elif array.ndim == 1:
                n = p = array.shape[0]
            else:
                n, p = array.shape

            if prev_col is None:
                prev_col = p
            else:
                if prev_col != n:
                    raise ValueError(f'expected chain-able dimensions, got {[a.shape for a in arrays]}')
                else:
                    prev_col = p

        array = reduce((a if a.ndim == 2 else da.diag(a) for a in arrays), da.dot)

        self = super(ChainedArray, cls).__new__(cls, array.dask, array.name, array.chunks, array.dtype,
                                                array._meta, array.shape)
        self.tree_reduce = tree_reduce
        self.reduce = reduce
        self._arrays = arrays
        self.array = array

        return self

    @property
    def arrays(self) -> Generator[da.core.Array, None, None]:
        yield from self._arrays

    @property
    def T(self) -> "ChainedArray":
        if self.ndim != 2:
            raise NotImplementedError
        return ChainedArray(reversed([a.T if a.ndim == 2 else a for a in self.arrays]), tree_reduce=self.tree_reduce)

    @property
    def width(self):
        return len(self._arrays)

    def dot(self, x: Union[np.ndarray, da.core.Array]) -> da.core.Array:
        if x.ndim > 2:
            raise NotImplementedError
        for array in reversed(list(self.arrays)):
            if array.ndim == 2:
                x = array.dot(x)
            elif array.ndim == 1:
                if x.ndim == 2:
                    array = da.tile(array, (x.shape[1], 1)).T
                x = array * x
        return x.rechunk('auto')

    def get_subarray(self, item) -> da.core.Array:
        return self._arrays[item]

    def __getitem__(self, item) -> "ChainedArray":
        """ See numpp.ndarray.__getitem__

        Notes
        -----
        Due to the delayed construction of array, slicing is drastically different. Suppose A, is a chained array such
        that is decomposes into the matrix multiplication chain:
            [A1, A2, A3, ..., Ak] ==> A = A1@A2@A3@...@Ak

        Therefore, the shape of A is only determined by A1 and Ak.

            (n1 x p1)(p1 x p2)(p2 x p3)...(pk-2, pk-1)(pk-1 x pk)
                  ^^^^^^   ^^^^^^   ^^^^^^^^^^^  ^^^^^^^^^^
                    All inner dimensions cancel
        Therefore A is of shape
            (n1 x pk)

        To get sub-matrices of A we only need to find the appropriate sub matrices of A1 and Ak

        To select rows of A, we select the appropriate rows of A1

            A[I, :] ==> A1[I, :]@A2@A3@...@Ak

             (|I| x p1)(p1 x p2)(p2 x p3)...(pk-2, pk-1)(pk-1 x pk)

        To select columns of A, we select the appropriate columns of Ak

            A[:, I] ==> A1@A2@A3@...@Ak[:, I]

             (n1 x p1)(p1 x p2)(p2 x p3)...(pk-2, pk-1)(pk-1 x |I|)

        Rows and columns can be selected at the same time:

        A[I, J] ==> A1[I,:]@A2@A3@...@Ak[:, J]

             (|I| x p1)(p1 x p2)(p2 x p3)...(pk-2, pk-1)(pk-1 x |J|)

        However, only slices are implemented.
        TODO: Implement full numpy fancy indexing

        """
        if not isinstance(item, tuple):
            try:
                return ChainedArray((array[item] for array in self.arrays), tree_reduce=self.tree_reduce)
            except ValueError:
                raise NotImplementedError(f'__getitem__({item}) not implemented for {ChainedArray.__name__}')

        if len(item) > self.ndim:
            raise IndexError('Too many indices for array')
        if any(not isinstance(i, slice) for i in item):
            raise NotImplementedError('Only implements slice indexing')
        arrays = list(self.arrays)

        row_array = arrays[0]
        if row_array.ndim == 1:
            if item[0] != slice(None, None, None):
                row_array = vector_to_sparse(row_array, item[0], 0)
        else:
            row_array = row_array[item[0], :]
        arrays[0] = row_array

        if len(item) == 2:
            col_array = arrays[-1]
            if col_array.ndim == 1:
                if item[1] != slice(None, None, None):
                    col_array = vector_to_sparse(col_array, item[1], 1)
            else:
                col_array = col_array[:, item[1]]
            arrays[-1] = col_array

        return ChainedArray(arrays, tree_reduce=self.tree_reduce)

    def persist(self) -> "ChainedArray":
        return ChainedArray([a.persist() for a in self.arrays], tree_reduce=self.tree_reduce)

    def rechunk(self, chunks='auto', threshold=None, block_size_limit=None) -> "ChainedArray":
        arrays = []
        for array in self.arrays:
            if array.ndim == 1 or max(array.shape) == np.product(array.shape):
                arrays.append(array)
            else:
                arrays.append(array.rechunk(chunks, threshold, block_size_limit))
        return ChainedArray(arrays, tree_reduce=self.tree_reduce)

    def reshape(self) -> "ChainedArray":
        raise NotImplementedError
