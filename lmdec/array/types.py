from typing import Union, TYPE_CHECKING
from numpy import ndarray
from dask.array.core import Array


if TYPE_CHECKING:
    from lmdec.array.chained import ChainedArray
    from lmdec.array.stacked import StackedArray

LargeArrayType = Union[ndarray, Array, "ChainedArray", "StackedArray"]
ArrayType = Union[ndarray, Array]
DaskArrayType = Array
