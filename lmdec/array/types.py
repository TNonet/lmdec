from typing import Union, TYPE_CHECKING
from numpy import ndarray
from dask.array.core import Array


if TYPE_CHECKING:
    from lmdec import ScaledArray

LargeArrayType = Union[ndarray, Array, "ScaledArray"]
ArrayType = Union[ndarray, Array]
DaskArrayType = Array
