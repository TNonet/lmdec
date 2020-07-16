import sparse

from dask import array as da


def issparse(x: da.core.Array) -> bool:
    """ Tests to see if the underlying array type of a Dask Array is sparse.COO

    Parameters
    ----------
    array : array_like

    Returns
    -------
    issparse : bool
        Flag whether array.compute() would be a sparse array

    Notes
    -----
    Roughly equivalent to type(array.compute()) == parse._coo.core.COO

    """
    try:
        return isinstance(x._meta, sparse._coo.core.COO)
    except ZeroDivisionError:
        return True
