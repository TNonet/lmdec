import os
import warnings
from pathlib import Path
from typing import Optional, Union, Tuple, List, Iterable

import numpy as np
import sparse
from dask import array as da
from pandas_plink import read_plink, read_plink1_bin

from lmdec.array.chained import ChainedArray
from lmdec.array.stacked import StackedArray
from lmdec.array.utils import issparse


def load_plink_array(path_to_plink_files: Optional[Union[str, Path]] = None,
                     bed: Optional[Union[str, Path]] = None, bim: Optional[Union[str, Path]] = None,
                     fam: Optional[Union[str, Path]] = None, transpose: bool = False) -> da.core.Array:
    """Gathers plink array from possible formats

    Requires one of the following parameter configures to be satisfied to load an array:
        - array is specified and no other parameters are specified
        - path_to_files is specified and no other parameters are specified
        - bim AND fam AND bed are specified and no other parameters are specified

    Parameters
    ----------
    path_to_plink_files : path_like, optional
        Assuming bim, fam, bed files are in the following format
            </path/to/data.bim>
            </path/to/data.fam>
            </path/to/data.bed>
        Then, path_to_files would be '/path/to/data'
    bed : path_like, optional
        '/path/to/data.bed'
    bim : path_like, optional
        '/path/to/data.bim'
    fam : path_like, optional
        '/path/to/data.fam'
    transpose : bool
        Whether `array` is stored/loaded in transposed format

        If A is stored/loaded as A.T but SVD(A) is desired set transpose flag to True

    Returns
    -------
    array : dask.array.core.Array

    """
    if path_to_plink_files is not None and not all([bed, bim, fam]):
        (_, _, G) = read_plink(path_to_plink_files)
        array = G
    elif all(p is not None for p in [bed, bim, fam]) and not path_to_plink_files:
        G = read_plink1_bin(bed, bim, fam)
        array = G.data
    else:
        raise ValueError('Uninterpretable input.'
                         ' Please specify array, xor path_to_files, xor (bed and bim and fim)')

    try:
        array = da.from_array(array)
    except AttributeError:
        raise ValueError('Uninterpretable array.')
    except ValueError:
        pass

    if len(array.shape) != 2:
        raise ValueError("Must be a 2-D array")

    if transpose:
        array = array.T

    return array


def save_and_load_array(array, file: Optional[str] = None):
    pass


def sort_files_by_int(x: Iterable[str]) -> List[str]:
    def index(s: str):
        return int(s.split('_')[0])

    return list(sorted(x, key=index))


def walk_one_level(top: Path, ignore_hidden: bool = True) -> Tuple[List[str], List[str]]:
    file_list = []
    dir_list = []
    for (_, dirs, files) in os.walk(str(top)):
        file_list.extend(files)
        dir_list.extend(dirs)
        break

    if ignore_hidden:
        # Removes .DS_STORE or other hidden files
        file_list = [f for f in file_list if not f.startswith('.')]

    return dir_list, file_list


def load_sprase_array(file: Path, **kwargs):
    _, files = walk_one_level(file)

    coords = [file for file in files if file.startswith('coords')][0]
    shape = [file for file in files if file.startswith('shape')][0]
    data = [file for file in files if file.startswith('data')][0]

    shape = load_array_from_disk(Path(file, shape), **kwargs)

    if shape.shape == ():
        shape = (int(shape), )
    else:
        shape = tuple(int(i) for i in shape)

    if coords.endswith('.txt'):
        coords = load_dense_array(Path(file, coords), ndmin=len(shape))
    else:
        coords = load_dense_array(Path(file, coords), **kwargs)

    array = sparse.COO(coords=coords,
                       data=load_dense_array(Path(file, data), **kwargs),
                       shape=shape,
                       has_duplicates=False, cache=True)

    return da.from_array(array)


def load_stacked_array(file: Path, **kwargs):
    dirs, files = walk_one_level(file)
    sub_arrays = dirs + files
    sub_arrays = sort_files_by_int(sub_arrays)
    return StackedArray([load_array_from_disk(Path(file, sub_array), **kwargs) for sub_array in sub_arrays])


def load_chained_array(file: Path, **kwargs):
    dirs, files = walk_one_level(file)
    sub_arrays = dirs + files
    sub_arrays = sort_files_by_int(sub_arrays)
    return ChainedArray([load_array_from_disk(Path(file, sub_array), **kwargs) for sub_array in sub_arrays])


def load_dense_array(file: Path, **kwargs):
    if file.suffix == '.zarr':
        return da.from_zarr(str(file), **kwargs)
    elif file.suffix == '.hdf5':
        raise NotImplementedError
    elif file.suffix == '.txt':
        return np.loadtxt(str(file), **kwargs)
    elif file.suffix == '.npy':
        return np.load(str(file), **kwargs)
    else:
        raise ValueError(f'{file} does not reference an load-able array')


def load_array_from_disk(path_to_array: Union[str, Path], **kwargs):
    path_to_array = Path(path_to_array)
    if os.path.isdir(path_to_array):
        if str(path_to_array).endswith('sparse'):
            return load_sprase_array(path_to_array, **kwargs)
        elif str(path_to_array).endswith('stacked'):
            return load_stacked_array(path_to_array, **kwargs)
        elif str(path_to_array).endswith('chained'):
            return load_chained_array(path_to_array, **kwargs)
        elif str(path_to_array).endswith('npy_stack'):
            raise NotImplementedError('npy_stack is not implemented.')
        elif str(path_to_array).endswith('.zarr'):
            return load_dense_array(path_to_array, **kwargs)
        else:
            dirs, files = walk_one_level(path_to_array)
            if len(dirs) == 1:
                return load_array_from_disk(Path(path_to_array, dirs[0]), **kwargs)
    else:
        return load_dense_array(path_to_array, **kwargs)


def save_array(array: da.core.Array, file: Union[str, Path], dense_file_format: str = 'zarr',
               sparse_file_format: str = 'txt') -> None:
    """ Save a Stacked, Chained, or Dask Array, with a tree file format, if necssary.

    Parameters
    ----------
    array : array_like
        array to be saved
    file : str
        path or file name to save the array as
    dense_file_format : str
        format that all dense arrays will be saved with

        Options are {'zarr', 'tiledb', 'h5py', 'npy', 'npy_stack', 'txt'}
    sparse_file_format : str
        format that all sparse arrays will be saved with

        Options are the same as 'dense_file_format'
    Returns
    -------
    None

    Notes
    -----
    Base Level: Save 'file/' base directory

    Array types and how they are saved
    da.core.Array
        dense:
            |file/
            |   file.<file_format>
        sparse:
            |file/
            |   sparse/
            |       coords.<file_format>
            |       data.<file_format>
            |       shape.<file_format>
    StackedArray
        |file/
        |   stacked/
        |       array1.<file_format>
        |       array2.<file_format>
        |       ...

    ChainedArray
        |file/
        |   chained/
        |       array1.<file_format>
        |       array2.<file_format>
        |       ...

    """
    if os.path.exists(file):
        raise ValueError(f'file, {file} already exists')

    if type(array) == da.core.Array:
        if issparse(array):
            if not file.endswith('sparse'):
                os.mkdir(file)
                file = "/".join([file, 'sparse'])
            array = array.compute()
            save_sparse_array(array, path=file, file_format=sparse_file_format)
        else:
            if not file.endswith('array'):
                os.mkdir(file)
                file = "/".join([file, 'array'])
            save_dense_array(array, file=file, file_format=dense_file_format)
    else:
        os.mkdir(file)
        if not (file.endswith('stacked') or file.endswith('chained')):
            """
            The following format does not fit into any other 
            |file/
            |   {stacked, chained, sparse}/
            Therefore we must capture it here and drop one directory to continue the recursive search
            """
            array_type = 'stacked' if type(array) == StackedArray else 'chained'
            sub_file = '/'.join([file, array_type])
            os.mkdir(sub_file)
            file = sub_file
        for i, array in enumerate(array.arrays):
            if type(array) == da.core.Array:
                if issparse(array):
                    save_array(array, file=f'{file}/{i}_sparse',
                               dense_file_format=dense_file_format, sparse_file_format=sparse_file_format)
                else:
                    save_array(array, file=f'{file}/{i}_array',
                               dense_file_format=dense_file_format, sparse_file_format=sparse_file_format)
            elif isinstance(array, StackedArray):
                save_array(array, file=f"{file}/{i}_stacked",
                           dense_file_format=dense_file_format, sparse_file_format=sparse_file_format)
            elif isinstance(array, ChainedArray):
                save_array(array, file=f"{file}/{i}_chained",
                           dense_file_format=dense_file_format, sparse_file_format=sparse_file_format)
            else:
                raise ValueError(f'expected array of type ({type(da.core.Array), type(ChainedArray), type(ChainedArray)}'
                                 f', but got type {type(array)}')


def save_sparse_array(array: sparse._coo.core.COO, path: Path, file_format: str = 'npy'):
    """
    Notes
    -----
    Sparse File Directory:
        |sparse/
        |   coords.<file_format> # file/sparse/coords.<file_format>
        |   data.<file_format>
        |   shape.<file_format>
    """
    formats = ['npy', 'txt']
    if file_format not in formats:
        raise ValueError(f'expected file_format in {formats}, but got {file_format}')

    os.mkdir(path)

    save_dense_array(array.coords, file=path + '/coords', file_format=file_format)
    save_dense_array(array.data, file=path + '/data', file_format=file_format)
    save_dense_array(array.shape, file=path + '/shape', file_format=file_format)


def save_dense_array(array: Union[da.core.Array, np.ndarray, Iterable], file: Path, file_format: str = 'zarr',
                     **kwargs):
    file_formats = ['zarr', 'hdf5', 'npy', 'tiledb', 'txt', 'npy_stack']

    if file_format not in file_formats:
        raise ValueError(f'file_format, {file_format}, not implemented.')

    if file_format == 'zarr':
        if not isinstance(array, (da.core.Array, np.ndarray)):
            array = da.array(array)
        if not file.endswith('.zarr'):
            file += '.zarr'
        array.to_zarr(file, **kwargs)
    elif file_format == 'hdf5':
        raise NotImplementedError
    elif file_format == 'npy_stack':
        raise NotImplementedError
    elif file_format == 'npy':
        if isinstance(array, da.core.Array):
            if array.nbytes > 1e9:
                warnings.warn(f'Dask array of size {array.nbytes} bytes saved to disk as numpy array.'
                              f' Will cause performance issues')
            array = array.compute()

        if not file.endswith('.npy'):
            file += '.npy'
        np.save(file, array, **kwargs)

    elif file_format == 'txt':
        if isinstance(array, da.core.Array):
            if isinstance(array, da.core.Array):
                if array.nbytes > 1e9:
                    warnings.warn(f'Dask array of size {array.nbytes} bytes saved to disk as numpy array.'
                                  f' Will cause performance issues')
                array = array.compute()
        if not file.endswith('.txt'):
            file += '.txt'
        np.savetxt(file, array, **kwargs)
    else:
        raise NotImplementedError
