import dask
from dask.utils import format_bytes
from multiprocessing.pool import ThreadPool
from typing import Union


def set_worker_number(num: int = 8):
    dask.config.set(pool=ThreadPool(num))


def set_chunk_size(num_bytes: Union[int, float] = 128e6):
    dask.config.set({'array.chunk_size': format_bytes(num_bytes)})
