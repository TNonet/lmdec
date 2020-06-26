from hypothesis.extra import numpy as npst
from hypothesis.strategies import composite, integers, booleans, lists, just

import numpy as np


@composite
def get_boardcastable_arrays_shapes(draw, base_min_dims: int = 2, base_max_dims: int = 3, broadcast_min_dims: int = 1,
                                    broadcast_max_dims: int = 5):
    """ Returns a tuple of 'broadcastable valid numpy/dask array shape' tuples.

    In other words, each item in the tuple will be a tuple that is a numpy/dask array shape that can be broadcasted with
    the other shapes in the outer tuple.

    Parameters
    ----------
    draw : draw object required for hypothesis
        see https://hypothesis.readthedocs.io/en/latest/data.html#composite-strategies
    min_dims : int
    max_dims : int

    Returns
    -------
    shapes: tuple of tuples of (integers, optional)

    """
    base_shape = draw(npst.array_shapes(min_dims=base_min_dims, max_dims=base_max_dims))
    array_shapes = draw(npst.mutually_broadcastable_shapes(base_shape=base_shape, num_shapes=3,
                                                           min_dims=broadcast_min_dims, max_dims=broadcast_max_dims))
    return (base_shape, *[shape for shape in array_shapes.input_shapes])


@composite
def get_boardcastable_arrays_shapes_and_indices(draw, base_min_dims: int = 2, base_max_dims: int = 3,
                                                broadcast_min_dims: int = 1, broadcast_max_dims: int = 5):
    shape_strategy = get_boardcastable_arrays_shapes(base_min_dims=base_min_dims,  base_max_dims=base_max_dims,
                                                     broadcast_min_dims=broadcast_min_dims,
                                                     broadcast_max_dims=broadcast_max_dims)

    shapes = draw(shape_strategy)
    # What is final shape?
    ndim = max(len(shape) for shape in shapes)
    padded_shapes = np.array([[1] * (ndim - len(shape)) + list(shape) for shape in shapes])
    shape = tuple(np.max(padded_shapes, axis=0).astype(int))
    # TODO: raise Issue with hypothesis. Numpy types aren't allowed. uncomment next line and it fails
    shape = tuple(int(s) for s in shape)
    indices = draw(npst.basic_indices(shape, allow_ellipsis=False, min_dims=2))
    return shapes, indices, shape


@composite
def get_chainable_array_shapes(draw, base_min_dims: int = 1, base_max_dims: int = 2, max_chain: int = 8,
                               chain_min_dims: int = 1):
    base_shape = draw(npst.array_shapes(min_dims=base_min_dims, max_dims=base_max_dims, min_side=2))
    if chain_min_dims == 1:
        dimensions = draw(lists(booleans(), min_size=2, max_size=max_chain))
    elif chain_min_dims == 2:
        dimensions = draw(lists(just(False), min_size=2, max_size=max_chain))
    else:
        raise ValueError

    shapes = [base_shape]
    col = base_shape[-1]
    for d in dimensions:
        first_value = col
        if not d:
            shapes.append((first_value, draw((integers(min_value=2, max_value=10)))))
        else:
            shapes.append((first_value, ))
        col = shapes[-1][-1]

    return shapes


@composite
def get_chainable_arrays_shapes_and_indices(draw, base_min_dims: int = 1, base_max_dims: int = 2, max_chain: int = 8,
                                            chain_min_dims: int = 1):
    shape_strategy = get_chainable_array_shapes(base_min_dims=base_min_dims, base_max_dims=base_max_dims,
                                                max_chain=max_chain, chain_min_dims=chain_min_dims)
    shapes = draw(shape_strategy)
    shape = (shapes[0][0], shapes[-1][-1])
    indices = draw(npst.basic_indices(shape, allow_ellipsis=False, min_dims=2))

    return shapes, indices, shape


@composite
def get_vector_index_axis(draw, max_shape: int = 10):
    N = draw(integers(min_value=2, max_value=max_shape))
    index = draw(npst.basic_indices(shape=(N,), min_dims=1, max_dims=1, allow_ellipsis=False))
    axis = draw(integers(min_value=0, max_value=1))
    return (N, *index, axis)

