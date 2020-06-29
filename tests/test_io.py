from lmdec.array.io import save_array, load_array_from_disk

import sparse
import shutil
import utils_io
import numpy as np


def test_create_save_load_array(num_tests: int = 100):

    file_name = 'test'
    dense_file_format = 'zarr'

    for _ in range(num_tests):
        array = utils_io.tree_array()
        save_array(array, file_name, dense_file_format=dense_file_format)

        new_array = load_array_from_disk(file_name)

        try:
            array = array.compute()
            new_array = new_array.compute()
        except ValueError:
            x = np.random.random(array.T.shape)
            np.testing.assert_array_equal(array.dot(x), new_array.dot(x))
        else:

            if type(array) == sparse._coo.core.COO:
                array = array.todense()

            if type(new_array) == sparse._coo.core.COO:
                new_array = new_array.todense()

            np.testing.assert_array_equal(array, new_array)

        shutil.rmtree(file_name)
