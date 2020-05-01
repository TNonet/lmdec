import numpy as np

from importlib_resources import files, as_file
import lmdec.examples


def load_np_file(file_name):
    source = files(lmdec.examples).joinpath(file_name)
    with as_file(source) as file:
        return np.load(file)


def get_ex1():
    """Loads array A for example 1 and its TruncatedSVD with top 10 components

    Uk, Sk, Vk = argmin || A - Uk*diag(Sk)*Vk||

    Over;
    Uk, Sk, Vk

    Where;
    Uk is a Orthonormal Matrix of size (20000, 10)
    Sk is a 10 dimensional non-negative vector
    Vk is a Orthonormal Matrix of size (10, 8000)

    Returns
    -------
    A : numpy.ndarray
        array of size (20000, 8000)
    Uk : numpy.ndarray
        orthonormal array of size (20000, 10)
        Top 10 Left Singular Vectors of `A`
    Sk : numpy.ndarray
        array of size (10, )
        Top 10 Singular Values of `A`
    Vk : numpy.ndarray
        transposed orthonormal array of size (10, 8000)
        Top 10 Right Singular Vectors of `A`
    """
    try:
        Uk = load_np_file('ex1_Uk.npy')
        Sk = load_np_file('ex1_Sk.npy')
        Vk = load_np_file('ex1_Vk.npy')
        ex1 = _make_a_ex1()
        return ex1, Uk, Sk, Vk
    except FileNotFoundError:
        raise FileNotFoundError("A, Uk, Sk, Vk cannot be loaded. Try make_ex1()")


def _make_a_ex1():
    n, p = 20000, 8000
    np.random.seed(10)
    a = np.random.randn(n, p) - 0.05 * np.random.rand(n, p)
    return a


def _make_ex1():
    k = 10
    a = _make_a_ex1()
    U, S, V = np.linalg.svd(a, full_matrices=False)
    Uk, Sk, Vk = U[:, :k], S[:k], V[:k, :]
    return a, Uk, Sk, Vk
