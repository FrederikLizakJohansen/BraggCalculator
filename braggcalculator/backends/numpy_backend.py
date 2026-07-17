import numpy as np


class NumpyBackend:
    """Small numerical interface shared by the diffraction kernels.

    Double precision is intentional.  Systematic absences are cancellations of
    complex-valued sums and single precision produces visible ghost peaks for
    larger cells.
    """

    complex64 = np.complex64
    complex128 = np.complex128
    float32 = np.float32
    float64 = np.float64
    int64 = np.int64
    bool = np.bool_
    is_torch = False

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def asarray(self, x, dtype=None):
        return np.asarray(x, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype)

    def exp(self, x):
        return np.exp(x)

    def log(self, x):
        return np.log(x)

    def softplus(self, x):
        return np.logaddexp(0.0, x)

    def softmax(self, x, axis=-1):
        shifted = x - np.max(x, axis=axis, keepdims=True)
        weights = np.exp(shifted)
        return weights / np.sum(weights, axis=axis, keepdims=True)

    def pi(self):
        return np.pi

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sinh(self, x):
        return np.sinh(x)

    def arcsin(self, x):
        return np.arcsin(x)

    def arccos(self, x):
        return np.arccos(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def clip(self, x, a, b):
        return np.clip(x, a, b)

    def where(self, condition, left, right):
        return np.where(condition, left, right)

    def inverse(self, x):
        return np.linalg.inv(x)

    def matrix_exp(self, x):
        """Matrix exponential for a real symmetric matrix."""
        eigenvalues, eigenvectors = np.linalg.eigh(x)
        return (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T

    def einsum(self, s, *ops):
        return np.einsum(s, *ops)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def linspace(self, a, b, n):
        return np.linspace(a, b, n, dtype=self.dtype)

    def concat(self, xs, axis=0):
        return np.concatenate(xs, axis=axis)

    def stack(self, xs, axis=0):
        return np.stack(xs, axis=axis)

    def conj(self, x):
        return np.conj(x)

    def real(self, x):
        return np.real(x)

    def sum(self, x, axis=None):
        return np.sum(x, axis=axis)

    def max(self, x):
        return np.max(x)

    def scatter_sum(self, values, indices, size):
        result = np.zeros(size, dtype=values.dtype)
        np.add.at(result, np.asarray(indices, dtype=np.int64), values)
        return result

    def degrees(self, x):
        return np.degrees(x)

    def q_from_two_theta(self, two_theta, wavelength):
        return 4.0 * np.pi * np.sin(two_theta / 2.0) / wavelength

    def two_theta_from_q(self, q, wavelength):
        theta = np.arcsin(np.clip(q * wavelength / (4.0 * np.pi), -1.0, 1.0))
        return 2.0 * theta
