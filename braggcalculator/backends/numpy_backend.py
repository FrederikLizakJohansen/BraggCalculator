import numpy as np


class NumpyBackend:
    complex64 = np.complex64
    float32 = np.float32

    def asarray(self, x, dtype=None):
        return np.asarray(x, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype)

    def exp(self, x):
        return np.exp(x)

    def pi(self):
        return np.pi

    def sin(self, x):
        return np.sin(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def einsum(self, s, *ops):
        return np.einsum(s, *ops)

    def linspace(self, a, b, n):
        return np.linspace(a, b, n)

    def conj(self, x):
        return np.conj(x)

    def real(self, x):
        return np.real(x)

    def degrees(self, x):
        return np.degrees(x)

    def q_from_two_theta(self, two_theta, wavelength):
        # TODO: degrees vs. radians
        return 4.0 * np.pi * np.sin(two_theta / 2.0) / wavelength
    
    def two_theta_from_q(self, q, wavelength):
        # q = 4 pi sin(theta) / lambda -> theta = arcsin(q lambda / 4 pi)
        # TODO: degrees vs. radians
        theta = np.arcsin(np.clip(q * wavelength / (4.0 * np.pi), -1.0, 1.0))
        return 2.0 * theta
