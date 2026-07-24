import torch


class TorchBackend:
    """PyTorch backend for BraggCalculator.

    Provides the same API as NumpyBackend but backed by torch tensors,
    so operations are GPU-accelerated and autograd-compatible.
    """

    complex64 = torch.complex64
    complex128 = torch.complex128
    float32 = torch.float32
    float64 = torch.float64
    int64 = torch.int64
    bool = torch.bool
    is_torch = True

    def __init__(self, device: str = "cpu", dtype=torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype

    def asarray(self, x, dtype=None):
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=dtype) if dtype else x.to(self.device)
        return torch.tensor(x, device=self.device, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return torch.zeros(shape, device=self.device, dtype=dtype)

    def ones(self, shape, dtype=None):
        return torch.ones(shape, device=self.device, dtype=dtype)

    def exp(self, x):
        return torch.exp(x)

    def pi(self):
        return torch.pi

    def sin(self, x):
        return torch.sin(x)

    def cos(self, x):
        return torch.cos(x)

    def arcsin(self, x):
        return torch.arcsin(x)

    def sqrt(self, x):
        return torch.sqrt(x)

    def clip(self, x, a, b):
        return torch.clamp(x, a, b)

    def where(self, condition, x, y):
        return torch.where(condition, x, y)

    def round(self, x):
        return torch.round(x)

    def inverse(self, x):
        return torch.linalg.inv(x)

    def abs(self, x):
        return torch.abs(x)

    def real(self, x):
        return torch.real(x)

    def conj(self, x):
        return torch.conj(x)

    def einsum(self, s, *ops):
        return torch.einsum(s, *ops)

    def matmul(self, a, b):
        return torch.matmul(a, b)

    def linspace(self, a, b, n):
        return torch.linspace(a, b, steps=n, device=self.device, dtype=self.dtype)

    def concat(self, xs, axis=0):
        return torch.cat(xs, dim=axis)

    def sum(self, x, axis=None):
        return torch.sum(x, dim=axis)

    def max(self, x):
        return torch.max(x)

    def scatter_sum(self, values, indices, size):
        index = self.asarray(indices, dtype=torch.int64)
        result = torch.zeros(size, device=self.device, dtype=values.dtype)
        return result.scatter_add(0, index, values)

    def degrees(self, x):
        return torch.rad2deg(x)

    def q_from_two_theta(self, two_theta, wavelength):
        return 4.0 * torch.pi * torch.sin(two_theta / 2.0) / wavelength

    def two_theta_from_q(self, q, wavelength):
        theta = torch.arcsin(torch.clamp(q * wavelength / (4.0 * torch.pi), -1.0, 1.0))
        return 2.0 * theta
