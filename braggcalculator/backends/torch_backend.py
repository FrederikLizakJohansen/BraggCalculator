import torch


class TorchBackend:
    """PyTorch backend for BraggCalculator.

    Provides the same API as NumpyBackend but backed by torch tensors,
    so operations are GPU-accelerated and autograd-compatible.
    """

    complex64 = torch.complex64
    float32 = torch.float32

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def asarray(self, x, dtype=None):
        # Convert input to torch tensor on this backend's device
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

    def sqrt(self, x):
        return torch.sqrt(x)

    def abs(self, x):
        return torch.abs(x)

    def real(self, x):
        return torch.real(x)

    def conj(self, x):
        return torch.conj(x)

    def einsum(self, s, *ops):
        return torch.einsum(s, *ops)

    def linspace(self, a, b, n):
        return torch.linspace(a, b, steps=n, device=self.device)

    def concat(self, xs, axis=0):
        return torch.cat(xs, dim=axis)

    def degrees(self, x):
        return torch.rad2deg(x)

    def q_from_two_theta(self, two_theta, wavelength):
        return 4.0 * torch.pi * torch.sin(two_theta / 2.0) / wavelength

    def two_theta_from_q(self, q, wavelength):
        theta = torch.arcsin(torch.clamp(q * wavelength / (4.0 * torch.pi), -1.0, 1.0))
        return 2.0 * theta
