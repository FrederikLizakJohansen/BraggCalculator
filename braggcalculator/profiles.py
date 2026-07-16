"""Area-normalized differentiable peak profiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt


FWHM_TO_SIGMA = 1.0 / (2.0 * sqrt(2.0 * log(2.0)))


def _render_gaussian(grid, centers, amplitudes, sigma, backend, max_entries):
    if sigma <= 0:
        raise ValueError("Gaussian FWHM must be positive")
    if max_entries <= 0:
        raise ValueError("max_entries must be positive")
    if int(centers.shape[0]) == 0:
        return backend.zeros((int(grid.shape[0]),), dtype=backend.dtype)

    peak_chunk = max(1, max_entries // max(int(grid.shape[0]), 1))
    result = backend.zeros((int(grid.shape[0]),), dtype=backend.dtype)
    normalization = 1.0 / (sigma * sqrt(2.0 * pi))
    for start in range(0, int(centers.shape[0]), peak_chunk):
        stop = min(start + peak_chunk, int(centers.shape[0]))
        x = grid[:, None] - centers[None, start:stop]
        contribution = amplitudes[None, start:stop] * backend.exp(-0.5 * (x / sigma) ** 2)
        result = result + normalization * backend.sum(contribution, axis=1)
    return result


@dataclass(frozen=True)
class GaussianProfile:
    """Gaussian in degrees with amplitudes interpreted as integrated areas."""

    fwhm_deg: float = 0.1
    max_entries: int = 4_194_304

    def render(self, grid, centers, amplitudes, backend):
        sigma = self.fwhm_deg * FWHM_TO_SIGMA
        return _render_gaussian(grid, centers, amplitudes, sigma, backend, self.max_entries)


@dataclass(frozen=True)
class GaussianProfileQ:
    """Gaussian in inverse angstroms with integrated-area amplitudes."""

    fwhm_q: float = 0.02
    max_entries: int = 4_194_304

    def render(self, grid_q, centers_q, amplitudes, backend):
        sigma = self.fwhm_q * FWHM_TO_SIGMA
        return _render_gaussian(grid_q, centers_q, amplitudes, sigma, backend, self.max_entries)
