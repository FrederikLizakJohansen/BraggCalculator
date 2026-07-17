"""Differentiable pseudo-Voigt profiles for experimental fitting."""

from __future__ import annotations

from math import log, pi, sqrt


def render_pseudo_voigt(
    grid,
    centers,
    amplitudes,
    fwhm,
    eta,
    backend,
    *,
    max_entries: int = 4_194_304,
):
    """Render area-normalized pseudo-Voigt peaks with per-peak widths."""
    if int(centers.shape[0]) == 0:
        return backend.zeros((int(grid.shape[0]),), dtype=backend.dtype)
    peak_chunk = max(1, max_entries // max(int(grid.shape[0]), 1))
    result = backend.zeros((int(grid.shape[0]),), dtype=backend.dtype)
    gaussian_factor = 2.0 * sqrt(2.0 * log(2.0))
    for start in range(0, int(centers.shape[0]), peak_chunk):
        stop = min(start + peak_chunk, int(centers.shape[0]))
        x = grid[:, None] - centers[None, start:stop]
        width = fwhm[start:stop][None, :]
        sigma = width / gaussian_factor
        gaussian = backend.exp(-0.5 * (x / sigma) ** 2) / (sigma * sqrt(2.0 * pi))
        half_width = 0.5 * width
        lorentzian = (half_width / pi) / (x**2 + half_width**2)
        shape = (1.0 - eta) * gaussian + eta * lorentzian
        result = result + backend.sum(amplitudes[None, start:stop] * shape, axis=1)
    return result


def caglioti_fwhm(two_theta_radians, u, v, w, backend):
    """Return FWHM in degrees from positive Caglioti U, V, W terms."""
    tangent = backend.sin(two_theta_radians / 2.0) / backend.cos(two_theta_radians / 2.0)
    variance = u * tangent**2 + v * tangent + w
    return backend.sqrt(variance)
