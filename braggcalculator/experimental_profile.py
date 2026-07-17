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


def thompson_cox_hastings(
    two_theta_radians,
    u,
    v,
    w,
    x,
    y,
    backend,
    *,
    extra_lorentzian=0.0,
):
    """Return TCH pseudo-Voigt FWHM and mixing for Gaussian/Lorentzian terms.

    ``u``, ``v`` and ``w`` define the squared Gaussian FWHM in degrees, while
    ``x`` and ``y`` define the Lorentzian FWHM in degrees. The returned mixing
    is clipped to the physical interval to remain robust during optimization.
    """
    theta = two_theta_radians / 2.0
    tangent = backend.sin(theta) / backend.cos(theta)
    gaussian_squared = backend.clip(u * tangent**2 + v * tangent + w, 1e-12, None)
    gaussian = backend.sqrt(gaussian_squared)
    lorentzian = backend.clip(
        x / backend.cos(theta) + y * tangent + extra_lorentzian, 1e-12, None
    )
    combined_fifth = (
        gaussian**5
        + 2.69269 * gaussian**4 * lorentzian
        + 2.42843 * gaussian**3 * lorentzian**2
        + 4.47163 * gaussian**2 * lorentzian**3
        + 0.07842 * gaussian * lorentzian**4
        + lorentzian**5
    )
    fwhm = combined_fifth ** 0.2
    ratio = lorentzian / fwhm
    eta = backend.clip(
        1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3,
        0.0,
        1.0,
    )
    return fwhm, eta


def emission_lorentzian_fwhm(
    two_theta_radians,
    wavelength_angstrom,
    line_fwhm_angstrom,
    backend,
):
    """Map a Lorentzian emission-line width from wavelength to degrees 2-theta."""
    theta = two_theta_radians / 2.0
    tangent = backend.sin(theta) / backend.cos(theta)
    width_radians = 2.0 * tangent * line_fwhm_angstrom / wavelength_angstrom
    return backend.degrees(width_radians)


def axial_divergence_widths(fwhm, two_theta_radians, asymmetry, backend):
    """Return empirical low-/high-angle widths for an axial-divergence tail.

    This differentiable split-width approximation is not the full
    Finger--Cox--Jephcoat convolution. ``asymmetry`` is dimensionless and a
    value of zero gives an exactly symmetric profile. The low-angle broadening
    scales with cot(theta), matching the dominant angular trend of axial
    divergence while keeping a compact refinement model.
    """
    theta = two_theta_radians / 2.0
    tangent = backend.clip(backend.sin(theta) / backend.cos(theta), 1e-8, None)
    low_angle = fwhm * (1.0 + asymmetry / tangent)
    return low_angle, fwhm


def specimen_displacement_shift(
    two_theta_radians,
    displacement_mm,
    goniometer_radius_mm,
    backend,
):
    """Bragg--Brentano specimen-displacement shift in degrees 2-theta."""
    if goniometer_radius_mm <= 0:
        raise ValueError("goniometer_radius_mm must be positive")
    theta = two_theta_radians / 2.0
    shift_radians = -2.0 * displacement_mm * backend.cos(theta) / goniometer_radius_mm
    return backend.degrees(shift_radians)


def render_split_pseudo_voigt(
    grid,
    centers,
    amplitudes,
    low_fwhm,
    high_fwhm,
    eta,
    backend,
    *,
    max_entries: int = 4_194_304,
):
    """Render an area-normalized split pseudo-Voigt for each reflection."""
    if int(centers.shape[0]) == 0:
        return backend.zeros((int(grid.shape[0]),), dtype=backend.dtype)
    peak_chunk = max(1, max_entries // max(int(grid.shape[0]), 1))
    result = backend.zeros((int(grid.shape[0]),), dtype=backend.dtype)
    gaussian_exponent = 4.0 * log(2.0)
    gaussian_integral = sqrt(pi) / (4.0 * sqrt(log(2.0)))
    for start in range(0, int(centers.shape[0]), peak_chunk):
        stop = min(start + peak_chunk, int(centers.shape[0]))
        offset = grid[:, None] - centers[None, start:stop]
        low = low_fwhm[start:stop][None, :]
        high = high_fwhm[start:stop][None, :]
        width = backend.where(offset < 0.0, low, high)
        gaussian = backend.exp(-gaussian_exponent * (offset / width) ** 2)
        gaussian = gaussian / (gaussian_integral * (low + high))
        lorentzian = 1.0 / (1.0 + 4.0 * (offset / width) ** 2)
        lorentzian = lorentzian / (0.25 * pi * (low + high))
        mixing = eta[start:stop][None, :] if getattr(eta, "shape", ()) else eta
        shape = (1.0 - mixing) * gaussian + mixing * lorentzian
        result = result + backend.sum(amplitudes[None, start:stop] * shape, axis=1)
    return result
