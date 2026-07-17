"""Powder corrections and line-profile rendering."""

from __future__ import annotations

from typing import Literal


def lp_factor(two_theta, backend, mode: Literal["xray", "neutron"]):
    """Return the standard unpolarized powder Lorentz-polarization factor.

    The X-ray expression matches the Bragg-Brentano powder correction used by
    pymatgen.  Neutrons have no polarization numerator.
    """
    if mode not in {"xray", "neutron"}:
        raise ValueError("mode must be 'xray' or 'neutron'")
    theta = two_theta / 2.0
    denominator = backend.sin(theta) ** 2 * backend.cos(theta)
    if mode == "xray":
        return (1.0 + backend.cos(two_theta) ** 2) / denominator
    return 1.0 / denominator


def apply_lp_and_multiplicity(mode, backend, F2, two_theta, multiplicity=None):
    intensity = F2 * lp_factor(two_theta, backend, mode)
    if multiplicity is not None:
        intensity = intensity * backend.asarray(multiplicity, dtype=backend.dtype)
    return intensity


def render_profile(profile, backend, grid, centers, amplitudes, *, fwhm=None):
    arguments = (
        grid,
        backend.asarray(centers, dtype=backend.dtype),
        backend.asarray(amplitudes, dtype=backend.dtype),
        backend,
    )
    return profile.render(*arguments) if fwhm is None else profile.render(*arguments, fwhm=fwhm)


def render_profile_q(profile_q, backend, grid_q, centers_q, amplitudes, *, fwhm=None):
    arguments = (
        grid_q,
        backend.asarray(centers_q, dtype=backend.dtype),
        backend.asarray(amplitudes, dtype=backend.dtype),
        backend,
    )
    return profile_q.render(*arguments) if fwhm is None else profile_q.render(
        *arguments, fwhm=fwhm
    )
