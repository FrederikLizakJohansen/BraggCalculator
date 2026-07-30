"""Reusable characteristic-radiation spectrum definitions."""

from __future__ import annotations


def nist_copper_ka_spectrum() -> tuple[dict[str, float], ...]:
    """Return the six-line Cu K-alpha spectrum used by the NIST FPA code.

    Wavelengths and Lorentzian FWHM values are in angstroms. Weights are
    relative integrated intensities and are normalized by ``RefinementSession``.
    """
    wavelengths = (1.5405925, 1.5443873, 1.5446782, 1.5410769, 1.53471, 1.53382)
    intensities = (0.58384351, 0.2284605, 0.11258773, 0.07077796, 0.0043303, 0.00208613)
    widths = (0.000436, 0.000487, 0.000630, 0.000558, 0.00293, 0.00293)
    return tuple(
        {
            "wavelength_angstrom": wavelength,
            "weight": intensity,
            "lorentzian_fwhm_angstrom": width,
        }
        for wavelength, intensity, width in zip(wavelengths, intensities, widths)
    )
