"""Tabulated X-ray and neutron scattering amplitudes.

The numerical tables are maintained by pymatgen, a required dependency.  The
X-ray expression is the Doyle-Turner/International Tables parameterization used by
``pymatgen.analysis.diffraction.xrd``; neutron values are coherent bound
scattering lengths in femtometres from the same package.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import numpy as np
from pymatgen.analysis.diffraction.neutron import ATOMIC_SCATTERING_LEN
from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS, WAVELENGTHS
from pymatgen.core import Element


_XRAY_FORM_FACTOR_SCALE = 41.78214


def resolve_wavelength(value: float | str) -> float:
    """Return a wavelength in angstroms from a number or radiation name."""
    if isinstance(value, str):
        try:
            return float(WAVELENGTHS[value])
        except KeyError as exc:
            choices = ", ".join(sorted(WAVELENGTHS))
            raise ValueError(f"Unknown radiation {value!r}; choose one of {choices}") from exc
    wavelength = float(value)
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be a positive finite value in angstroms")
    return wavelength


def _numpy_atomic_numbers(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64)


@lru_cache(maxsize=None)
def _xray_coefficients(numbers: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    coeffs = []
    for z in numbers:
        symbol = Element.from_Z(z).symbol
        try:
            coeffs.append(ATOMIC_SCATTERING_PARAMS[symbol])
        except KeyError as exc:
            raise ValueError(f"No X-ray scattering coefficients for {symbol}") from exc
    return np.asarray(numbers, dtype=float), np.asarray(coeffs, dtype=float)


def xray_form_factors(Z, s, backend) -> Any:
    """Evaluate neutral-atom X-ray form factors.

    Args:
        Z: Atomic numbers, shape ``(N,)``.
        s: ``sin(theta) / wavelength`` in inverse angstroms, shape ``(H,)``.
        backend: NumPy or Torch backend.

    Returns:
        Form factors with shape ``(H, N)``.
    """
    numbers_array = _numpy_atomic_numbers(Z)
    unique_numbers, inverse = np.unique(numbers_array, return_inverse=True)
    numbers = tuple(int(z) for z in unique_numbers.tolist())
    zs_np, coeff_np = _xray_coefficients(numbers)
    dtype = getattr(s, "dtype", backend.dtype)
    zs = backend.asarray(zs_np, dtype=dtype)
    coeff = backend.asarray(coeff_np, dtype=dtype)
    s2 = s * s
    a = coeff[:, :, 0]
    b = coeff[:, :, 1]
    terms = a[None, :, :] * backend.exp(-s2[:, None, None] * b[None, :, :])
    unique_factors = zs[None, :] - _XRAY_FORM_FACTOR_SCALE * s2[:, None] * backend.sum(
        terms, axis=2
    )
    backend_inverse = backend.asarray(inverse, dtype=backend.int64)
    return unique_factors[:, backend_inverse]


@lru_cache(maxsize=None)
def _neutron_lengths(numbers: tuple[int, ...]) -> np.ndarray:
    values = []
    for z in numbers:
        symbol = Element.from_Z(z).symbol
        try:
            values.append(ATOMIC_SCATTERING_LEN[symbol])
        except KeyError as exc:
            raise ValueError(f"No coherent neutron scattering length for {symbol}") from exc
    return np.asarray(values, dtype=float)


def neutron_b_coherent(
    Z,
    backend,
    overrides: Mapping[str | int, float | str] | None = None,
) -> Any:
    """Return coherent neutron scattering lengths in femtometres.

    ``overrides`` supports element symbols or atomic numbers. Values can be an
    exact length or a tabulated isotope key such as ``"2H"``. This is the
    explicit route for isotope-specific samples, which pymatgen ``Structure``
    objects do not encode unambiguously.
    """
    numbers = tuple(int(z) for z in _numpy_atomic_numbers(Z).tolist())
    values = _neutron_lengths(numbers).copy()
    if overrides:
        for idx, z in enumerate(numbers):
            symbol = Element.from_Z(z).symbol
            override = overrides.get(symbol, overrides.get(z))
            if override is not None:
                if isinstance(override, str):
                    try:
                        values[idx] = ATOMIC_SCATTERING_LEN[override]
                    except KeyError as exc:
                        raise ValueError(
                            f"No coherent neutron scattering length for isotope {override!r}"
                        ) from exc
                else:
                    values[idx] = float(override)
    return backend.asarray(values, dtype=backend.dtype)
