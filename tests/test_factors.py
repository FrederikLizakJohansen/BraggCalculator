import numpy as np
import pytest

from braggcalculator.backends import NumpyBackend
from braggcalculator.factors import (
    neutron_b_coherent,
    resolve_wavelength,
    xray_form_factors,
)


def test_named_wavelengths_and_validation():
    assert resolve_wavelength("CuKa1") == pytest.approx(1.54056)
    assert resolve_wavelength(0.71073) == pytest.approx(0.71073)
    with pytest.raises(ValueError):
        resolve_wavelength("not-radiation")
    with pytest.raises(ValueError):
        resolve_wavelength(0)


def test_xray_form_factor_is_z_at_zero_scattering_vector():
    backend = NumpyBackend()
    result = xray_form_factors(np.array([1, 8, 26]), np.array([0.0]), backend)
    np.testing.assert_allclose(result, [[1.0, 8.0, 26.0]], rtol=0, atol=0)


def test_exact_neutron_lengths_and_override():
    backend = NumpyBackend()
    result = neutron_b_coherent(np.array([1, 8]), backend)
    np.testing.assert_allclose(result, [-3.739, 5.803])
    overridden = neutron_b_coherent(np.array([1, 8]), backend, {"H": "2H"})
    np.testing.assert_allclose(overridden, [6.671, 5.803])
