import numpy as np
import pytest

from braggcalculator.backends import NumpyBackend
from braggcalculator.profiles import GaussianProfile


def test_gaussian_amplitude_is_integrated_area():
    backend = NumpyBackend()
    grid = np.linspace(-2, 2, 200_001)
    values = GaussianProfile(fwhm_deg=0.1).render(grid, np.array([0.0]), np.array([3.5]), backend)
    assert np.trapezoid(values, grid) == pytest.approx(3.5, rel=1e-10)


def test_chunked_and_unchunked_rendering_are_identical():
    backend = NumpyBackend()
    grid = np.linspace(10, 80, 1000)
    centers = np.linspace(15, 75, 30)
    amplitudes = np.linspace(1, 2, 30)
    chunked = GaussianProfile(max_entries=2000).render(grid, centers, amplitudes, backend)
    whole = GaussianProfile(max_entries=1_000_000).render(grid, centers, amplitudes, backend)
    np.testing.assert_allclose(chunked, whole, rtol=1e-14, atol=1e-14)
