import numpy as np
import pytest

from braggcalculator.hkl import HKLEnumerator


def test_simple_cubic_reflection_geometry_is_analytical():
    result = HKLEnumerator(wavelength=1.0, qmax=2 * np.pi * 1.01).enumerate(
        np.diag([4.0, 4.0, 4.0])
    )
    index = np.flatnonzero(np.all(result["hkl"] == [1, 0, 0], axis=1))[0]
    assert result["g"][index] == pytest.approx(0.25)


def test_every_enumerated_reflection_satisfies_bragg_condition(nacl):
    wavelength = 1.5406
    result = HKLEnumerator(wavelength=wavelength, qmax=100.0).enumerate(nacl.lattice.matrix)
    assert len(result["hkl"]) > 0
    assert np.all(0.5 * wavelength * result["g"] < 1.0)
    assert np.all(result["two_theta"] < np.pi)


def test_no_fixed_hkl_limit_for_long_axis():
    lattice = np.diag([100.0, 3.0, 3.0])
    result = HKLEnumerator(wavelength=1.0, qmax=2.0 * np.pi).enumerate(lattice)
    assert np.max(np.abs(result["hkl"][:, 0])) >= 99
