import numpy as np
import pytest

from braggcalculator import BraggCalculator


def test_calculation_requires_loaded_structure():
    with pytest.raises(RuntimeError, match="load a structure"):
        BraggCalculator().fq()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "electron"},
        {"wavelength": -1},
        {"two_theta_range": (80, 10)},
        {"two_theta_range": (-1, 80)},
        {"q_range": (1, 1)},
        {"q_step": 0},
        {"qmax": 1.0},
        {"intensity_tolerance": -1},
        {"symprec": 0},
        {"phase_chunk_entries": 1.5},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BraggCalculator(**kwargs)


def test_q_and_two_theta_positions_are_consistent(nacl):
    wavelength = 1.5406
    q_bounds = tuple(
        4 * np.pi * np.sin(np.radians(angle) / 2) / wavelength for angle in (10.0, 80.0)
    )
    calculator = BraggCalculator(wavelength=wavelength, q_range=q_bounds).load(nacl)
    two_theta, _ = calculator.line_pattern("two_theta")
    q, _ = calculator.line_pattern("q")
    expected_q = 4 * np.pi * np.sin(np.radians(two_theta) / 2) / calculator.wavelength
    np.testing.assert_allclose(q, expected_q, rtol=1e-13, atol=1e-13)


def test_parameter_shapes_are_checked(nacl):
    calculator = BraggCalculator().load(nacl)
    with pytest.raises(ValueError, match="frac_coords"):
        calculator.fq(parameters={"frac_coords": np.zeros((1, 3))})


def test_lattice_change_outside_prepared_bragg_sphere_is_rejected(nacl):
    calculator = BraggCalculator().load(nacl)
    with pytest.raises(ValueError, match="rebuild the calculator"):
        calculator.fq(parameters={"lattice": calculator._symm["lattice"] * 0.5})


def test_grid_step_is_not_changed_to_force_upper_endpoint(nacl):
    calculator = BraggCalculator(two_theta_range=(10.0, 10.1), two_theta_step=0.03).load(nacl)
    grid, _ = calculator.pattern()
    np.testing.assert_allclose(np.diff(grid), 0.03, rtol=0, atol=1e-14)
    assert grid[-1] == pytest.approx(10.09)


def test_automatic_qmax_covers_short_wavelength_two_theta_range(nacl):
    calculator = BraggCalculator(wavelength="AgKa1").load(nacl)
    assert calculator.qmax >= 4 * np.pi * np.sin(np.radians(80) / 2) / calculator.wavelength
    positions, _ = calculator.line_pattern()
    assert positions[-1] > 75
