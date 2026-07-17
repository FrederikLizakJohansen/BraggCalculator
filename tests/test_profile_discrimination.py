import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator
from braggcalculator.diagnostics import compare_profile_counts, profile_discrimination


def test_diagonal_profile_discrimination_matches_expected_delta_chi_squared():
    coordinate = np.array([1.0, 2.0, 3.0])
    expected_a = np.array([10.0, 20.0, 40.0])
    expected_b = np.array([12.0, 17.0, 44.0])
    variance = np.array([4.0, 9.0, 16.0])
    result = profile_discrimination(
        coordinate, expected_a, expected_b, variance=variance
    )
    expected_local = (expected_a - expected_b) ** 2 / variance
    np.testing.assert_allclose(result.pointwise_discrimination, expected_local)
    assert result.total_discrimination == pytest.approx(expected_local.sum())


def test_correlated_profile_discrimination_uses_full_covariance():
    coordinate = np.array([1.0, 2.0, 3.0])
    difference = np.array([1.5, -0.5, 2.0])
    covariance = np.array([[2.0, 0.4, 0.1], [0.4, 1.5, 0.3], [0.1, 0.3, 1.0]])
    result = profile_discrimination(
        coordinate,
        difference,
        np.zeros_like(difference),
        covariance=covariance,
    )
    expected = difference @ np.linalg.solve(covariance, difference)
    assert result.total_discrimination == pytest.approx(expected)
    assert result.pointwise_discrimination is None
    assert result.covariance is not None


def _profile_pair(q_step):
    lattice = Lattice.cubic(5.0)
    model_a = Structure(lattice, ["Si", "O"], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    model_b = model_a.copy()
    model_b.translate_sites([1], [0.015, 0.0, 0.0], frac_coords=True)
    settings = dict(q_range=(0.5, 6.0), q_step=q_step)
    return BraggCalculator(**settings).load(model_a), BraggCalculator(**settings).load(model_b)


def test_count_discrimination_is_stable_to_profile_grid_refinement():
    coarse = compare_profile_counts(
        *_profile_pair(0.02), count_scale=100.0, background_density=10.0
    )
    fine = compare_profile_counts(
        *_profile_pair(0.01), count_scale=100.0, background_density=10.0
    )
    assert coarse.bin_widths is not None
    assert fine.total_discrimination == pytest.approx(
        coarse.total_discrimination, rel=0.03
    )


def test_profile_discrimination_rejects_ambiguous_error_model():
    values = np.ones(3)
    with pytest.raises(ValueError, match="exactly one"):
        profile_discrimination(values, values, values)
    with pytest.raises(ValueError, match="exactly one"):
        profile_discrimination(
            values,
            values,
            values,
            variance=values,
            covariance=np.eye(3),
        )
