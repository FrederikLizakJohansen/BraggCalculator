import numpy as np
import pytest

from scripts.plot_pattern_comparison import compare_pattern, gaussian_pattern, regular_grid


def test_gaussian_pattern_preserves_integrated_area_away_from_boundaries():
    grid = regular_grid(10.0, 80.0, 0.002)
    profile = gaussian_pattern(
        grid,
        centers=np.array([25.0, 50.0, 70.0]),
        integrated_intensities=np.array([1.0, 2.0, 3.0]),
        fwhm_deg=0.1,
    )
    assert np.trapezoid(profile, grid) == pytest.approx(6.0, rel=1e-12)


def test_pattern_comparison_matches_oracle(nacl):
    comparison = compare_pattern(
        "NaCl",
        nacl,
        mode="xray",
        two_theta_range=(10.0, 80.0),
        step_deg=0.02,
        fwhm_deg=0.1,
        position_atol=1e-10,
        intensity_atol=1e-9,
        profile_atol=1e-8,
    )
    assert comparison.metrics.peaks > 0
    assert comparison.metrics.max_position_error_deg <= 1e-10
    assert comparison.metrics.max_line_intensity_error_percent <= 1e-9
    assert comparison.metrics.max_profile_error_percent <= 1e-8
