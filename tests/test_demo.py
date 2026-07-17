import numpy as np

from demo.compare_with_pymatgen import calculate_patterns, plot_comparison
from demo.analyze_profile_information import calculate_information, plot_information
from demo.diagnose_compatible_models import calculate_diagnostics, plot_disk
from demo.refine_symmetry_coordinates import plot_refinement, run_refinement


def test_demo_matches_pymatgen_and_writes_figure(tmp_path):
    actual_x, actual_y, expected_x, expected_y = calculate_patterns()
    assert actual_x.shape == expected_x.shape
    assert actual_y.shape == expected_y.shape

    output = tmp_path / "comparison.png"
    position_error, intensity_error = plot_comparison(output)
    assert output.stat().st_size > 0
    assert position_error < 1e-10
    assert intensity_error < 1e-10


def test_mismatch_disk_demo_aligns_origin_and_writes_figure(tmp_path):
    unaligned, aligned, matched_q = calculate_diagnostics()
    assert len(matched_q) == len(aligned.match)
    assert 0 < aligned.d_sf < unaligned.d_sf
    assert aligned.identity_error < 1e-14

    output = tmp_path / "mismatch.png"
    plotted = plot_disk(output)
    assert output.stat().st_size > 0
    assert plotted.d_sf == aligned.d_sf


def test_profile_information_demo_identifies_supported_oxygen_direction(tmp_path):
    comparison, diagnostics = calculate_information()
    assert comparison.total_discrimination > 0
    assert diagnostics.covariance_is_identifiable
    strongest = np.argmax(np.abs(diagnostics.residual_support))
    assert diagnostics.parameter_names[strongest] == "O1 x"

    output = tmp_path / "information.png"
    plotted_comparison, plotted_diagnostics = plot_information(output)
    assert output.stat().st_size > 0
    assert plotted_comparison.total_discrimination == comparison.total_discrimination
    assert plotted_diagnostics.rank == diagnostics.rank


def test_symmetry_refinement_demo_recovers_displacement_and_writes_figure(tmp_path):
    target, recovered, history, orbit_change = run_refinement()
    np.testing.assert_allclose(recovered, target, atol=1e-5)
    np.testing.assert_allclose(orbit_change[0], -orbit_change[1], atol=1e-12)
    assert history[-1] < 1e-10

    output = tmp_path / "symmetry-refinement.png"
    plotted_target, plotted_recovered, _, _ = plot_refinement(output)
    assert output.stat().st_size > 0
    np.testing.assert_allclose(plotted_recovered, plotted_target, atol=1e-5)
