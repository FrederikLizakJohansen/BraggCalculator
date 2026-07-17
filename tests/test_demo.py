import numpy as np
import pytest

from demo.compare_with_pymatgen import calculate_patterns, plot_comparison
from demo.analyze_profile_information import calculate_information, plot_information
from demo.diagnose_compatible_models import calculate_diagnostics, plot_disk
from demo.refine_symmetry_coordinates import plot_refinement, run_refinement
from demo.refine_staged import plot_staged_refinement, run_staged_example
from demo.refine_occupancy_adp import run_occupancy_adp_demo
from demo.refine_anisotropic_restraints import run_anisotropic_restraint_demo


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


def test_staged_refinement_demo_recovers_structure_and_profile(tmp_path):
    result = run_staged_example()
    np.testing.assert_allclose(
        result["recovered_coordinates"], result["target_coordinates"], atol=2e-5
    )
    tolerances = {"scale": 2e-3, "background": 1.0, "zero_shift": 2e-5, "fwhm": 2e-4}
    for name, tolerance in tolerances.items():
        assert result["recovered_physical"][name] == pytest.approx(
            result["target_physical"][name], abs=tolerance
        )
    assert result["trace"].loss[-1] < 0.05

    output = tmp_path / "staged.png"
    plot_staged_refinement(output, result=result)
    assert output.stat().st_size > 0


def test_occupancy_adp_demo_recovers_controlled_case_and_flags_joint_ambiguity(tmp_path):
    figure = tmp_path / "occupancy-adp.png"
    report = tmp_path / "occupancy-adp.html"
    _, controlled, joint = run_occupancy_adp_demo(figure, report)

    controlled_ca = controlled["occupancy"][0]["species"]["Ca"]
    controlled_b = [item["B_iso"] for item in controlled["b_iso"]]
    joint_ca = joint.physical_parameters["occupancy_groups"][0]["species"]["Ca"]

    assert figure.stat().st_size > 0
    assert report.stat().st_size > 0
    assert controlled_ca == pytest.approx(0.55, abs=1e-5)
    np.testing.assert_allclose(controlled_b, [0.6, 0.35, 1.1], atol=1e-5)
    assert joint.r_wp < 0.02
    assert abs(joint_ca - 0.55) > 0.01
    assert any("strongly correlated" in warning for warning in joint.warnings)


def test_anisotropic_restraint_demo_recovers_tensor_and_resolves_geometry(tmp_path):
    figure = tmp_path / "anisotropic-restraints.png"
    report = tmp_path / "anisotropic-restraints.html"
    candidate, target_u, recovered_u, unrestrained, restrained = run_anisotropic_restraint_demo(
        figure, report
    )

    assert figure.stat().st_size > 0
    assert report.stat().st_size > 0
    assert candidate.r_wp < 0.001
    np.testing.assert_allclose(recovered_u, target_u, atol=1e-4)
    assert np.linalg.eigvalsh(recovered_u).min() > 0
    assert unrestrained["history"][-1, 0] < 1e-8
    assert abs(unrestrained["metrics"][2] - 109.5) > 20
    np.testing.assert_allclose(restrained["metrics"], [1.62, 1.62, 109.5], atol=1e-5)
    assert restrained["history"][-1, 0] < 1e-6
    assert candidate.physical_parameters["structural_restraint_contributions"] == {
        "composition[0].Si": 0.0
    }
    assert any("prior information" in warning for warning in candidate.warnings)
