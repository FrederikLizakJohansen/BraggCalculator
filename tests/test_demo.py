from demo.compare_with_pymatgen import calculate_patterns, plot_comparison
from demo.diagnose_compatible_models import calculate_diagnostics, plot_disk


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
