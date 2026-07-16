from demo.compare_with_pymatgen import calculate_patterns, plot_comparison


def test_demo_matches_pymatgen_and_writes_figure(tmp_path):
    actual_x, actual_y, expected_x, expected_y = calculate_patterns()
    assert actual_x.shape == expected_x.shape
    assert actual_y.shape == expected_y.shape

    output = tmp_path / "comparison.png"
    position_error, intensity_error = plot_comparison(output)
    assert output.stat().st_size > 0
    assert position_error < 1e-10
    assert intensity_error < 1e-10
