import json
from pathlib import Path

import numpy as np

from scripts.plot_artifact_gallery import (
    artifact_configurations,
    calculate_gallery,
    plot_gallery,
    write_metadata,
)


def test_artifact_gallery_covers_every_public_effect_family(tmp_path):
    titles = [title for title, _, _ in artifact_configurations()]
    assert titles == [
        "Ideal reference",
        "Calibration",
        "Peak profile",
        "Intensities",
        "Background",
        "Spurious peaks",
        "Noise",
        "Detector",
        "Combined",
    ]

    grid, ideal, results = calculate_gallery("demo/NaCl.cif")
    assert len(results) == 9
    assert all(values.shape == grid.shape for _, _, values in results)
    assert all(np.all(np.isfinite(values)) for _, _, values in results)
    assert all(np.min(values) >= 0 for _, _, values in results)
    assert all(np.max(values) <= 1 for _, _, values in results)
    assert all(
        not np.allclose(values, ideal) for _, _, values in results[1:]
    )

    output_stem = tmp_path / "gallery"
    metadata = tmp_path / "gallery.json"
    plot_gallery(grid, ideal, results, output_stem)
    write_metadata(grid, ideal, results, metadata, "demo/NaCl.cif")
    for suffix in (".pdf", ".svg", ".png"):
        assert output_stem.with_suffix(suffix).stat().st_size > 0
    record = json.loads(metadata.read_text())
    assert record["schema_version"] == 1
    assert len(record["panels"]) == 9


def test_artifact_gallery_is_referenced_by_both_manuscript_sources():
    root = Path(__file__).resolve().parents[1]
    assert "figures/artifact_gallery.pdf" in (root / "paper" / "paper.md").read_text()
    assert "figures/artifact_gallery.pdf" in (root / "paper" / "paper.tex").read_text()
