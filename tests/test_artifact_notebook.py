from pathlib import Path

import nbformat

from scripts.build_artifact_simulation_notebook import build_notebook


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "artifact_simulation.ipynb"


def test_artifact_notebook_is_reproducibly_generated_and_complete():
    stored = nbformat.read(NOTEBOOK, as_version=4)
    generated = build_notebook()
    assert [
        (cell["cell_type"], cell["source"]) for cell in stored["cells"]
    ] == [
        (cell["cell_type"], cell["source"]) for cell in generated["cells"]
    ]

    source = "\n".join(cell["source"] for cell in stored["cells"])
    for required in (
        "NaCl.cif",
        "BackgroundPattern.from_file",
        "CalibrationArtifacts",
        "PeakProfileArtifacts",
        "PreferredOrientation",
        "NoiseArtifacts",
        "DetectorArtifacts",
        "SpuriousPeakArtifacts",
        "SimulationArtifacts",
        "calculator.pattern(artifacts=artifacts)",
        "Verify reproducibility",
    ):
        assert required in source


def test_artifact_notebook_has_no_execution_errors():
    stored = nbformat.read(NOTEBOOK, as_version=4)
    errors = [
        output
        for cell in stored["cells"]
        if cell["cell_type"] == "code"
        for output in cell.get("outputs", ())
        if output.get("output_type") == "error"
    ]
    assert errors == []
