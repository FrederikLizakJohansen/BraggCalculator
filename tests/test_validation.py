import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.reference_cases import reference_structures
from braggcalculator import (
    DiffractionDataset,
    ValidationCase,
    ValidationMatrix,
    ValidationMetric,
    load_reference_sources,
    validate_line_oracle,
    validate_public_sources,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "points", "start", "step", "radiation", "wavelength"),
    [
        ("PBSO4.XRA", 6001, 10.0, 0.025, "xray", 1.5405),
        ("PBSO4.CWN", 2919, 10.0, 0.05, "neutron", 1.909),
        ("FAP.XRA", 5753, 15.0, 0.02, "xray", 1.5405),
        ("garnet.raw", 2679, 24.0, 0.05, "neutron", 1.909),
    ],
)
def test_gsas_constant_step_reference_data_are_parsed(
    filename, points, start, step, radiation, wavelength
):
    dataset = DiffractionDataset.from_gsas_constant_step(
        ROOT / "data/reference_validation" / filename,
        wavelength=wavelength,
        radiation=radiation,
    )
    assert len(dataset.coordinate) == points
    assert dataset.coordinate[0] == pytest.approx(start)
    assert dataset.step == pytest.approx(step)
    assert np.all(np.isfinite(dataset.intensity))
    assert np.all(dataset.sigma > 0)


def test_public_reference_manifest_checksums_and_ingestion_pass():
    sources = load_reference_sources(ROOT / "data/reference_validation/manifest.json")
    cases = validate_public_sources(ROOT, sources)
    assert len(sources) == 5
    assert {source.radiation for source in sources} == {"xray", "neutron"}
    assert len({source.material for source in sources}) >= 4
    assert all(case.status == "pass" for case in cases)


def test_validation_matrix_retains_warnings_failures_and_review_status(tmp_path):
    passing = ValidationCase(
        "pass", "oracle", "passes",
        (ValidationMetric("error", 1e-12, direction="maximum", pass_limit=1e-10, warn_limit=1e-8),),
    )
    warning = ValidationCase(
        "warning", "capability", "warns", declared_status="unsupported"
    )
    matrix = ValidationMatrix(
        (passing, warning), required_categories=("oracle", "capability"),
        expert_review_status="pending_review",
    )
    assert matrix.overall_status == "unsupported"
    assert matrix.status_counts["pass"] == 1
    assert matrix.status_counts["unsupported"] == 1
    output = tmp_path / "validation.json"
    matrix.write_json(output)
    encoded = json.loads(output.read_text())
    assert encoded["cases"][1]["status"] == "unsupported"
    assert encoded["overall_status"] == "unsupported"


def test_missing_required_validation_category_fails_matrix():
    matrix = ValidationMatrix((), required_categories=("synthetic_recovery",))
    assert matrix.missing_categories == ("synthetic_recovery",)
    assert matrix.overall_status == "fail"


def test_small_reference_line_oracle_matrix_passes():
    structures = reference_structures()
    cases = validate_line_oracle({"NaCl": structures["NaCl"]})
    assert len(cases) == 2
    assert all(case.status == "pass" for case in cases)
