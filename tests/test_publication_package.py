import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from braggcalculator.publication import (
    PUBLICATION_SCHEMA,
    WEIGHTING_SCHEMES,
    cyclic_difference_multiset,
    cyclic_sets_dihedrally_equivalent,
    gaussian_cross_correlation_similarity,
    jensen_shannon_similarity,
    mismatch_weights,
    profile_metric_suite,
    publication_gate_summary,
    verify_input_manifest,
)


ROOT = Path(__file__).parents[1]


def test_frozen_cyclic_pair_is_homometric_but_not_congruent():
    first = (0, 3, 4, 5)
    second = (0, 1, 3, 4)
    assert cyclic_difference_multiset(first, 8) == cyclic_difference_multiset(second, 8)
    assert not cyclic_sets_dihedrally_equivalent(first, second, 8)
    assert cyclic_sets_dihedrally_equivalent(first, (1, 4, 5, 6), 8)


def test_profile_baselines_are_bounded_and_cross_correlation_tolerates_shift():
    coordinate = np.linspace(-2, 2, 801)
    first = np.exp(-0.5 * (coordinate / 0.08) ** 2)
    second = np.exp(-0.5 * ((coordinate - 0.025) / 0.08) ** 2)
    metrics = profile_metric_suite(
        first, second, coordinate_step=coordinate[1] - coordinate[0],
        cross_correlation_tolerance=0.04,
    )
    assert all(-1 <= value <= 1 for value in metrics.values())
    assert metrics["gaussian_cross_correlation"] > metrics["cosine"]
    assert jensen_shannon_similarity(first, first) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-negative"):
        gaussian_cross_correlation_similarity(
            first - 2, second, coordinate_step=0.01, tolerance=0.1
        )


def test_shell_balanced_weights_assign_equal_total_to_each_shell():
    q = np.array([0.1, 0.2, 0.7, 0.8, 1.2])
    amplitude_a = np.array([10.0, 2.0, 4.0, 1.0, 0.5])
    amplitude_b = amplitude_a * 0.9
    weights = mismatch_weights(
        q, amplitude_a, amplitude_b,
        scheme="shell_balanced_intensity", shell_width=0.5,
    )
    shell = np.floor((q - q.min()) / 0.5).astype(int)
    totals = [weights[shell == index].sum() for index in np.unique(shell)]
    np.testing.assert_allclose(totals, np.full(3, 1 / 3), atol=1e-15)
    assert weights.sum() == pytest.approx(1.0)
    for scheme in WEIGHTING_SCHEMES:
        assert np.all(mismatch_weights(q, amplitude_a, amplitude_b, scheme=scheme) >= 0)


def test_publication_manifest_and_release_gate_are_explicit():
    data = ROOT / "data" / "publication_diagnostics"
    verified = verify_input_manifest(data, data / "manifest.json")
    assert len(verified) == 7
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["schema"] == PUBLICATION_SCHEMA
    assert publication_gate_summary({"numerical": True, "external": None}) == (
        "pending_external_review"
    )
    assert publication_gate_summary({"numerical": False, "external": None}) == "failed"
    assert publication_gate_summary({"numerical": True, "external": True}) == "passed"


def test_checked_in_publication_artifacts_match_their_byte_manifest():
    publication = ROOT / "paper" / "diagnostics"
    result = json.loads((publication / "results.json").read_text())
    assert result["release_status"] == "pending_external_review"
    assert result["gates"]["external_expert_review"] is None
    assert all(
        value is True
        for name, value in result["gates"].items()
        if name != "external_expert_review"
    )
    manifest = json.loads((publication / "artifact_manifest.json").read_text())
    for record in manifest["generated_artifacts"]:
        path = publication / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256(path.read_bytes()).hexdigest() == record["sha256"]
    for record in manifest["analysis_sources"]:
        path = ROOT / record["path"]
        assert sha256(path.read_bytes()).hexdigest() == record["sha256"]
