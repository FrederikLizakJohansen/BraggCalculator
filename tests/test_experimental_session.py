from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from braggcalculator import (
    BraggCalculator,
    DiffractionDataset,
    OptimizationStage,
    RefinementPolicy,
    RefinementSession,
)
from braggcalculator.backends import NumpyBackend
from braggcalculator.experimental_profile import render_pseudo_voigt


ROOT = Path(__file__).resolve().parents[1]


def test_xye_ingestion_preserves_uncertainty_mask_and_provenance(tmp_path):
    path = tmp_path / "pattern.xye"
    path.write_text("# x y w\n10,100,0.25\n11,121,0.20\n12,144,0.16\n")
    dataset = DiffractionDataset.from_xye(
        path, wavelength=1.54, third_column="weight"
    )
    np.testing.assert_allclose(dataset.sigma, [2.0, np.sqrt(5.0), 2.5])
    assert dataset.source_sha256 is not None
    assert dataset.step == pytest.approx(1.0)
    excluded = dataset.exclude([(10.5, 11.5)])
    np.testing.assert_array_equal(excluded.mask, [True, False, True])


def test_pseudo_voigt_is_area_normalized():
    backend = NumpyBackend()
    grid = np.linspace(-5, 5, 20001)
    profile = render_pseudo_voigt(
        grid,
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.2]),
        0.35,
        backend,
    )
    assert np.trapezoid(profile, grid) == pytest.approx(3.0, rel=0.01)


def test_candidate_session_ranks_generating_structure_and_writes_report(tmp_path, nacl):
    generator = BraggCalculator(
        two_theta_range=(20.0, 60.0), two_theta_step=0.05
    ).load(nacl)
    coordinate, profile = generator.pattern()
    observed = 0.002 * profile + 10.0
    sigma = np.sqrt(np.maximum(observed, 1.0))
    dataset = DiffractionDataset(
        coordinate=coordinate,
        intensity=observed,
        sigma=sigma,
        mask=np.ones(len(coordinate), dtype=bool),
        domain="two_theta",
        wavelength=generator.wavelength,
        metadata={"kind": "synthetic candidate regression"},
    )
    competing = Structure(
        Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    session = RefinementSession(dataset, [nacl, competing], names=["NaCl", "CsCl-like"])
    result = session.run(RefinementPolicy.quick())
    assert result.ranking[0] == "NaCl"
    assert result.pairwise_discrimination
    report = session.write_html(result, tmp_path / "report.html")
    text = report.read_text()
    assert "Diffraction diagnostic report" in text
    assert "Largest unexplained regions" in text
    assert "SHA-256" in text


def test_nist_srm660c_real_data_regression():
    metadata = {
        "wavelength_components": [(1.5405929, 2.0), (1.5444274, 1.0)],
        "reference_lattice_angstrom": 4.156826,
    }
    dataset = DiffractionDataset.from_xye(
        ROOT / "data/nist_srm660c_100a_20-50.xye",
        wavelength=1.5405929,
        metadata=metadata,
    )
    policy = RefinementPolicy(
        stages=(
            OptimizationStage("scale/background", ("scale", "background"), 40, 0.03),
            OptimizationStage("profile/lattice", ("profile", "lattice"), 60, 0.015),
            OptimizationStage(
                "joint", ("scale", "background", "profile", "lattice"), 80, 0.005
            ),
        )
    )
    result = RefinementSession(
        dataset, [ROOT / "data/LaB6_srm660c.cif"], names=["LaB6"]
    ).run(policy)
    candidate = result.candidates[0]
    refined_a = candidate.physical_parameters["lattice"][0][0]
    assert candidate.r_wp < 0.4
    assert refined_a == pytest.approx(metadata["reference_lattice_angstrom"], abs=0.01)
    assert candidate.provenance["dataset_sha256"] == dataset.source_sha256
    assert candidate.warnings  # The simplified profile must not claim certification-grade agreement.
