import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from braggcalculator import (
    BraggCalculator,
    DiffractionDataset,
    OptimizationStage,
    RefinementPolicy,
    load_refinement_dataset,
    refine_generated_cif,
)


def _short_policy():
    return RefinementPolicy(
        refine_lattice=False,
        background_degree=0,
        diagnostic_points=0,
        stages=(
            OptimizationStage("scale and background", ("scale", "background"), 3, 0.02),
        ),
    )


def _synthetic_pattern(structure):
    calculator = BraggCalculator(
        two_theta_range=(20.0, 60.0),
        two_theta_step=0.2,
        primitive=False,
    ).load(structure)
    coordinate, profile = calculator.pattern()
    intensity = 0.001 * np.asarray(profile) + 2.0
    return calculator, np.column_stack((coordinate, intensity, np.ones(len(coordinate))))


def test_generated_cif_entry_point_returns_complete_structured_result(tmp_path):
    structure = Structure(
        Lattice.cubic(4.1),
        ["Cs", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    calculator, pattern = _synthetic_pattern(structure)
    cif = tmp_path / "candidate.cif"
    CifWriter(structure, symprec=None).write_file(cif)

    result = refine_generated_cif(
        pattern,
        cif,
        wavelength=calculator.wavelength,
        policy=_short_policy(),
    )

    assert result.refined_structure.composition.reduced_formula == "CsCl"
    assert "_cell_length_a" in result.refined_cif
    assert result.coordinate.shape == result.observed.shape == result.calculated.shape
    np.testing.assert_allclose(result.residual, result.observed - result.calculated)
    assert len(result.objective_history) == len(result.stage_history) == 3
    assert result.convergence["classification"]
    assert result.status in {"converged", "completed"}
    assert result.fit_statistics["r_wp"] >= 0
    assert result.parameters
    scale = next(item for item in result.parameters if item.path == "scale")
    assert scale.lower_bound == 0
    assert scale.released
    output = result.write_cif(tmp_path / "refined.cif")
    assert output.read_text(encoding="utf-8") == result.refined_cif


def test_dataset_loader_accepts_sigma_weights_and_checked_dataset():
    values = np.array([[10.0, 100.0], [11.0, 121.0], [12.0, 144.0]])
    weighted = load_refinement_dataset(
        values,
        wavelength=1.54,
        weights=np.array([0.25, 0.2, 0.16]),
    )
    np.testing.assert_allclose(weighted.sigma, [2.0, np.sqrt(5.0), 2.5])

    supplied = DiffractionDataset(
        coordinate=values[:, 0],
        intensity=values[:, 1],
        sigma=np.ones(3),
        mask=np.ones(3, dtype=bool),
        domain="two_theta",
        wavelength=1.54,
    )
    assert load_refinement_dataset(supplied, wavelength=1.54) is supplied


@pytest.mark.parametrize(
    ("pattern", "message"),
    [
        (np.ones((3, 4)), "two or three columns"),
        (np.array([[10.0, 1.0], [10.0, 2.0], [12.0, 3.0]]), "strictly increasing"),
        (np.array([[10.0, 1.0], [11.0, np.nan], [12.0, 3.0]]), "intensity must be finite"),
    ],
)
def test_generated_cif_workflow_rejects_invalid_experimental_arrays(pattern, message):
    with pytest.raises(ValueError, match=message):
        load_refinement_dataset(pattern, wavelength=1.54)


def test_generated_cif_workflow_rejects_invalid_cif_and_q_session(tmp_path):
    pattern = np.array([[10.0, 1.0], [11.0, 2.0], [12.0, 3.0]])
    invalid = tmp_path / "invalid.cif"
    invalid.write_text("this is not a CIF", encoding="utf-8")
    with pytest.raises((ValueError, KeyError, IndexError)):
        refine_generated_cif(pattern, invalid, wavelength=1.54, policy=_short_policy())

    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    with pytest.raises(ValueError, match="two-theta"):
        refine_generated_cif(
            pattern,
            structure,
            wavelength=1.54,
            domain="q",
            policy=_short_policy(),
        )
