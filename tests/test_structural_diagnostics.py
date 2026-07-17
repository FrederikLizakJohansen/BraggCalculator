import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from braggcalculator import (
    BraggCalculator,
    classify_structural_relationship,
    compare_pair_distributions,
    counterfactual_site_substitutions,
    diagnose_structures,
    identify_superstructure_reflections,
    peak_group_attribution,
    suggest_measurements,
)
from braggcalculator.profiles import GaussianProfileQ


def _compatible_pair():
    lattice = Lattice.from_parameters(4.2, 5.1, 6.3, 78, 82, 73)
    structure_a = Structure(
        lattice,
        ["Si", "O", "O"],
        [[0.13, 0.21, 0.34], [0.31, 0.47, 0.11], [0.72, 0.08, 0.59]],
    )
    structure_b = structure_a.copy()
    structure_b.translate_sites([1], [0.025, 0.0, 0.0], frac_coords=True)
    return structure_a, structure_b


def _calculator(structure):
    return BraggCalculator(
        primitive=False,
        q_range=(0.5, 4.5),
        q_step=0.03,
        profile_q=GaussianProfileQ(0.1),
    ).load(structure)


def test_relationship_classifier_covers_all_three_regimes():
    structure_a, structure_b = _compatible_pair()
    shifted = Structure(
        structure_a.lattice,
        [site.species for site in reversed(structure_a)],
        [(site.frac_coords + [0.125, 0.25, 0.375]) % 1 for site in reversed(structure_a)],
    )
    equivalent = classify_structural_relationship(structure_a, shifted)
    compatible = classify_structural_relationship(structure_a, structure_b)

    supercell = structure_a.copy()
    supercell.make_supercell([2, 1, 1])
    supercell.replace(3, "N")
    commensurate = classify_structural_relationship(structure_a, supercell)

    unrelated = Structure(
        Lattice.hexagonal(3.05, 5.37),
        ["Si", "O", "O"],
        [[0, 0, 0], [1 / 3, 2 / 3, 0.22], [2 / 3, 1 / 3, 0.78]],
    )
    regime_three = classify_structural_relationship(structure_a, unrelated)

    assert equivalent.classification == "equivalent"
    assert compatible.classification == "lattice_compatible"
    assert commensurate.regime == "II"
    assert commensurate.transformation_direction == "a_to_b"
    assert abs(round(np.linalg.det(commensurate.transformation))) == 2
    assert regime_three.regime == "III"
    assert not regime_three.complex_comparison_allowed


def test_rotated_cartesian_lattice_does_not_destroy_hkl_correspondence():
    structure_a, _ = _compatible_pair()
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_equivalent = Structure(
        Lattice(structure_a.lattice.matrix @ rotation),
        [site.species for site in structure_a],
        [site.frac_coords for site in structure_a],
    )
    equivalent_result = diagnose_structures(
        structure_a,
        rotated_equivalent,
        q_range=(0.5, 3.5),
        q_step=0.05,
        profile_fwhm_q=0.12,
        pair_r_max=4.0,
    )
    structure_b = rotated_equivalent.copy()
    structure_b.translate_sites([1], [0.02, 0.0, 0.0], frac_coords=True)
    relationship = classify_structural_relationship(structure_a, structure_b)
    result = diagnose_structures(
        structure_a,
        structure_b,
        q_range=(0.5, 3.5),
        q_step=0.05,
        profile_fwhm_q=0.12,
        pair_r_max=4.0,
    )

    assert relationship.classification == "lattice_compatible"
    np.testing.assert_array_equal(relationship.transformation, np.eye(3, dtype=int))
    assert equivalent_result.relationship.classification == "equivalent"
    assert equivalent_result.mismatch.d_sf < 1e-12
    assert result.mismatch is not None


def test_unrelated_diagnostic_disables_phase_but_retains_powder_and_pdf():
    structure_a = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    structure_b = Structure(
        Lattice.hexagonal(3.1, 5.2), ["Si", "Si"], [[0, 0, 0], [1 / 3, 2 / 3, 0.25]]
    )
    result = diagnose_structures(
        structure_a,
        structure_b,
        q_range=(0.5, 4.0),
        q_step=0.04,
        profile_fwhm_q=0.12,
        pair_r_max=5.0,
    )

    assert result.relationship.regime == "III"
    assert result.mismatch is None
    assert result.similarities["complex"] is None
    assert 0 <= result.similarities["profile"] <= 1
    assert 0 <= result.pair_distribution.similarity <= 1
    assert result.dominant_information_loss == "unrelated_lattices"


def test_counterfactual_and_peak_groups_expose_site_evidence():
    structure_a, structure_b = _compatible_pair()
    calculator_a = _calculator(structure_a)
    calculator_b = _calculator(structure_b)
    substitutions = counterfactual_site_substitutions(
        calculator_a, calculator_b, {"all sites": [0, 1, 2], "moved oxygen": [1]}
    )
    all_sites, oxygen = substitutions

    assert all_sites.effect_norm == pytest.approx(1.0, rel=1e-8)
    assert all_sites.alignment_fraction == pytest.approx(1.0, rel=1e-8)
    assert oxygen.effect_norm > 0
    groups = peak_group_attribution(
        calculator_a,
        fwhm_q=0.1,
        site_groups={"silicon": [0], "oxygen": [1, 2]},
        maximum_groups=5,
    )
    assert groups
    assert all(group.effective_reflections >= 1.0 for group in groups)
    assert set(groups[0].site_effects) == {"silicon", "oxygen"}


def test_information_ladder_detects_phase_loss_for_a_site_displacement():
    structure_a, structure_b = _compatible_pair()
    result = diagnose_structures(
        structure_a,
        structure_b,
        q_range=(0.5, 4.5),
        q_step=0.04,
        profile_fwhm_q=0.15,
        pair_r_max=5.0,
    )

    assert result.relationship.classification == "lattice_compatible"
    assert result.mismatch is not None
    assert result.similarities["intensity"] > result.similarities["complex"]
    assert result.dominant_information_loss == "phase_loss"


def test_superstructure_and_pair_distribution_are_quantitative():
    parent = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    ordered = parent.copy()
    ordered.make_supercell([2, 1, 1])
    ordered.replace(1, "P")
    calculator_parent = _calculator(parent)
    calculator_ordered = _calculator(ordered)
    relationship = classify_structural_relationship(parent, ordered)
    superstructure = identify_superstructure_reflections(
        calculator_parent, calculator_ordered, relationship
    )

    assert superstructure is not None
    assert len(superstructure.hkl) > 0
    assert 0 < superstructure.intensity_fraction < 1
    identical_pdf = compare_pair_distributions(parent, parent, r_max=5.0)
    changed_pdf = compare_pair_distributions(parent, ordered, r_max=5.0)
    assert identical_pdf.similarity == pytest.approx(1.0)
    assert 0 <= changed_pdf.similarity <= 1


def test_measurement_suggestions_reward_resolution_for_split_peaks():
    structure_a = Structure(
        Lattice.cubic(4.0), ["Si", "O"], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    structure_b = Structure(
        Lattice.cubic(4.04), ["Si", "O"], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    recommendations = suggest_measurements(
        structure_a,
        structure_b,
        [
            {
                "name": "standard resolution",
                "q_range": (0.5, 5.0),
                "q_step": 0.02,
                "fwhm_q": 0.25,
                "count_scale": 1000.0,
            },
            {
                "name": "high resolution",
                "q_range": (0.5, 5.0),
                "q_step": 0.02,
                "fwhm_q": 0.04,
                "count_scale": 1000.0,
            },
        ],
    )

    assert recommendations[0].name == "high resolution"
    assert recommendations[0].total_discrimination > recommendations[1].total_discrimination
    assert recommendations[0].assumptions["variance"].startswith("symmetric")
