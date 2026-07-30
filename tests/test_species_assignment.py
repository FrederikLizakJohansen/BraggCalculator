import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from braggcalculator import (
    BraggCalculator,
    DiffractionDataset,
    OptimizationStage,
    RefinementPolicy,
    SpeciesAssignmentConfig,
    apply_species_assignment,
    asymmetric_unit_sites,
    enumerate_species_assignments,
    refine_structure,
    refine_species_assignments,
)
from braggcalculator.io import to_pmg_structure


def _structures():
    lattice = Lattice.from_parameters(5.2, 6.1, 7.3, 78, 83, 71)
    coordinates = [[0.11, 0.22, 0.33], [0.43, 0.57, 0.68], [0.27, 0.14, 0.82]]
    target = Structure(lattice, ["Na", "Cl", "Cs"], coordinates)
    swapped = Structure(lattice, ["Cl", "Na", "Cs"], coordinates)
    return target, swapped


def _dataset(structure, radiation="xray"):
    calculator = BraggCalculator(
        mode=radiation,
        two_theta_range=(15.0, 70.0),
        two_theta_step=0.12,
        primitive=False,
    ).load(structure)
    coordinate, profile = calculator.pattern()
    intensity = 0.002 * np.asarray(profile) + 3.0
    return DiffractionDataset(
        coordinate=np.asarray(coordinate),
        intensity=intensity,
        sigma=np.ones(len(coordinate)),
        mask=np.ones(len(coordinate), dtype=bool),
        domain="two_theta",
        wavelength=calculator.wavelength,
        radiation=radiation,
    )


def _screening_policy():
    return RefinementPolicy(
        refine_lattice=False,
        background_degree=0,
        diagnostic_points=0,
        stages=(
            OptimizationStage("scale and background", ("scale", "background"), 8, 0.03),
        ),
    )


def test_complete_enumeration_is_deterministic_and_honors_site_rules():
    _, swapped = _structures()
    config = SpeciesAssignmentConfig(
        search="complete",
        fixed_sites=("framework",),
        site_groups={"framework": (2,), "exchange": (0, 1)},
        allowed_species={"exchange": ("Na", "Cl")},
    )
    first_sites, first = enumerate_species_assignments(swapped, config)
    second_sites, second = enumerate_species_assignments(swapped, config)

    assert first_sites == second_sites
    assert first.assignments == second.assignments == (
        ("Cl", "Na", "Cs"),
        ("Na", "Cl", "Cs"),
    )
    assert all(assignment[2] == "Cs" for assignment in first.assignments)
    assert all(site.multiplicity == 1 for site in first_sites)


def test_pairwise_and_random_search_preserve_composition_and_seed():
    _, swapped = _structures()
    pair_config = SpeciesAssignmentConfig(search="pairwise", fixed_sites=(2,))
    _, pairwise = enumerate_species_assignments(swapped, pair_config)
    assert pairwise.assignments == (("Cl", "Na", "Cs"), ("Na", "Cl", "Cs"))

    random_config = SpeciesAssignmentConfig(
        search="random",
        fixed_sites=(2,),
        max_candidates=2,
        seed=17,
    )
    _, first = enumerate_species_assignments(swapped, random_config)
    _, second = enumerate_species_assignments(swapped, random_config)
    assert first.assignments == second.assignments


def test_multiplicity_blocks_composition_changing_swaps():
    structure = Structure.from_spacegroup(
        "Pm-3m",
        Lattice.cubic(3.9),
        ["Ti", "Sr", "O"],
        [[0.5, 0.5, 0.5], [0, 0, 0], [0.5, 0.5, 0]],
    )
    sites, enumeration = enumerate_species_assignments(
        structure,
        SpeciesAssignmentConfig(search="complete"),
    )
    oxygen_index = next(site.site_index for site in sites if site.multiplicity == 3)
    assert len(enumeration.assignments) == 2
    assert all(assignment[oxygen_index] == "O" for assignment in enumeration.assignments)
    assert all(
        apply_species_assignment(structure, assignment).composition
        == structure.composition
        for assignment in enumeration.assignments
    )


def test_asymmetric_orbits_deduplicate_full_structure_sites():
    structure = Structure.from_spacegroup(
        "Fd-3m",
        Lattice.cubic(5.43),
        ["Si"],
        [[0, 0, 0]],
    )
    sites = asymmetric_unit_sites(structure)
    assert len(structure) > 1
    assert len(sites) == 1
    assert sites[0].equivalent_indices == tuple(range(len(structure)))


def test_bounded_search_reports_truncation():
    lattice = Lattice.from_parameters(5.2, 6.1, 7.3, 78, 83, 71)
    structure = Structure(
        lattice,
        ["Na", "Cl", "K"],
        [[0.11, 0.22, 0.33], [0.43, 0.57, 0.68], [0.27, 0.14, 0.82]],
    )
    _, enumeration = enumerate_species_assignments(
        structure,
        SpeciesAssignmentConfig(search="bounded", max_candidates=2),
    )
    assert len(enumeration.assignments) == 2
    assert enumeration.truncated
    _, complete = enumerate_species_assignments(
        structure,
        SpeciesAssignmentConfig(search="complete", max_candidates=10),
    )
    assert len(complete.assignments) == 6


def test_mixed_occupancy_and_displacement_policies_are_explicit():
    lattice = Lattice.from_parameters(5.2, 6.1, 7.3, 78, 83, 71)
    mixed = Structure(
        lattice,
        [{"Na": 0.5, "K": 0.5}, "Cl", "Cs"],
        [[0.11, 0.22, 0.33], [0.43, 0.57, 0.68], [0.27, 0.14, 0.82]],
    )
    _, enumeration = enumerate_species_assignments(
        mixed,
        SpeciesAssignmentConfig(search="complete", mixed_occupancy_policy="fixed"),
    )
    assert enumeration.assignments
    with pytest.raises(ValueError, match="mixed occupancies"):
        enumerate_species_assignments(
            mixed,
            SpeciesAssignmentConfig(mixed_occupancy_policy="reject"),
        )

    _, swapped = _structures()
    swapped.add_site_property("B_iso", [1.0, 2.0, 3.0])
    site_attached = apply_species_assignment(
        swapped,
        ("Na", "Cl", "Cs"),
        SpeciesAssignmentConfig(displacement_policy="site"),
    )
    species_attached = apply_species_assignment(
        swapped,
        ("Na", "Cl", "Cs"),
        SpeciesAssignmentConfig(displacement_policy="species"),
    )
    assert site_attached.site_properties["B_iso"] == [1.0, 2.0, 3.0]
    assert species_attached.site_properties["B_iso"] == [2.0, 1.0, 3.0]


def test_user_supplied_oxidation_states_filter_assignments():
    _, swapped = _structures()
    config = SpeciesAssignmentConfig(
        search="complete",
        fixed_sites=(2,),
        oxidation_states={"Na": 1, "Cl": -1, "Cs": 1},
        target_charge=1,
    )
    _, enumeration = enumerate_species_assignments(swapped, config)
    assert len(enumeration.assignments) == 2
    with pytest.raises(ValueError, match="missing values"):
        enumerate_species_assignments(
            swapped,
            SpeciesAssignmentConfig(oxidation_states={"Na": 1}),
        )


@pytest.mark.parametrize("radiation", ["xray", "neutron"])
def test_two_site_swap_is_recovered_with_a_fixed_reference_site(radiation):
    target, swapped = _structures()
    result = refine_species_assignments(
        _dataset(target, radiation),
        swapped,
        config=SpeciesAssignmentConfig(
            search="complete",
            fixed_sites=(2,),
            continuous_top_k=2,
            screening_background_degree=0,
            ambiguity_tolerance=1e-4,
        ),
        policy=_screening_policy(),
    )

    best = result.candidates[0]
    assert tuple(site.proposed_species for site in best.sites) == ("Na", "Cl", "Cs")
    assert best.screening_score < result.candidates[1].screening_score
    assert best.continuous_score < result.candidates[1].continuous_score
    assert best.refined_structure is not None
    assert "_cell_length_a" in best.refined_cif
    assert best.convergence["classification"]


def test_homometric_two_site_swap_is_reported_as_ambiguous():
    lattice = Lattice.from_parameters(5.2, 6.1, 7.3, 78, 83, 71)
    target = Structure(lattice, ["Na", "Cl"], [[0.11, 0.22, 0.33], [0.43, 0.57, 0.68]])
    result = refine_species_assignments(
        _dataset(target),
        target,
        config=SpeciesAssignmentConfig(
            search="complete",
            continuous_top_k=2,
            screening_background_degree=0,
            ambiguity_tolerance=1e-8,
        ),
        policy=_screening_policy(),
    )
    assert len(result.indistinguishable_assignments) == 2
    assert all(candidate.indistinguishable for candidate in result.candidates)
    assert any("Several species assignments" in warning for warning in result.warnings)


def test_observed_to_ranked_refined_cif_workflow(tmp_path):
    target, swapped = _structures()
    dataset = _dataset(target)
    q = (
        4.0
        * np.pi
        * np.sin(np.radians(dataset.coordinate) / 2.0)
        / dataset.wavelength
    )
    pattern = np.column_stack(
        (q, dataset.intensity, dataset.sigma)
    )
    cif = tmp_path / "candidate.cif"
    CifWriter(swapped, symprec=None).write_file(cif)
    parsed_sites = asymmetric_unit_sites(to_pmg_structure(cif))
    fixed = next(
        site.site_index
        for site in parsed_sites
        if site.original_species[0][0] == "Cs"
    )

    result = refine_structure(
        pattern,
        cif,
        wavelength=dataset.wavelength,
        domain="q",
        policy=_screening_policy(),
        species_assignment=SpeciesAssignmentConfig(
            search="complete",
            fixed_sites=(fixed,),
            continuous_top_k=2,
            screening_background_degree=0,
        ),
    )

    assignments = result.species_assignments
    assert assignments.candidates[0].continuous_result is not None
    assert result.refined_cif == assignments.candidates[0].refined_cif
    assert assignments.evaluated_count == 2
    assert result.dataset.domain == "q"
    np.testing.assert_allclose(result.coordinate, q)
