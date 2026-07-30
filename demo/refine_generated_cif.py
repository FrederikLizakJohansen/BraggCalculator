"""Run the generated-CIF refinement and species-assignment workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from braggcalculator import (
    BraggCalculator,
    OptimizationStage,
    RefinementPolicy,
    SpeciesAssignmentConfig,
    asymmetric_unit_sites,
    refine_generated_cif,
)
from braggcalculator.io import to_pmg_structure
from braggcalculator.experimental_profile import (
    axial_divergence_widths,
    render_split_pseudo_voigt,
    thompson_cox_hastings,
)


ROOT = Path(__file__).resolve().parent


def run(output: Path):
    lattice = Lattice.from_parameters(5.2, 6.1, 7.3, 78, 83, 71)
    coordinates = [[0.11, 0.22, 0.33], [0.43, 0.57, 0.68], [0.27, 0.14, 0.82]]
    target = Structure(lattice, ["Na", "Cl", "Cs"], coordinates)
    generator = BraggCalculator(
        two_theta_range=(15.0, 70.0),
        two_theta_step=0.12,
        primitive=False,
    ).load(target)
    two_theta = np.arange(15.0, 70.0001, 0.12)
    centers, areas = generator.line_components(
        [generator.wavelength],
        domain="two_theta",
    )[0]
    radians = np.radians(centers)
    widths, eta = thompson_cox_hastings(
        radians,
        0.0025,
        0.0,
        0.0036,
        0.01,
        0.01,
        generator.backend,
    )
    low_widths, high_widths = axial_divergence_widths(
        widths,
        radians,
        0.05,
        generator.backend,
    )
    profile = render_split_pseudo_voigt(
        two_theta,
        centers,
        areas,
        low_widths,
        high_widths,
        eta,
        generator.backend,
    )
    observed = 0.002 * np.asarray(profile) + 3.0
    sigma = np.ones(len(two_theta))
    pattern = np.column_stack((two_theta, observed, sigma))

    candidate_path = ROOT / "refinement_candidate.cif"
    candidate = to_pmg_structure(candidate_path)
    sites = asymmetric_unit_sites(candidate)
    fixed_cs = next(
        site.site_index
        for site in sites
        if site.original_species[0][0] == "Cs"
    )
    policy = RefinementPolicy(
        refine_lattice=False,
        background_degree=0,
        diagnostic_points=16,
        stages=(
            OptimizationStage(
                "scale and background",
                ("scale", "background"),
                20,
                0.03,
            ),
        ),
    )
    result = refine_generated_cif(
        pattern,
        candidate_path,
        wavelength=generator.wavelength,
        policy=policy,
        species_assignment=SpeciesAssignmentConfig(
            search="complete",
            fixed_sites=(fixed_cs,),
            continuous_top_k=2,
            screening_background_degree=0,
            ambiguity_tolerance=1e-4,
        ),
    )
    result.write_cif(output)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "refined-demo.cif")
    args = parser.parse_args()
    result = run(args.output)
    assignments = result.species_assignments
    print(f"Search mode: {assignments.search_mode}")
    print(f"Evaluated assignments: {assignments.evaluated_count}")
    for rank, candidate in enumerate(assignments.candidates, start=1):
        proposed = ", ".join(
            f"site {site.site_index}={site.proposed_species}"
            for site in candidate.sites
        )
        print(
            f"{rank}. {proposed}; screen={candidate.screening_score:.5f}; "
            f"refined={candidate.continuous_score:.5f}"
        )
    print(f"Best Rwp: {result.fit_statistics['r_wp']:.5f}")
    print(f"Convergence: {result.convergence['classification']}")
    print(f"Refined CIF: {args.output}")


if __name__ == "__main__":
    main()
