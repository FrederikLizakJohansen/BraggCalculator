#!/usr/bin/env python3
"""Refine the vendored NIST SRM 660c excerpt and write an HTML diagnostic report."""

from __future__ import annotations

import argparse
from pathlib import Path

from braggcalculator import (
    DiffractionDataset,
    OptimizationStage,
    RefinementPolicy,
    RefinementSession,
    nist_copper_ka_spectrum,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "nist_lab6_report.html"
HISTORICAL_EXCERPT_RWP = 0.19877
HISTORICAL_EXCERPT_LATTICE = 4.1549091


def characterize(output: Path):
    metadata = {
        "sample": "NIST SRM 660c LaB6",
        "scan": "100a",
        "wavelength_components": nist_copper_ka_spectrum(),
        "reference_lattice_angstrom": 4.156826,
        "reference_lattice_expanded_uncertainty_angstrom": 0.000080,
        "source": "ark:/88434/mds2-2315",
        "instrument": {
            "geometry": "Bragg-Brentano",
            "goniometer_radius_mm": 217.5,
            "post_specimen_monochromator": "graphite analyzer",
            "reported_specimen_displacement_mm": -0.07877,
        },
        "model_limitations": [
            "The compact split pseudo-Voigt does not reproduce the graphite-analyzer "
            "passband or the complete NIST fundamental-parameters convolution."
        ],
    }
    dataset = DiffractionDataset.from_xye(
        ROOT / "data/nist_srm660c_100a_full.xye",
        wavelength=1.5405925,
        metadata=metadata,
    )
    # The supplied angular scale is calibrated, so zero shift is fixed. The
    # specimen displacement reported in the pdCIF is applied explicitly.
    policy = RefinementPolicy(
        background_degree=6,
        profile_model="tch",
        axial_asymmetry=True,
        goniometer_radius_mm=217.5,
        specimen_displacement_mm=-0.07877,
        diagnostic_points=24,
        stages=(
            OptimizationStage("scale/background", ("scale", "background"), 80, 0.03),
            OptimizationStage("profile/lattice", ("profile", "lattice"), 180, 0.008),
            OptimizationStage(
                "joint", ("scale", "background", "profile", "lattice"), 300, 0.002
            ),
        )
    )
    session = RefinementSession(
        dataset,
        [ROOT / "data/LaB6_srm660c.cif"],
        names=["LaB6 SRM 660c"],
    )
    result = session.run(policy)
    session.write_html(result, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = characterize(args.output)
    candidate = result.candidates[0]
    refined_a = candidate.physical_parameters["lattice"][0][0]
    reference_a = result.dataset.metadata["reference_lattice_angstrom"]
    expanded_uncertainty = result.dataset.metadata[
        "reference_lattice_expanded_uncertainty_angstrom"
    ]
    difference = refined_a - reference_a
    print(f"wrote {args.output}")
    print(
        "historical limited-scan baseline: "
        f"Rwp={HISTORICAL_EXCERPT_RWP:.5f}, a={HISTORICAL_EXCERPT_LATTICE:.7f} A"
    )
    print(f"instrument-aware full-scan Rwp: {candidate.r_wp:.5f}")
    print(f"refined a: {refined_a:.7f} A")
    print(f"certified a: {reference_a:.7f} A")
    print(f"difference: {difference:+.3e} A")
    print(
        "difference / certified expanded uncertainty: "
        f"{abs(difference) / expanded_uncertainty:.2f}"
    )
    if abs(difference) > expanded_uncertainty:
        print("validation status: improved, but outside the certified uncertainty interval")
    print(f"recommendation: {candidate.recommendation}")


if __name__ == "__main__":
    main()
