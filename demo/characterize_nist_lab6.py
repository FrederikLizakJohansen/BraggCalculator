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
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "nist_lab6_report.html"


def characterize(output: Path):
    metadata = {
        "sample": "NIST SRM 660c LaB6",
        "scan": "100a",
        "wavelength_components": [(1.5405929, 2.0), (1.5444274, 1.0)],
        "reference_lattice_angstrom": 4.156826,
        "source": "ark:/88434/mds2-2315",
    }
    dataset = DiffractionDataset.from_xye(
        ROOT / "data/nist_srm660c_100a_20-50.xye",
        wavelength=1.5405929,
        metadata=metadata,
    )
    # Corrected NIST coordinates are used, so zero shift is deliberately fixed.
    policy = RefinementPolicy(
        stages=(
            OptimizationStage("scale/background", ("scale", "background"), 80, 0.03),
            OptimizationStage("profile/lattice", ("profile", "lattice"), 120, 0.015),
            OptimizationStage(
                "joint", ("scale", "background", "profile", "lattice"), 160, 0.005
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
    print(f"wrote {args.output}")
    print(f"Rwp: {candidate.r_wp:.5f}")
    print(f"refined a: {refined_a:.7f} A")
    print(f"certified a: {reference_a:.7f} A")
    print(f"difference: {refined_a - reference_a:+.3e} A")
    print(f"recommendation: {candidate.recommendation}")


if __name__ == "__main__":
    main()
