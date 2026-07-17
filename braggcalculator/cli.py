"""Command-line entry points for experimental diffraction diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .dataset import DiffractionDataset
from .radiation import nist_copper_ka_spectrum
from .session import RefinementPolicy, RefinementSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bragg-diagnose",
        description="Refine plausible crystal models against one powder pattern.",
    )
    parser.add_argument("data", type=Path, help="two- or three-column XY/XYE data")
    parser.add_argument("--model", type=Path, action="append", required=True, help="candidate CIF")
    parser.add_argument("--name", action="append", help="candidate name; repeat with --model")
    parser.add_argument("--wavelength", type=float, required=True, help="wavelength in angstroms")
    parser.add_argument(
        "--copper-ka-spectrum",
        action="store_true",
        help="use the NIST six-line Cu K-alpha emission spectrum",
    )
    parser.add_argument(
        "--weight-column", action="store_true", help="third column is weight, not sigma"
    )
    parser.add_argument("--lower", type=float)
    parser.add_argument("--upper", type=float)
    parser.add_argument(
        "--coordinates", action="store_true", help="release symmetry-compatible coordinates"
    )
    parser.add_argument(
        "--occupancy-mode",
        choices=("fixed", "composition", "vacancy"),
        default="fixed",
        help="shared-site occupancy policy",
    )
    parser.add_argument("--b-iso", action="store_true", help="release positive orbit Biso values")
    parser.add_argument(
        "--u-aniso",
        action="store_true",
        help="release positive-definite site-symmetry-compatible U tensors",
    )
    parser.add_argument("--occupancy-restraint", type=float, default=1.0)
    parser.add_argument("--b-iso-restraint", type=float, default=0.1)
    parser.add_argument("--u-aniso-restraint", type=float, default=0.1)
    parser.add_argument(
        "--restraints",
        type=Path,
        help="JSON file containing composition, bond, angle, and minimum-distance restraints",
    )
    parser.add_argument("--structural-restraint-weight", type=float, default=1.0)
    parser.add_argument("--quick", action="store_true", help="use the shorter validation recipe")
    parser.add_argument(
        "--legacy-profile", action="store_true", help="use the original symmetric pseudo-Voigt"
    )
    parser.add_argument(
        "--no-axial-asymmetry",
        action="store_true",
        help="disable the empirical low-angle axial-divergence tail",
    )
    parser.add_argument("--goniometer-radius-mm", type=float)
    parser.add_argument("--specimen-displacement-mm", type=float, default=0.0)
    parser.add_argument("--refine-specimen-displacement", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("bragg-report.html"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    metadata = {}
    if args.copper_ka_spectrum:
        metadata["wavelength_components"] = nist_copper_ka_spectrum()
    if args.goniometer_radius_mm is not None:
        metadata["instrument"] = {"goniometer_radius_mm": args.goniometer_radius_mm}
    dataset = DiffractionDataset.from_xye(
        args.data,
        wavelength=args.wavelength,
        third_column="weight" if args.weight_column else "sigma",
        metadata=metadata,
    )
    if (args.lower is None) != (args.upper is None):
        raise SystemExit("--lower and --upper must be supplied together")
    if args.lower is not None:
        dataset = dataset.select_range(args.lower, args.upper)
    names = args.name
    if names is not None and len(names) != len(args.model):
        raise SystemExit("repeat --name exactly once per --model")
    restraints = None
    if args.restraints is not None:
        restraints = json.loads(args.restraints.read_text(encoding="utf-8"))
    policy = (
        RefinementPolicy.quick(
            refine_coordinates=args.coordinates,
            occupancy_mode=args.occupancy_mode,
            refine_b_iso=args.b_iso,
            refine_u_aniso=args.u_aniso,
        )
        if args.quick
        else RefinementPolicy.cautious(
            refine_coordinates=args.coordinates,
            occupancy_mode=args.occupancy_mode,
            refine_b_iso=args.b_iso,
            refine_u_aniso=args.u_aniso,
        )
    )
    policy = replace(
        policy,
        profile_model="legacy" if args.legacy_profile else "tch",
        axial_asymmetry=not args.no_axial_asymmetry,
        occupancy_restraint=args.occupancy_restraint,
        b_iso_restraint=args.b_iso_restraint,
        u_aniso_restraint=args.u_aniso_restraint,
        structural_restraints=restraints,
        structural_restraint_weight=args.structural_restraint_weight,
        goniometer_radius_mm=args.goniometer_radius_mm,
        specimen_displacement_mm=args.specimen_displacement_mm,
        refine_specimen_displacement=args.refine_specimen_displacement,
    )
    session = RefinementSession(dataset, args.model, names=names)
    result = session.run(policy)
    session.write_html(result, args.output)
    print(f"wrote {args.output}")
    print(result.conclusion)
    for candidate in result.candidates:
        print(f"{candidate.name}: Rwp={candidate.r_wp:.5f}, chi2={candidate.chi_squared:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
