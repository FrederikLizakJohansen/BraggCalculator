"""Command-line entry points for experimental diffraction diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import DiffractionDataset
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
    parser.add_argument("--weight-column", action="store_true", help="third column is weight, not sigma")
    parser.add_argument("--lower", type=float)
    parser.add_argument("--upper", type=float)
    parser.add_argument("--coordinates", action="store_true", help="release symmetry-compatible coordinates")
    parser.add_argument("--quick", action="store_true", help="use the shorter validation recipe")
    parser.add_argument("--output", type=Path, default=Path("bragg-report.html"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dataset = DiffractionDataset.from_xye(
        args.data,
        wavelength=args.wavelength,
        third_column="weight" if args.weight_column else "sigma",
    )
    if (args.lower is None) != (args.upper is None):
        raise SystemExit("--lower and --upper must be supplied together")
    if args.lower is not None:
        dataset = dataset.select_range(args.lower, args.upper)
    names = args.name
    if names is not None and len(names) != len(args.model):
        raise SystemExit("repeat --name exactly once per --model")
    policy = (
        RefinementPolicy.quick(refine_coordinates=args.coordinates)
        if args.quick
        else RefinementPolicy.cautious(refine_coordinates=args.coordinates)
    )
    session = RefinementSession(dataset, args.model, names=names)
    result = session.run(policy)
    session.write_html(result, args.output)
    print(f"wrote {args.output}")
    print(result.conclusion)
    for candidate in result.candidates:
        print(
            f"{candidate.name}: Rwp={candidate.r_wp:.5f}, "
            f"chi2={candidate.chi_squared:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
