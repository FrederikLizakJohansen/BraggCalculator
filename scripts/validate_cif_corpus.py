#!/usr/bin/env python3
"""Validate a frozen CIF corpus against pymatgen X-ray and neutron patterns."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.cif_corpus import compare_case, load_corpus, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "cif_validation" / "manifest.json",
    )
    parser.add_argument("--mode", choices=("xray", "neutron", "both"), default="both")
    parser.add_argument("--position-atol", type=float, default=1e-8)
    parser.add_argument("--intensity-atol", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.position_atol < 0 or args.intensity_atol < 0:
        raise SystemExit("comparison tolerances must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    manifest, cases = load_corpus(args.manifest)
    resolved_manifest = args.manifest.resolve()
    try:
        manifest_label = str(resolved_manifest.relative_to(ROOT))
    except ValueError:
        manifest_label = str(resolved_manifest)
    if args.limit is not None:
        cases = cases[: args.limit]
    modes = ("xray", "neutron") if args.mode == "both" else (args.mode,)
    results = []
    print(
        f"{'id':<12} {'system':<12} {'mode':<8} {'sites':>7} {'peaks':>7} "
        f"{'max |d2theta|':>16} {'max |dI|':>14} {'status':>8}"
    )
    for case in cases:
        for mode in modes:
            try:
                result = compare_case(
                    case,
                    mode,
                    position_atol=args.position_atol,
                    intensity_atol=args.intensity_atol,
                )
            except Exception as error:  # retain every corpus failure in the artifact
                result = {
                    "id": case.identifier,
                    "mode": mode,
                    "source_url": case.source_url,
                    "sha256": case.sha256,
                    "crystal_system": case.crystal_system,
                    "manifest_space_group_number": case.space_group_number,
                    "disordered": case.disordered,
                    "passed": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            results.append(result)
            dx = result.get("max_position_error_deg")
            dy = result.get("max_intensity_error_percent")
            print(
                f"{case.identifier:<12} {case.crystal_system:<12} {mode:<8} "
                f"{result.get('input_sites', 0):>7d} {result.get('braggcalculator_peaks', 0):>7d} "
                f"{dx if dx is not None else float('nan'):>16.3e} "
                f"{dy if dy is not None else float('nan'):>14.3e} "
                f"{'PASS' if result['passed'] else 'FAIL':>8}"
            )

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    failures = [result for result in results if not result["passed"]]
    parser_warning_cases = {
        result["id"] for result in results if result.get("parser_warnings")
    }
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            package: version(package)
            for package in ("braggcalculator", "numpy", "pymatgen", "spglib")
        },
        "corpus": {
            "name": manifest["name"],
            "manifest": manifest_label,
            "manifest_sha256": sha256_file(args.manifest),
            "source": manifest["source"],
            "case_count": len(cases),
        },
        "modes": list(modes),
        "position_atol_deg": args.position_atol,
        "intensity_atol_percent": args.intensity_atol,
        "comparison_count": len(results),
        "passed_count": len(results) - len(failures),
        "failed_count": len(failures),
        "parser_warning_case_count": len(parser_warning_cases),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}")
    if failures:
        raise SystemExit(f"{len(failures)} of {len(results)} corpus comparisons failed")


if __name__ == "__main__":
    main()
