#!/usr/bin/env python3
"""Benchmark validated BraggCalculator line patterns against pymatgen.

Two timings are reported:

* ``cached`` reuses BraggCalculator's discrete crystal/HKL topology, the normal
  mode for repeated calculations and refinement.
* ``end_to_end`` includes calculator construction, symmetry preprocessing, HKL
  enumeration and the line calculation.

Every timed case is first checked for numerical agreement with pymatgen.
Reported values are medians of independent ``timeit`` repeats.
Use ``--require-speedup 1`` to fail unless every result is faster.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import timeit
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.reference_cases import reference_structures  # noqa: E402
from braggcalculator import BraggCalculator  # noqa: E402


def median_seconds(function, number: int, repeat: int) -> float:
    return statistics.median(timeit.repeat(function, number=number, repeat=repeat)) / number


def benchmark_case(name, structure, number, repeat):
    calculator = BraggCalculator().load(structure)
    oracle = XRDCalculator(wavelength=calculator.wavelength)
    actual_x, actual_y = calculator.line_pattern(scaled=True)
    expected = oracle.get_pattern(
        structure, two_theta_range=calculator.two_theta_range, scaled=True
    )
    np.testing.assert_allclose(actual_x, expected.x, rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual_y, expected.y, rtol=1e-10, atol=1e-10)

    cached = median_seconds(lambda: calculator.line_pattern(scaled=True), number, repeat)
    pymatgen_time = median_seconds(
        lambda: oracle.get_pattern(
            structure, two_theta_range=calculator.two_theta_range, scaled=True
        ),
        number,
        repeat,
    )
    end_to_end = median_seconds(
        lambda: BraggCalculator().load(structure).line_pattern(scaled=True),
        max(1, number // 5),
        repeat,
    )
    return {
        "case": name,
        "sites": len(structure),
        "peaks": len(actual_x),
        "cached_seconds": cached,
        "end_to_end_seconds": end_to_end,
        "pymatgen_seconds": pymatgen_time,
        "cached_speedup": pymatgen_time / cached,
        "end_to_end_speedup": pymatgen_time / end_to_end,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--require-speedup", type=float, default=1.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.number <= 0 or args.repeat <= 0 or args.require_speedup <= 0:
        parser.error("number, repeat, and require-speedup must be positive")

    results = [
        benchmark_case(name, structure, args.number, args.repeat)
        for name, structure in reference_structures().items()
    ]
    print(
        f"{'case':<20} {'sites':>6} {'peaks':>6} {'cached ms':>11} "
        f"{'e2e ms':>10} {'pymatgen ms':>12} {'cached x':>10} {'e2e x':>8}"
    )
    for result in results:
        print(
            f"{result['case']:<20} {result['sites']:>6d} {result['peaks']:>6d} "
            f"{1e3 * result['cached_seconds']:>11.3f} "
            f"{1e3 * result['end_to_end_seconds']:>10.3f} "
            f"{1e3 * result['pymatgen_seconds']:>12.3f} "
            f"{result['cached_speedup']:>10.2f} {result['end_to_end_speedup']:>8.2f}"
        )

    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {package: version(package) for package in ("numpy", "pymatgen", "spglib")},
        "number": args.number,
        "repeat": args.repeat,
        "results": results,
    }
    if args.json:
        args.json.write_text(json.dumps(metadata, indent=2) + "\n")

    failures = [
        result
        for result in results
        if min(result["cached_speedup"], result["end_to_end_speedup"]) < args.require_speedup
    ]
    if failures:
        names = ", ".join(result["case"] for result in failures)
        raise SystemExit(f"Required {args.require_speedup:.2f}x speedup was not met by: {names}")


if __name__ == "__main__":
    main()
