#!/usr/bin/env python3
"""Collect validated atom-count and symmetry scaling measurements.

The P1 series increases the number of irreducible sites at approximately fixed
atomic density. The NaCl series increases the input supercell while retaining
a two-site primitive cell, isolating the benefit and cost of symmetry
reduction. Every structure must match pymatgen before either implementation is
timed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import timeit
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.scaling_cases import scaling_cases  # noqa: E402
from braggcalculator import BraggCalculator  # noqa: E402


def cpu_model() -> str:
    """Return a useful CPU identifier without requiring a platform utility."""
    processor = platform.processor().strip()
    if processor:
        return processor
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.machine()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def timing_samples(function, *, number: int, repeat: int) -> list[float]:
    """Return independent per-call timeit samples in seconds."""
    return [sample / number for sample in timeit.repeat(function, number=number, repeat=repeat)]


def benchmark_case(case, *, number: int, repeat: int) -> dict:
    structure = case.structure
    calculator = BraggCalculator().load(structure)
    oracle = XRDCalculator(wavelength=calculator.wavelength)
    actual_x, actual_y = calculator.line_pattern(scaled=True)
    expected = oracle.get_pattern(
        structure,
        two_theta_range=calculator.two_theta_range,
        scaled=True,
    )
    np.testing.assert_allclose(actual_x, expected.x, rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual_y, expected.y, rtol=1e-10, atol=1e-10)

    cached_samples = timing_samples(
        lambda: calculator.line_pattern(scaled=True),
        number=number,
        repeat=repeat,
    )
    end_to_end_samples = timing_samples(
        lambda: BraggCalculator().load(structure).line_pattern(scaled=True),
        number=max(1, number // 5),
        repeat=repeat,
    )
    pymatgen_samples = timing_samples(
        lambda: oracle.get_pattern(
            structure,
            two_theta_range=calculator.two_theta_range,
            scaled=True,
        ),
        number=number,
        repeat=repeat,
    )
    medians = {
        "cached_seconds": statistics.median(cached_samples),
        "end_to_end_seconds": statistics.median(end_to_end_samples),
        "pymatgen_seconds": statistics.median(pymatgen_samples),
    }
    return {
        "case": case.name,
        "series": case.series,
        "control_value": case.control_value,
        "input_sites": len(structure),
        "reduced_sites": len(calculator._symm["structure"]),
        "space_group": calculator._symm["spacegroup_symbol"],
        "symmetry_operations": len(calculator._symm["symm_rot"]),
        "peaks": len(actual_x),
        "max_position_error_deg": float(
            np.max(np.abs(np.asarray(actual_x) - np.asarray(expected.x)), initial=0.0)
        ),
        "max_intensity_error_percent": float(
            np.max(np.abs(np.asarray(actual_y) - np.asarray(expected.y)), initial=0.0)
        ),
        "samples_seconds": {
            "cached": cached_samples,
            "end_to_end": end_to_end_samples,
            "pymatgen": pymatgen_samples,
        },
        **medians,
        "cached_speedup": medians["pymatgen_seconds"] / medians["cached_seconds"],
        "end_to_end_speedup": medians["pymatgen_seconds"]
        / medians["end_to_end_seconds"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-sites", type=int, nargs="+", default=(4, 8, 16, 32, 64, 128))
    parser.add_argument("--symmetry-factors", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--series", choices=("p1", "symmetry", "both"), default="both")
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--hardware-label")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.number <= 0 or args.repeat <= 0:
        raise SystemExit("number and repeat must be positive")
    p1_counts = tuple(args.p1_sites) if args.series in {"p1", "both"} else ()
    symmetry_factors = (
        tuple(args.symmetry_factors) if args.series in {"symmetry", "both"} else ()
    )
    try:
        cases = scaling_cases(p1_counts, symmetry_factors)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    model = cpu_model()
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "hardware": {
            "label": args.hardware_label or model,
            "cpu_model": model,
            "logical_cpus": os.cpu_count(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "versions": {
            package: version(package)
            for package in ("braggcalculator", "numpy", "pymatgen", "spglib")
        },
        "number": args.number,
        "repeat": args.repeat,
        "results": [],
    }

    for case in cases:
        result = benchmark_case(case, number=args.number, repeat=args.repeat)
        metadata["results"].append(result)
        print(
            f"{result['case']:<16} {result['input_sites']:>5} input sites, "
            f"{result['reduced_sites']:>4} reduced, "
            f"cached {result['cached_speedup']:>8.2f}x, "
            f"end-to-end {result['end_to_end_speedup']:>8.2f}x"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
