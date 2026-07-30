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
import time
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.scaling_cases import scaling_cases  # noqa: E402
from braggcalculator import BraggCalculator  # noqa: E402
from braggcalculator.backends import NumpyBackend, TorchBackend  # noqa: E402


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


def interleaved_timing_samples(
    functions: dict[str, Callable[[], object]],
    *,
    numbers: dict[str, int],
    repeat: int,
    synchronizers: dict[str, Callable[[], None]] | None = None,
) -> dict[str, list[float]]:
    """Time methods in a rotating order to limit thermal and frequency bias."""
    names = tuple(functions)
    samples = {name: [] for name in names}
    synchronizers = synchronizers or {name: lambda: None for name in names}
    for repeat_index in range(repeat):
        offset = repeat_index % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            synchronizers[name]()
            start = time.perf_counter()
            for _ in range(numbers[name]):
                functions[name]()
            synchronizers[name]()
            elapsed = time.perf_counter() - start
            samples[name].append(elapsed / numbers[name])
    return samples


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def benchmark_case(
    case,
    *,
    number: int,
    repeat: int,
    backend_factory: Callable[[], object] = NumpyBackend,
    synchronize: Callable[[], None] = lambda: None,
) -> dict:
    structure = case.structure
    calculator = BraggCalculator(backend=backend_factory()).load(structure)
    oracle = XRDCalculator(wavelength=calculator.wavelength)
    actual_x, actual_y = calculator.line_pattern(scaled=True)
    synchronize()
    actual_x = _as_numpy(actual_x)
    actual_y = _as_numpy(actual_y)
    expected = oracle.get_pattern(
        structure,
        two_theta_range=calculator.two_theta_range,
        scaled=True,
    )
    np.testing.assert_allclose(actual_x, expected.x, rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual_y, expected.y, rtol=1e-10, atol=1e-10)

    samples = interleaved_timing_samples(
        {
            "cached": lambda: calculator.line_pattern(scaled=True),
            "end_to_end": lambda: BraggCalculator(backend=backend_factory())
            .load(structure)
            .line_pattern(scaled=True),
            "pymatgen": lambda: oracle.get_pattern(
                structure,
                two_theta_range=calculator.two_theta_range,
                scaled=True,
            ),
        },
        numbers={"cached": number, "end_to_end": max(1, number // 5), "pymatgen": number},
        repeat=repeat,
        synchronizers={
            "cached": synchronize,
            "end_to_end": synchronize,
            "pymatgen": lambda: None,
        },
    )
    cached_samples = samples["cached"]
    end_to_end_samples = samples["end_to_end"]
    pymatgen_samples = samples["pymatgen"]
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
            np.max(np.abs(actual_x - np.asarray(expected.x)), initial=0.0)
        ),
        "max_intensity_error_percent": float(
            np.max(np.abs(actual_y - np.asarray(expected.y)), initial=0.0)
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
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--device")
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
    device = args.device or ("cuda" if args.backend == "torch" else "cpu")
    gpu_metadata = None

    def synchronize() -> None:
        return None

    if args.backend == "numpy":
        if device != "cpu":
            raise SystemExit("the NumPy backend only supports --device cpu")
        backend_factory = NumpyBackend
        execution_label = model
    else:
        if TorchBackend is None:
            raise SystemExit("install BraggCalculator with the torch extra")
        import torch

        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
        if torch_device.type == "cuda":
            device_index = torch_device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            gpu_metadata = {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "cuda_runtime": torch.version.cuda,
            }
            def synchronize() -> None:
                torch.cuda.synchronize(torch_device)

            execution_label = f"{properties.name}; oracle on {model} CPU"
        else:
            execution_label = f"PyTorch {device}; oracle on {model} CPU"

        def backend_factory():
            return TorchBackend(device=device, dtype=torch.float64)

    package_names = ["braggcalculator", "numpy", "pymatgen", "spglib"]
    if args.backend == "torch":
        package_names.append("torch")
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "hardware": {
            "label": args.hardware_label or execution_label,
            "cpu_model": model,
            "logical_cpus": os.cpu_count(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "gpu": gpu_metadata,
        },
        "execution": {
            "braggcalculator_backend": args.backend,
            "braggcalculator_device": device,
            "braggcalculator_dtype": "float64",
            "pymatgen_device": "cpu",
        },
        "site_count_definition": (
            "Number of crystallographic sites in the Structure supplied to both calculators "
            "before BraggCalculator primitive-cell reduction"
        ),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "versions": {package: version(package) for package in package_names},
        "number": args.number,
        "repeat": args.repeat,
        "results": [],
    }

    for case in cases:
        result = benchmark_case(
            case,
            number=args.number,
            repeat=args.repeat,
            backend_factory=backend_factory,
            synchronize=synchronize,
        )
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
