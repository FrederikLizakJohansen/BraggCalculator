#!/usr/bin/env python3
"""Measure end-to-end diffraction throughput on the frozen COD CIF corpus.

Each selected structure is parsed before timing, then evaluated exactly once
per selected radiation mode by BraggCalculator and pymatgen.  The two timed
outputs are also compared, so a performance result is only valid when the
implementations agree on that same heterogeneous workload.
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
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.cif_corpus import CorpusCase, load_corpus, sha256_file  # noqa: E402
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


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _time_once(function: Callable[[], object], synchronize: Callable[[], None]):
    synchronize()
    start = time.perf_counter()
    value = function()
    synchronize()
    return value, time.perf_counter() - start


def _comparison_metrics(
    actual,
    expected,
    *,
    position_atol: float,
    intensity_atol: float,
) -> dict:
    actual_position = np.asarray(_as_numpy(actual[0]), dtype=float)
    actual_intensity = np.asarray(_as_numpy(actual[1]), dtype=float)
    expected_position = np.asarray(expected.x, dtype=float)
    expected_intensity = np.asarray(expected.y, dtype=float)
    peak_count_matches = actual_position.shape == expected_position.shape
    if peak_count_matches:
        max_position_error = float(
            np.max(np.abs(actual_position - expected_position), initial=0.0)
        )
        max_intensity_error = float(
            np.max(np.abs(actual_intensity - expected_intensity), initial=0.0)
        )
    else:
        max_position_error = None
        max_intensity_error = None
    passed = bool(
        peak_count_matches
        and max_position_error <= position_atol
        and max_intensity_error <= intensity_atol
    )
    return {
        "braggcalculator_peaks": int(actual_position.size),
        "pymatgen_peaks": int(expected_position.size),
        "max_position_error_deg": max_position_error,
        "max_intensity_error_percent": max_intensity_error,
        "passed": passed,
    }


def benchmark_case(
    case: CorpusCase,
    structure: Structure,
    mode: str,
    *,
    calculator: BraggCalculator,
    oracle,
    bragg_first: bool,
    synchronize: Callable[[], None] = lambda: None,
    position_atol: float = 1e-8,
    intensity_atol: float = 1e-7,
    parser_warnings: Sequence[str] = (),
) -> dict:
    """Time each implementation once for one parsed structure and mode."""

    def run_braggcalculator():
        return calculator.load(structure).line_pattern(scaled=True)

    def run_pymatgen():
        return oracle.get_pattern(
            structure,
            two_theta_range=calculator.two_theta_range,
            scaled=True,
        )

    if bragg_first:
        actual, bragg_seconds = _time_once(run_braggcalculator, synchronize)
        expected, pymatgen_seconds = _time_once(run_pymatgen, lambda: None)
        timing_order = ["braggcalculator", "pymatgen"]
    else:
        expected, pymatgen_seconds = _time_once(run_pymatgen, lambda: None)
        actual, bragg_seconds = _time_once(run_braggcalculator, synchronize)
        timing_order = ["pymatgen", "braggcalculator"]

    comparison = _comparison_metrics(
        actual,
        expected,
        position_atol=position_atol,
        intensity_atol=intensity_atol,
    )
    return {
        "id": case.identifier,
        "mode": mode,
        "source_url": case.source_url,
        "sha256": case.sha256,
        "crystal_system": case.crystal_system,
        "manifest_space_group_number": case.space_group_number,
        "detected_space_group_number": calculator._symm["spacegroup_number"],
        "disordered": case.disordered,
        "parser_warnings": list(parser_warnings),
        "formula": structure.composition.formula,
        "input_sites": len(structure),
        "reduced_sites": len(calculator._symm["structure"]),
        "timing_order": timing_order,
        "braggcalculator_seconds": bragg_seconds,
        "pymatgen_seconds": pymatgen_seconds,
        "speedup": pymatgen_seconds / bragg_seconds,
        **comparison,
    }


def summarize_results(results: Sequence[dict]) -> dict:
    """Aggregate a varied workload using the ratio of total runtimes."""
    if not results:
        raise ValueError("cannot summarize an empty benchmark")
    bragg_seconds = sum(result["braggcalculator_seconds"] for result in results)
    pymatgen_seconds = sum(result["pymatgen_seconds"] for result in results)
    speedups = [result["speedup"] for result in results]
    peak_count = sum(result["braggcalculator_peaks"] for result in results)
    return {
        "structure_mode_evaluations": len(results),
        "unique_structures": len({result["id"] for result in results}),
        "braggcalculator_total_seconds": bragg_seconds,
        "pymatgen_total_seconds": pymatgen_seconds,
        "total_corpus_speedup": pymatgen_seconds / bragg_seconds,
        "braggcalculator_evaluations_per_second": len(results) / bragg_seconds,
        "pymatgen_evaluations_per_second": len(results) / pymatgen_seconds,
        "braggcalculator_lines_per_second": peak_count / bragg_seconds,
        "pymatgen_lines_per_second": peak_count / pymatgen_seconds,
        "median_per_structure_speedup": statistics.median(speedups),
        "per_structure_speedup_quartiles": [
            float(np.percentile(speedups, 25)),
            float(np.percentile(speedups, 75)),
        ],
        "passed": all(result["passed"] for result in results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "cif_validation" / "manifest.json",
    )
    parser.add_argument("--mode", choices=("xray", "neutron", "both"), default="both")
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--device")
    parser.add_argument("--position-atol", type=float, default=1e-8)
    parser.add_argument("--intensity-atol", type=float, default=1e-7)
    parser.add_argument("--hardware-label")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.position_atol < 0 or args.intensity_atol < 0:
        raise SystemExit("comparison tolerances must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    manifest, cases = load_corpus(args.manifest)
    if args.limit is not None:
        cases = cases[: args.limit]
    modes = ("xray", "neutron") if args.mode == "both" else (args.mode,)

    parsed = []
    for case in cases:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            structure = Structure.from_file(case.path)
        parser_warnings = sorted({str(item.message) for item in caught_warnings})
        parsed.append((case, structure, parser_warnings))

    model = cpu_model()
    device = args.device or ("cuda" if args.backend == "torch" else "cpu")
    gpu_metadata = None

    def synchronize() -> None:
        return None

    if args.backend == "numpy":
        if device != "cpu":
            raise SystemExit("the NumPy backend only supports --device cpu")
        backend = NumpyBackend()
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
        backend = TorchBackend(device=device, dtype=torch.float64)

    calculators = {mode: BraggCalculator(mode=mode, backend=backend) for mode in modes}
    oracle_types = {"xray": XRDCalculator, "neutron": NDCalculator}
    oracles = {
        mode: oracle_types[mode](wavelength=calculators[mode].wavelength) for mode in modes
    }

    # Warm libraries and, for CUDA, initialize kernels with a structure outside
    # the measured COD corpus. No measured structure is repeated.
    warmup_structure = Structure.from_file(ROOT / "demo" / "NaCl.cif")
    for mode in modes:
        calculators[mode].load(warmup_structure).line_pattern(scaled=True)
        synchronize()
        oracles[mode].get_pattern(
            warmup_structure,
            two_theta_range=calculators[mode].two_theta_range,
            scaled=True,
        )

    results = []
    print(
        f"{'id':<12} {'system':<12} {'mode':<8} {'sites':>7} {'peaks':>7} "
        f"{'Bragg / s':>12} {'pymatgen / s':>13} {'speedup':>10} {'status':>8}"
    )
    for workload_index, (case, structure, parser_warnings) in enumerate(parsed):
        for mode_index, mode in enumerate(modes):
            result = benchmark_case(
                case,
                structure,
                mode,
                calculator=calculators[mode],
                oracle=oracles[mode],
                bragg_first=(workload_index * len(modes) + mode_index) % 2 == 0,
                synchronize=synchronize,
                position_atol=args.position_atol,
                intensity_atol=args.intensity_atol,
                parser_warnings=parser_warnings,
            )
            results.append(result)
            print(
                f"{case.identifier:<12} {case.crystal_system:<12} {mode:<8} "
                f"{result['input_sites']:>7d} {result['braggcalculator_peaks']:>7d} "
                f"{result['braggcalculator_seconds']:>12.5f} "
                f"{result['pymatgen_seconds']:>13.5f} "
                f"{result['speedup']:>9.2f}x "
                f"{'PASS' if result['passed'] else 'FAIL':>8}"
            )

    resolved_manifest = args.manifest.resolve()
    try:
        manifest_label = str(resolved_manifest.relative_to(ROOT))
    except ValueError:
        manifest_label = str(resolved_manifest)
    summaries = {
        mode: summarize_results([result for result in results if result["mode"] == mode])
        for mode in modes
    }
    summaries["all"] = summarize_results(results)
    package_names = ["braggcalculator", "numpy", "pymatgen", "spglib"]
    if args.backend == "torch":
        package_names.append("torch")
    output = {
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
        "timing_protocol": {
            "timed_evaluations_per_structure_and_mode": 1,
            "cif_parsing_timed": False,
            "braggcalculator_scope": "load(structure).line_pattern(scaled=True)",
            "pymatgen_scope": "get_pattern(structure, scaled=True)",
            "calculators_reused_across_structures": True,
            "implementation_order": "alternated for consecutive structure-mode pairs",
            "warmup": "demo/NaCl.cif; outside the measured COD corpus",
            "cuda_synchronization": "immediately before and after each timed call",
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "versions": {package: version(package) for package in package_names},
        "corpus": {
            "name": manifest["name"],
            "manifest": manifest_label,
            "manifest_sha256": sha256_file(args.manifest),
            "source": manifest["source"],
            "case_count": len(cases),
            "limited": args.limit is not None,
        },
        "modes": list(modes),
        "position_atol_deg": args.position_atol,
        "intensity_atol_percent": args.intensity_atol,
        "summaries": summaries,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"total varied-workload speedup: {summaries['all']['total_corpus_speedup']:.2f}x"
    )
    print(f"wrote {args.output}")
    if not summaries["all"]["passed"]:
        raise SystemExit("one or more timed diffraction results failed validation")


if __name__ == "__main__":
    main()
