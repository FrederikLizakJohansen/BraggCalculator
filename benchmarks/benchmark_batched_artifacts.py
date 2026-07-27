#!/usr/bin/env python3
"""Measure device-native batched artifact throughput on synthetic powder lines."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from braggcalculator import (  # noqa: E402
    AmorphousHump,
    BackgroundArtifacts,
    CalibrationArtifacts,
    DetectorArtifacts,
    IntensityArtifacts,
    NoiseArtifacts,
    PeakProfileArtifacts,
    SimulationArtifacts,
    SpuriousPeakArtifacts,
    apply_peak_artifact_batch,
    render_artifact_batch,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(function, *, warmup: int, repeat: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        function()
    synchronize(device)
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        synchronize(device)
        timings.append(time.perf_counter() - start)
    return timings


def artifacts(*, full: bool) -> SimulationArtifacts:
    if not full:
        return SimulationArtifacts(
            calibration=CalibrationArtifacts(
                zero_shift=(-0.01, 0.01),
                axis_scale=(0.995, 1.005),
                peak_jitter_std=(0.0, 0.005),
            ),
            intensity=IntensityArtifacts(
                scale=(0.8, 1.2),
                peak_jitter=(0.9, 1.1),
                peak_dropout_probability=0.05,
            ),
        )
    return SimulationArtifacts(
        calibration=CalibrationArtifacts(
            zero_shift=(-0.01, 0.01),
            axis_scale=(0.995, 1.005),
            peak_jitter_std=(0.0, 0.005),
        ),
        intensity=IntensityArtifacts(
            scale=(0.8, 1.2),
            peak_jitter=(0.9, 1.1),
            peak_dropout_probability=0.05,
        ),
        profile=PeakProfileArtifacts(
            model="tch",
            caglioti_u=(0.001, 0.004),
            caglioti_w=(0.002, 0.006),
            lorentzian_x=(0.001, 0.004),
            lorentzian_y=(0.001, 0.004),
            crystallite_size_nm=(20.0, 100.0),
            microstrain=(0.0, 0.001),
        ),
        background=BackgroundArtifacts(
            constant=(0.0, 0.03),
            linear_slope=(-0.002, 0.002),
            chebyshev_coefficients=(0.01, 0.002, -0.001),
            amorphous_humps=(
                AmorphousHump(
                    center=(2.0, 5.0),
                    fwhm=(0.3, 1.0),
                    height=(0.01, 0.08),
                    eta=(0.0, 0.4),
                ),
            ),
        ),
        spurious_peaks=SpuriousPeakArtifacts(
            count=(0, 3), intensity=(0.005, 0.05), fwhm=(0.03, 0.1)
        ),
        noise=NoiseArtifacts(
            gaussian_std=(0.0, 0.003),
            correlated_std=(0.0, 0.003),
            correlation_length=(0.02, 0.1),
            poisson_count_scale=(5000.0, 20000.0),
        ),
        detector=DetectorArtifacts(
            random_mask_probability=0.005,
            excluded_ranges=((9.7, 9.8),),
            saturation_level=2.0,
        ),
        normalize_signal=True,
        final_normalize=True,
        domain="q",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--peaks", type=int, default=512)
    parser.add_argument("--grid-points", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--peak-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.peaks, args.grid_points, args.repeat) <= 0:
        parser.error("batch-size, peaks, grid-points, and repeat must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is false")
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device=device).manual_seed(2026)
    positions = 0.2 + 9.6 * torch.rand(
        (args.batch_size, args.peaks),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    positions, _ = torch.sort(positions, dim=-1)
    intensities = torch.rand(
        positions.shape, dtype=dtype, device=device, generator=generator
    )
    peak_mask = torch.ones_like(positions, dtype=torch.bool)
    grid = torch.linspace(
        0.1, 10.0, args.grid_points, dtype=dtype, device=device
    )
    configuration = artifacts(full=not args.peak_only)
    run_generator = torch.Generator(device=device).manual_seed(17)

    if args.peak_only:
        def function():
            return apply_peak_artifact_batch(
                positions,
                intensities,
                peak_mask=peak_mask,
                artifacts=configuration,
                generator=run_generator,
            )

        output_values = args.batch_size * args.peaks
        mode = "peak"
    else:
        def function():
            return render_artifact_batch(
                positions,
                intensities,
                peak_mask=peak_mask,
                grid=grid,
                artifacts=configuration,
                wavelength=1.5406,
                generator=run_generator,
            )

        output_values = args.batch_size * args.grid_points
        mode = "dense"

    timings = time_call(
        function, warmup=args.warmup, repeat=args.repeat, device=device
    )
    median = statistics.median(timings)
    result = {
        "mode": mode,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "peaks": args.peaks,
        "grid_points": args.grid_points,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "median_seconds": median,
        "patterns_per_second": args.batch_size / median,
        "output_values_per_second": output_values / median,
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
