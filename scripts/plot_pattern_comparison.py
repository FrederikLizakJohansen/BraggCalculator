#!/usr/bin/env python3
"""Plot broadened BraggCalculator and pymatgen powder patterns.

pymatgen reports powder lines rather than a continuous profile. To compare the
diffraction calculations without introducing an instrument-model difference,
this script applies the same area-normalized Gaussian kernel to the line
positions and integrated intensities from both implementations.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.reference_cases import reference_structures  # noqa: E402
from braggcalculator import BraggCalculator  # noqa: E402


@dataclass(frozen=True)
class PatternMetrics:
    """Numerical agreement recorded alongside one plotted pattern."""

    case: str
    mode: str
    sites: int
    peaks: int
    gaussian_fwhm_deg: float
    max_position_error_deg: float
    max_line_intensity_error_percent: float
    max_profile_error_percent: float
    profile_rmse_percent: float


@dataclass(frozen=True)
class PatternComparison:
    """Arrays and summary metrics needed to render one comparison panel."""

    grid: np.ndarray
    braggcalculator_positions: np.ndarray
    braggcalculator_intensities: np.ndarray
    pymatgen_positions: np.ndarray
    pymatgen_intensities: np.ndarray
    braggcalculator_profile: np.ndarray
    pymatgen_profile: np.ndarray
    metrics: PatternMetrics


def regular_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Construct the same inclusive regular grid convention as BraggCalculator."""
    if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
        raise ValueError("grid bounds must be finite and increasing")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("grid step must be positive and finite")
    intervals = int(np.floor((stop - start) / step + 8 * np.finfo(float).eps))
    return np.linspace(start, start + intervals * step, intervals + 1)


def gaussian_pattern(
    grid: np.ndarray,
    centers: np.ndarray,
    integrated_intensities: np.ndarray,
    fwhm_deg: float,
) -> np.ndarray:
    """Render an area-normalized Gaussian for every powder line."""
    if not np.isfinite(fwhm_deg) or fwhm_deg <= 0:
        raise ValueError("fwhm_deg must be positive and finite")
    sigma = fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    offsets = grid[:, None] - centers[None, :]
    kernels = np.exp(-0.5 * (offsets / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    return kernels @ integrated_intensities


def scale_profile(profile: np.ndarray) -> np.ndarray:
    maximum = float(np.max(profile, initial=0.0))
    if maximum <= 0:
        return np.zeros_like(profile)
    return 100.0 * profile / maximum


def compare_pattern(
    case: str,
    structure,
    *,
    mode: str,
    two_theta_range: tuple[float, float],
    step_deg: float,
    fwhm_deg: float,
    position_atol: float,
    intensity_atol: float,
    profile_atol: float,
) -> PatternComparison:
    """Calculate, broaden, validate, and summarize a single oracle comparison."""
    calculator = BraggCalculator(mode=mode, two_theta_range=two_theta_range).load(structure)
    actual_x, actual_y = calculator.line_pattern(scaled=True)
    actual_x = np.asarray(actual_x, dtype=float)
    actual_y = np.asarray(actual_y, dtype=float)

    oracle_type = XRDCalculator if mode == "xray" else NDCalculator
    oracle = oracle_type(wavelength=calculator.wavelength).get_pattern(
        structure,
        two_theta_range=two_theta_range,
        scaled=True,
    )
    expected_x = np.asarray(oracle.x, dtype=float)
    expected_y = np.asarray(oracle.y, dtype=float)

    np.testing.assert_allclose(actual_x, expected_x, rtol=0, atol=position_atol)
    np.testing.assert_allclose(actual_y, expected_y, rtol=0, atol=intensity_atol)

    grid = regular_grid(*two_theta_range, step_deg)
    actual_profile = scale_profile(gaussian_pattern(grid, actual_x, actual_y, fwhm_deg))
    expected_profile = scale_profile(gaussian_pattern(grid, expected_x, expected_y, fwhm_deg))
    np.testing.assert_allclose(actual_profile, expected_profile, rtol=0, atol=profile_atol)
    profile_difference = actual_profile - expected_profile

    metrics = PatternMetrics(
        case=case,
        mode=mode,
        sites=len(structure),
        peaks=len(actual_x),
        gaussian_fwhm_deg=fwhm_deg,
        max_position_error_deg=float(np.max(np.abs(actual_x - expected_x), initial=0.0)),
        max_line_intensity_error_percent=float(
            np.max(np.abs(actual_y - expected_y), initial=0.0)
        ),
        max_profile_error_percent=float(np.max(np.abs(profile_difference), initial=0.0)),
        profile_rmse_percent=float(np.sqrt(np.mean(profile_difference**2))),
    )
    return PatternComparison(
        grid,
        actual_x,
        actual_y,
        expected_x,
        expected_y,
        actual_profile,
        expected_profile,
        metrics,
    )


def plot_comparisons(
    comparisons_by_fwhm: dict[float, list[PatternComparison]],
    *,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Render line patterns and profiles at every requested broadening."""
    fwhm_values = tuple(comparisons_by_fwhm)
    if not fwhm_values:
        raise ValueError("at least one Gaussian FWHM is required")
    comparisons = comparisons_by_fwhm[fwhm_values[0]]
    rows = len(comparisons)
    columns = 1 + len(fwhm_values)
    plt.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "font.size": 9,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.2, 1.9 * rows + 0.55),
        sharex=True,
        layout="constrained",
    )
    axes = np.asarray(axes).reshape(rows, columns)
    display_names = {
        "NaCl": "NaCl",
        "Si": "Si",
        "SrTiO3": r"SrTiO$_3$",
        "triclinic-SiO2": r"triclinic SiO$_2$",
        "NaKCl-disordered": r"Na$_{0.7}$K$_{0.3}$Cl",
        "P1-40-atom": "40-site P1 cell",
    }

    for index, line_comparison in enumerate(comparisons):
        line_axis = axes[index, 0]

        line_axis.axhline(0.0, color="0.65", linewidth=0.7)
        line_axis.vlines(
            line_comparison.pymatgen_positions,
            0.0,
            line_comparison.pymatgen_intensities,
            color="#D55E00",
            linewidth=1.5,
            label="pymatgen",
        )
        line_axis.vlines(
            line_comparison.braggcalculator_positions,
            0.0,
            -line_comparison.braggcalculator_intensities,
            color="#0072B2",
            linewidth=1.0,
            label="BraggCalculator",
        )
        line_axis.set_ylim(-108, 108)
        line_axis.set_yticks((-100, -50, 0, 50, 100), ("100", "50", "0", "50", "100"))
        line_axis.set_ylabel("Relative intensity (%)")
        line_axis.text(
            0.015,
            0.92,
            "pymatgen",
            color="#D55E00",
            transform=line_axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
        )
        line_axis.text(
            0.015,
            0.08,
            "BraggCalculator",
            color="#0072B2",
            transform=line_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
        )
        line_axis.text(
            0.985,
            0.07,
            f"max |Δ2θ| {line_comparison.metrics.max_position_error_deg:.1e}°\n"
            f"max |ΔI| {line_comparison.metrics.max_line_intensity_error_percent:.1e}",
            transform=line_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="0.3",
        )

        case_name = display_names.get(line_comparison.metrics.case, line_comparison.metrics.case)
        line_axis.text(
            0.5,
            1.025,
            f"{case_name} · {line_comparison.metrics.peaks} peaks",
            transform=line_axis.transAxes,
            ha="center",
            va="bottom",
            fontweight="semibold",
        )

        for column, fwhm_deg in enumerate(fwhm_values, start=1):
            comparison = comparisons_by_fwhm[fwhm_deg][index]
            profile_axis = axes[index, column]
            x = comparison.grid
            expected = comparison.pymatgen_profile
            actual = comparison.braggcalculator_profile
            profile_axis.plot(
                x,
                expected,
                color="#D55E00",
                linewidth=2.4,
                label="pymatgen",
            )
            profile_axis.plot(
                x,
                actual,
                color="#0072B2",
                linewidth=1.25,
                linestyle="--",
                label="BraggCalculator",
            )
            profile_axis.set_ylim(-3, 105)
            profile_axis.grid(axis="y", color="0.92", linewidth=0.6)
            if column == 1:
                profile_axis.set_ylabel("Relative intensity (%)")
            profile_axis.text(
                0.97,
                0.92,
                f"max |Δ| {comparison.metrics.max_profile_error_percent:.1e}",
                transform=profile_axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.2,
                color="0.3",
            )
            profile_axis.fill_between(
                x,
                np.minimum(actual, expected),
                np.maximum(actual, expected),
                color="#009E73",
                alpha=0.5,
                linewidth=0,
            )

    for axis in axes[-1]:
        axis.set_xlabel(r"$2\theta$ (degrees)")
    axes[0, 0].set_title("Lines · no broadening", pad=23)
    for column, fwhm_deg in enumerate(fwhm_values, start=1):
        axes[0, column].set_title(f"Shared FWHM {fwhm_deg:g}°", pad=23)
    legend_handles, legend_labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(
        legend_handles,
        legend_labels,
        loc="outside upper center",
        ncols=2,
        frameon=False,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=300)
    figure.savefig(
        pdf_path,
        metadata={"Title": "Powder-pattern agreement with pymatgen", "CreationDate": None},
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    structures = reference_structures()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(structures),
        default=("NaCl", "SrTiO3", "triclinic-SiO2"),
    )
    parser.add_argument("--mode", choices=("xray", "neutron"), default="xray")
    parser.add_argument("--two-theta-min", type=float, default=10.0)
    parser.add_argument("--two-theta-max", type=float, default=80.0)
    parser.add_argument("--step-deg", type=float, default=0.01)
    parser.add_argument("--fwhm-deg", type=float, nargs="+", default=(0.1, 0.5))
    parser.add_argument("--position-atol", type=float, default=1e-10)
    parser.add_argument("--intensity-atol", type=float, default=1e-9)
    parser.add_argument("--profile-atol", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.two_theta_min >= args.two_theta_max:
        raise SystemExit("two-theta-min must be smaller than two-theta-max")
    structures = reference_structures()
    if any(not np.isfinite(value) or value <= 0 for value in args.fwhm_deg):
        raise SystemExit("every fwhm-deg value must be positive and finite")
    if len(set(args.fwhm_deg)) != len(args.fwhm_deg):
        raise SystemExit("fwhm-deg values must be unique")
    comparisons_by_fwhm = {
        fwhm_deg: [
            compare_pattern(
                case,
                structures[case],
                mode=args.mode,
                two_theta_range=(args.two_theta_min, args.two_theta_max),
                step_deg=args.step_deg,
                fwhm_deg=fwhm_deg,
                position_atol=args.position_atol,
                intensity_atol=args.intensity_atol,
                profile_atol=args.profile_atol,
            )
            for case in args.cases
        ]
        for fwhm_deg in args.fwhm_deg
    }

    stem = f"pattern_comparison_{args.mode}"
    png_path = args.output_dir / f"{stem}.png"
    pdf_path = args.output_dir / f"{stem}.pdf"
    json_path = args.output_dir / f"{stem}.json"
    plot_comparisons(comparisons_by_fwhm, png_path=png_path, pdf_path=pdf_path)
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            package: version(package)
            for package in ("braggcalculator", "matplotlib", "numpy", "pymatgen", "spglib")
        },
        "mode": args.mode,
        "two_theta_range_deg": [args.two_theta_min, args.two_theta_max],
        "step_deg": args.step_deg,
        "gaussian_fwhm_deg": args.fwhm_deg,
        "broadening": {
            "applied_identically_to_both_line_patterns": True,
            "kernel": "area-normalized Gaussian",
        },
        "tolerances": {
            "position_atol_deg": args.position_atol,
            "intensity_atol_percent": args.intensity_atol,
            "profile_atol_percent": args.profile_atol,
        },
        "results": [
            asdict(comparison.metrics)
            for comparisons in comparisons_by_fwhm.values()
            for comparison in comparisons
        ],
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {json_path}")
    for fwhm_deg, comparisons in comparisons_by_fwhm.items():
        for comparison in comparisons:
            metrics = comparison.metrics
            print(
                f"{metrics.case}, FWHM {fwhm_deg:g}°: {metrics.peaks} peaks, "
                f"max Δ2θ={metrics.max_position_error_deg:.3e}°, "
                f"max ΔI={metrics.max_line_intensity_error_percent:.3e}, "
                f"max profile Δ={metrics.max_profile_error_percent:.3e}"
            )


if __name__ == "__main__":
    main()
