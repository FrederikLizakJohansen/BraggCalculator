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
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MM_PER_INCH = 25.4
NATURE_DOUBLE_COLUMN_MM = 183.0

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
    svg_path: Path,
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
            "axes.labelsize": 6.5,
            "axes.linewidth": 0.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titleweight": "normal",
            "axes.titlesize": 7,
            "font.family": "sans-serif",
            "font.sans-serif": ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"),
            "font.size": 6.5,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "braggcalculator-pattern-comparison",
            "svg.fonttype": "none",
            "xtick.labelsize": 6,
            "xtick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.labelsize": 6,
            "ytick.major.size": 2.5,
            "ytick.major.width": 0.6,
        }
    )
    figure = plt.figure(
        figsize=(NATURE_DOUBLE_COLUMN_MM / MM_PER_INCH, 158.0 / MM_PER_INCH),
        layout="constrained",
    )
    grid = figure.add_gridspec(
        2 * rows,
        columns,
        height_ratios=[3.2, 1.0] * rows,
    )
    display_names = {
        "NaCl": "NaCl",
        "Si": "Si",
        "SrTiO3": r"SrTiO$_3$",
        "triclinic-SiO2": r"triclinic SiO$_2$",
        "NaKCl-disordered": r"Na$_{0.7}$K$_{0.3}$Cl",
        "P1-40-atom": "40-site P1 cell",
    }

    line_width = 1.0
    line_axes = []
    profile_axes: list[list[plt.Axes]] = []
    residual_axes: list[list[plt.Axes]] = []
    for index, line_comparison in enumerate(comparisons):
        line_axis = figure.add_subplot(grid[2 * index : 2 * index + 2, 0])
        line_axes.append(line_axis)

        line_axis.axhline(0.0, color="0.6", linewidth=0.55)
        line_axis.vlines(
            line_comparison.pymatgen_positions,
            0.0,
            line_comparison.pymatgen_intensities,
            color="#D55E00",
            linewidth=line_width,
            label="pymatgen",
        )
        line_axis.vlines(
            line_comparison.braggcalculator_positions,
            0.0,
            -line_comparison.braggcalculator_intensities,
            color="#0072B2",
            linewidth=line_width,
            label="BraggCalculator",
        )
        line_axis.set_ylim(-110, 110)
        line_axis.set_xlim(
            float(line_comparison.grid[0]), float(line_comparison.grid[-1])
        )
        line_axis.set_yticks((-100, -50, 0, 50, 100), ("100", "50", "0", "50", "100"))
        line_axis.set_ylabel("Relative intensity (%)")
        line_axis.text(
            0.98,
            0.05,
            f"max |Δ2θ| {line_comparison.metrics.max_position_error_deg:.1e}°\n"
            f"max |ΔI| {line_comparison.metrics.max_line_intensity_error_percent:.1e}",
            transform=line_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.2,
            color="0.25",
        )

        case_name = display_names.get(line_comparison.metrics.case, line_comparison.metrics.case)
        line_axis.text(
            0.02,
            0.96,
            f"{chr(ord('a') + index)}  {case_name} · {line_comparison.metrics.peaks} peaks",
            transform=line_axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            fontweight="bold",
            color="0.1",
        )

        row_profiles = []
        row_residuals = []
        for column, fwhm_deg in enumerate(fwhm_values, start=1):
            comparison = comparisons_by_fwhm[fwhm_deg][index]
            profile_axis = figure.add_subplot(grid[2 * index, column])
            residual_axis = figure.add_subplot(
                grid[2 * index + 1, column], sharex=profile_axis
            )
            row_profiles.append(profile_axis)
            row_residuals.append(residual_axis)
            x = comparison.grid
            expected = comparison.pymatgen_profile
            actual = comparison.braggcalculator_profile
            profile_axis.plot(
                x,
                expected,
                color="#D55E00",
                linewidth=line_width,
                label="pymatgen",
                zorder=2,
            )
            profile_axis.plot(
                x,
                actual,
                color="#0072B2",
                linewidth=line_width,
                linestyle=(0, (3, 2)),
                label="BraggCalculator",
                zorder=3,
            )
            profile_axis.set_ylim(-2, 104)
            profile_axis.grid(axis="y", color="0.9", linewidth=0.45)
            profile_axis.tick_params(labelbottom=False)
            profile_axis.set_xlim(float(x[0]), float(x[-1]))

            difference = actual - expected
            maximum = float(np.max(np.abs(difference), initial=0.0))
            exponent = int(np.floor(np.log10(maximum))) if maximum > 0 else 0
            scale = 10.0**exponent
            scaled_difference = difference / scale
            limit = max(1.0, 1.12 * float(np.max(np.abs(scaled_difference), initial=0.0)))
            residual_axis.axhline(0.0, color="0.55", linewidth=0.55)
            residual_axis.plot(x, scaled_difference, color="#CC79A7", linewidth=0.85)
            residual_axis.set_ylim(-limit, limit)
            residual_axis.set_yticks((-limit, 0.0, limit))
            residual_axis.set_yticklabels((f"{-limit:.1f}", "0", f"{limit:.1f}"))
            residual_axis.grid(axis="x", color="0.92", linewidth=0.4)
            residual_axis.text(
                0.98,
                0.88,
                rf"$\times 10^{{{exponent}}}$",
                transform=residual_axis.transAxes,
                ha="right",
                va="top",
                fontsize=5.0,
                color="0.3",
            )
            if column == 1:
                residual_axis.set_ylabel("ΔI (%)", fontsize=5.4, labelpad=1.5)
            if index < rows - 1:
                residual_axis.tick_params(labelbottom=False)

        profile_axes.append(row_profiles)
        residual_axes.append(row_residuals)

    for axis in [line_axes[-1], *residual_axes[-1]]:
        axis.set_xlabel(r"$2\theta$ (degrees)")
    for axis in line_axes[:-1]:
        axis.tick_params(labelbottom=False)
    line_axes[0].set_title("Powder lines: no broadening", pad=8)
    for column, fwhm_deg in enumerate(fwhm_values, start=1):
        profile_axes[0][column - 1].set_title(
            f"Both profiles: FWHM {fwhm_deg:g}°", pad=8
        )
    legend_handles = [
        Line2D([0], [0], color="#D55E00", linewidth=1.2, label="pymatgen"),
        Line2D(
            [0],
            [0],
            color="#0072B2",
            linewidth=1.2,
            linestyle=(0, (3, 2)),
            label="BraggCalculator",
        ),
        Line2D([0], [0], color="#CC79A7", linewidth=1.0, label="Difference"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncols=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.6,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=450)
    figure.savefig(
        pdf_path,
        metadata={"Title": "Powder-pattern agreement with pymatgen", "CreationDate": None},
    )
    figure.savefig(
        svg_path,
        metadata={"Title": "Powder-pattern agreement with pymatgen", "Date": None},
    )
    svg_text = svg_path.read_text()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n")
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
    svg_path = args.output_dir / f"{stem}.svg"
    json_path = args.output_dir / f"{stem}.json"
    plot_comparisons(
        comparisons_by_fwhm,
        png_path=png_path,
        pdf_path=pdf_path,
        svg_path=svg_path,
    )
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
    print(f"wrote {svg_path}")
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
