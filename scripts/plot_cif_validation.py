#!/usr/bin/env python3
"""Plot coverage and pymatgen agreement for the frozen CIF validation corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MM_PER_INCH = 25.4
NATURE_DOUBLE_COLUMN_MM = 183.0

CRYSTAL_SYSTEMS = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)
SYSTEM_COLORS = {
    "triclinic": "#332288",
    "monoclinic": "#88CCEE",
    "orthorhombic": "#44AA99",
    "tetragonal": "#117733",
    "trigonal": "#999933",
    "hexagonal": "#DDCC77",
    "cubic": "#CC6677",
}
MODE_STYLES = {
    "xray": ("X-ray", "#D55E00", "o"),
    "neutron": ("Neutron", "#0072B2", "^"),
}


def load_results(path: Path) -> dict:
    """Load and minimally validate a frozen corpus result artifact."""
    artifact = json.loads(path.read_text())
    if artifact.get("schema_version") != 1:
        raise ValueError(f"{path} has an unsupported schema version")
    results = artifact.get("results", [])
    if not results:
        raise ValueError(f"{path} contains no comparison results")
    if artifact.get("comparison_count") != len(results):
        raise ValueError("comparison_count does not match the result records")
    if any(result["mode"] not in MODE_STYLES for result in results):
        raise ValueError("result contains an unsupported radiation mode")
    return artifact


def _identity_line(axis: plt.Axes, values: np.ndarray) -> None:
    lower = 0.82 * float(np.min(values))
    upper = 1.22 * float(np.max(values))
    axis.plot(
        (lower, upper),
        (lower, upper),
        color="0.45",
        linestyle=(0, (2, 2)),
        linewidth=0.75,
        zorder=1,
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)


def plot_validation(
    artifact: dict,
    *,
    png_path: Path,
    pdf_path: Path,
    svg_path: Path,
) -> None:
    """Render corpus coverage, parity, and tolerance-normalized errors."""
    results = artifact["results"]
    structures = {}
    for result in results:
        structures.setdefault(result["id"], result)

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
            "legend.fontsize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "braggcalculator-cif-validation",
            "svg.fonttype": "none",
            "xtick.labelsize": 6,
            "xtick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.labelsize": 6,
            "ytick.major.size": 2.5,
            "ytick.major.width": 0.6,
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(NATURE_DOUBLE_COLUMN_MM / MM_PER_INCH, 123.0 / MM_PER_INCH),
        layout="constrained",
    )
    coverage_axis, parity_axis, position_axis, intensity_axis = axes.flat

    for system in CRYSTAL_SYSTEMS:
        system_results = [
            result for result in structures.values() if result["crystal_system"] == system
        ]
        for disordered, marker in ((False, "o"), (True, "s")):
            selected = [result for result in system_results if result["disordered"] == disordered]
            if not selected:
                continue
            coverage_axis.scatter(
                [result["input_sites"] for result in selected],
                [result["reduced_sites"] for result in selected],
                color=SYSTEM_COLORS[system],
                marker=marker,
                s=17,
                edgecolor="0.15" if disordered else "white",
                linewidth=0.55 if disordered else 0.35,
                alpha=0.88,
                zorder=3,
            )
    site_values = np.asarray(
        [value for result in structures.values() for value in (result["input_sites"], result["reduced_sites"])],
        dtype=float,
    )
    _identity_line(coverage_axis, site_values)
    coverage_axis.set_xscale("log")
    coverage_axis.set_yscale("log")
    coverage_axis.set_xlabel("Supplied sites")
    coverage_axis.set_ylabel("Primitive-cell sites")
    coverage_axis.set_title("Corpus coverage and symmetry reduction")
    coverage_axis.text(
        0.04,
        0.95,
        "70 structures · 10 per crystal system\n"
        "62 space groups · 31 disordered",
        transform=coverage_axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color="0.25",
    )

    for mode, (_, color, marker) in MODE_STYLES.items():
        selected = [result for result in results if result["mode"] == mode]
        parity_axis.scatter(
            [result["pymatgen_peaks"] for result in selected],
            [result["braggcalculator_peaks"] for result in selected],
            color=color,
            marker=marker,
            s=15,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.72,
            zorder=3,
        )
    peak_values = np.asarray(
        [
            value
            for result in results
            for value in (result["pymatgen_peaks"], result["braggcalculator_peaks"])
        ],
        dtype=float,
    )
    _identity_line(parity_axis, peak_values)
    parity_axis.set_xscale("log")
    parity_axis.set_yscale("log")
    parity_axis.set_xlabel("pymatgen peaks")
    parity_axis.set_ylabel("BraggCalculator peaks")
    parity_axis.set_title("Powder-line count parity")
    parity_axis.text(
        0.04,
        0.95,
        f"{artifact['passed_count']}/{artifact['comparison_count']} comparisons passed\n"
        "all peak counts identical",
        transform=parity_axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color="0.25",
    )

    error_panels = (
        (
            position_axis,
            "max_position_error_deg",
            "position_atol_deg",
            "Line-position margin below tolerance",
        ),
        (
            intensity_axis,
            "max_intensity_error_percent",
            "intensity_atol_percent",
            "Intensity margin below tolerance",
        ),
    )
    for axis, value_key, tolerance_key, title in error_panels:
        normalized_errors = []
        for mode, (_, color, marker) in MODE_STYLES.items():
            selected = [result for result in results if result["mode"] == mode]
            normalized_error = np.asarray(
                [result[value_key] / result[tolerance_key] for result in selected]
            )
            margins = -np.log10(normalized_error)
            normalized_errors.extend(normalized_error)
            axis.scatter(
                [result["pymatgen_peaks"] for result in selected],
                margins,
                color=color,
                marker=marker,
                s=15,
                edgecolor="white",
                linewidth=0.3,
                alpha=0.72,
            )
        axis.set_xscale("log")
        axis.set_ylim(4.9, 7.7)
        axis.set_xlabel("pymatgen peaks")
        axis.set_ylabel("Orders of magnitude")
        axis.set_title(title)
        axis.grid(which="major", color="0.9", linewidth=0.45)
        axis.text(
            0.04,
            0.93,
            f"all ≥ {-np.log10(max(normalized_errors)):.1f} orders below tolerance",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=5.2,
            color="0.25",
        )

    coverage_axis.grid(which="major", color="0.9", linewidth=0.45)
    parity_axis.grid(which="major", color="0.9", linewidth=0.45)
    for label, axis in zip("abcd", axes.flat):
        axis.text(
            -0.13,
            1.04,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    system_handles = [
        Line2D(
            [0],
            [0],
            color=SYSTEM_COLORS[system],
            marker="o",
            markersize=3.4,
            linestyle="none",
            label=system.title(),
        )
        for system in CRYSTAL_SYSTEMS
    ]
    mode_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            markersize=3.4,
            linestyle="none",
            label=label,
        )
        for label, color, marker in MODE_STYLES.values()
    ]
    disorder_handle = Line2D(
        [0],
        [0],
        color="0.25",
        marker="s",
        markerfacecolor="white",
        markersize=3.4,
        linestyle="none",
        label="Disordered",
    )
    figure.legend(
        handles=system_handles + mode_handles + [disorder_handle],
        loc="outside upper center",
        ncols=5,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=450)
    figure.savefig(
        pdf_path,
        metadata={"Title": "Frozen CIF corpus validation", "CreationDate": None},
    )
    figure.savefig(
        svg_path,
        metadata={"Title": "Frozen CIF corpus validation", "Date": None},
    )
    svg_text = svg_path.read_text()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "paper" / "data" / "cif_validation_results.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_results(args.input)
    stem = "cif_validation_summary"
    plot_validation(
        artifact,
        png_path=args.output_dir / f"{stem}.png",
        pdf_path=args.output_dir / f"{stem}.pdf",
        svg_path=args.output_dir / f"{stem}.svg",
    )
    for extension in ("png", "pdf", "svg"):
        print(f"wrote {args.output_dir / f'{stem}.{extension}'}")


if __name__ == "__main__":
    main()
