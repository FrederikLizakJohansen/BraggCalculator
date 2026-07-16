#!/usr/bin/env python3
"""Plot scaling and speedup from one or more benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

METHODS = {
    "pymatgen": ("pymatgen", "#D55E00"),
    "end_to_end": ("BraggCalculator end-to-end", "#0072B2"),
    "cached": ("BraggCalculator cached", "#009E73"),
}
MARKERS = ("o", "s", "^", "D", "P", "X")
MM_PER_INCH = 25.4
NATURE_DOUBLE_COLUMN_MM = 183.0


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    labels = set()
    for path in paths:
        run = json.loads(path.read_text())
        if run.get("schema_version") != 1:
            raise ValueError(f"{path} has an unsupported schema version")
        label = run["hardware"]["label"]
        if label in labels:
            raise ValueError(f"hardware label {label!r} occurs more than once")
        labels.add(label)
        runs.append(run)
    return runs


def _series_results(run: dict, series: str) -> list[dict]:
    return sorted(
        (result for result in run["results"] if result["series"] == series),
        key=lambda result: result["input_sites"],
    )


def paired_speedup_samples(result: dict, method: str) -> np.ndarray:
    """Return per-repeat pymatgen-to-method runtime ratios."""
    samples = result["samples_seconds"]
    reference = np.asarray(samples["pymatgen"], dtype=float)
    measured = np.asarray(samples[method], dtype=float)
    if reference.shape != measured.shape:
        raise ValueError("paired timing series must have the same shape")
    if reference.ndim != 1 or not reference.size:
        raise ValueError("paired timing series must be non-empty and one-dimensional")
    if np.any(reference <= 0.0) or np.any(measured <= 0.0):
        raise ValueError("timing samples must be positive")
    return reference / measured


def _jittered_positions(site_count: int, sample_count: int, offset: float) -> np.ndarray:
    """Separate repeat samples slightly on a base-two logarithmic axis."""
    spread = np.linspace(-0.012, 0.012, sample_count)
    return site_count * np.power(2.0, offset + spread)


def plot_scaling(
    runs: list[dict],
    *,
    png_path: Path,
    pdf_path: Path,
    svg_path: Path,
) -> None:
    if len(runs) > len(MARKERS):
        raise ValueError(f"at most {len(MARKERS)} hardware runs can be plotted")
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
            "svg.hashsalt": "braggcalculator-scaling",
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
        figsize=(NATURE_DOUBLE_COLUMN_MM / MM_PER_INCH, 122.0 / MM_PER_INCH),
        sharex="col",
        layout="constrained",
    )
    series_info = {
        "p1": "P1 cells: all sites irreducible",
        "symmetry": "NaCl supercells: two-site primitive cell",
    }

    for column, (series, title) in enumerate(series_info.items()):
        runtime_axis = axes[0, column]
        speedup_axis = axes[1, column]
        runtime_axis.set_title(title)
        for run_index, run in enumerate(runs):
            results = _series_results(run, series)
            if not results:
                continue
            marker = MARKERS[run_index]
            sites = np.asarray([result["input_sites"] for result in results])
            for method_index, (method, (_, color)) in enumerate(METHODS.items()):
                samples_key = method
                sample_sets = [result["samples_seconds"][samples_key] for result in results]
                method_offset = (-0.052, 0.0, 0.052)[method_index]
                for site_count, samples in zip(sites, sample_sets):
                    runtime_axis.scatter(
                        _jittered_positions(site_count, len(samples), method_offset),
                        1e3 * np.asarray(samples),
                        color=color,
                        marker=marker,
                        s=5.0,
                        linewidths=0,
                        alpha=0.24,
                        zorder=1,
                    )
                medians = 1e3 * np.asarray([np.median(samples) for samples in sample_sets])
                lower = medians - 1e3 * np.asarray(
                    [np.percentile(samples, 25) for samples in sample_sets]
                )
                upper = (
                    1e3 * np.asarray([np.percentile(samples, 75) for samples in sample_sets])
                    - medians
                )
                runtime_axis.errorbar(
                    sites,
                    medians,
                    yerr=np.vstack((lower, upper)),
                    color=color,
                    marker=marker,
                    markersize=3.2,
                    linewidth=0.9,
                    capsize=1.5,
                    elinewidth=0.65,
                    markeredgewidth=0.5,
                    zorder=3,
                )

            for method_index, (method, speedup_key) in enumerate(
                (
                    ("end_to_end", "end_to_end_speedup"),
                    ("cached", "cached_speedup"),
                )
            ):
                _, color = METHODS[method]
                speedup_sets = [paired_speedup_samples(result, method) for result in results]
                method_offset = (-0.032, 0.032)[method_index]
                for site_count, samples in zip(sites, speedup_sets):
                    speedup_axis.scatter(
                        _jittered_positions(site_count, len(samples), method_offset),
                        samples,
                        color=color,
                        marker=marker,
                        s=5.0,
                        linewidths=0,
                        alpha=0.24,
                        zorder=1,
                    )
                quartiles = np.asarray(
                    [np.percentile(samples, (25, 75)) for samples in speedup_sets]
                )
                midpoints = quartiles.mean(axis=1)
                speedup_axis.errorbar(
                    sites,
                    midpoints,
                    yerr=(quartiles[:, 1] - quartiles[:, 0]) / 2.0,
                    color=color,
                    linestyle="none",
                    capsize=1.5,
                    elinewidth=0.65,
                    zorder=2,
                )
                speedup_axis.plot(
                    sites,
                    [result[speedup_key] for result in results],
                    color=color,
                    marker=marker,
                    markersize=3.2,
                    linewidth=0.9,
                    markeredgewidth=0.5,
                    zorder=3,
                )

        runtime_axis.set_xscale("log", base=2)
        runtime_axis.set_yscale("log")
        speedup_axis.set_xscale("log", base=2)
        speedup_axis.set_yscale("log")
        speedup_axis.axhline(1.0, color="0.45", linestyle=(0, (2, 2)), linewidth=0.7)
        speedup_axis.set_xlabel("Supplied sites (atoms)")
        runtime_axis.grid(which="major", color="0.9", linewidth=0.45)
        speedup_axis.grid(which="major", color="0.9", linewidth=0.45)
        speedup_axis.xaxis.set_major_formatter(ScalarFormatter())

    axes[0, 0].set_ylabel("Median runtime (ms)")
    axes[1, 0].set_ylabel("Speedup over pymatgen")
    for label, axis in zip("abcd", axes.flat):
        axis.text(
            -0.13,
            1.03,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    axes[0, 0].text(
        0.02,
        0.96,
        "Large points: central values; small points: repeats; bars: IQR",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=5.3,
        color="0.25",
    )
    method_handles = [
        Line2D([0], [0], color=color, linewidth=1.1, label=label)
        for label, color in METHODS.values()
    ]
    hardware_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            marker=MARKERS[index],
            markersize=3.5,
            linestyle="none",
            label=run["hardware"]["label"],
        )
        for index, run in enumerate(runs)
    ]
    figure.legend(
        handles=method_handles + hardware_handles,
        loc="outside upper center",
        ncols=4 if len(method_handles) + len(hardware_handles) <= 4 else 3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=450)
    figure.savefig(
        pdf_path,
        metadata={"Title": "Diffraction runtime scaling and speedup", "CreationDate": None},
    )
    figure.savefig(
        svg_path,
        metadata={"Title": "Diffraction runtime scaling and speedup", "Date": None},
    )
    svg_text = svg_path.read_text()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_runs(args.inputs)
    png_path = args.output_dir / "scaling_speedup.png"
    pdf_path = args.output_dir / "scaling_speedup.pdf"
    svg_path = args.output_dir / "scaling_speedup.svg"
    plot_scaling(runs, png_path=png_path, pdf_path=pdf_path, svg_path=svg_path)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
