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

EXECUTIONS = {
    ("numpy", "cpu"): ("NumPy CPU", "#0072B2"),
    ("torch", "cpu"): ("PyTorch CPU", "#E69F00"),
    ("torch", "cuda"): ("PyTorch CUDA", "#009E73"),
}
TIMINGS = {
    "cached": ("Cached", "-", "o"),
    "end_to_end": ("End-to-end", (0, (4, 2)), "s"),
}
FALLBACK_COLORS = ("#CC79A7", "#56B4E9", "#D55E00")
MM_PER_INCH = 25.4
NATURE_DOUBLE_COLUMN_MM = 183.0


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    executions = set()
    for path in paths:
        run = json.loads(path.read_text())
        if run.get("schema_version") != 1:
            raise ValueError(f"{path} has an unsupported schema version")
        execution = run.get("execution", {})
        key = (
            execution.get("braggcalculator_backend", "numpy"),
            execution.get("braggcalculator_device", "cpu"),
        )
        if key in executions:
            raise ValueError(f"execution {key!r} occurs more than once")
        executions.add(key)
        run["_execution_key"] = key
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


def _execution_style(run: dict, fallback_index: int) -> tuple[str, str]:
    key = run["_execution_key"]
    if key in EXECUTIONS:
        return EXECUTIONS[key]
    backend, device = key
    return (
        f"{backend.title()} {device.upper()}",
        FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)],
    )


def _summary(sample_sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.asarray([np.median(samples) for samples in sample_sets])
    lower = np.asarray([np.percentile(samples, 25) for samples in sample_sets])
    upper = np.asarray([np.percentile(samples, 75) for samples in sample_sets])
    return medians, lower, upper


def _plot_summary(
    axis: plt.Axes,
    sites: np.ndarray,
    sample_sets: list[np.ndarray],
    *,
    color: str,
    linestyle: str | tuple,
    marker: str,
    central_values: np.ndarray | None = None,
    zorder: int = 3,
) -> None:
    medians, lower, upper = _summary(sample_sets)
    if central_values is not None:
        medians = central_values
    axis.fill_between(sites, lower, upper, color=color, alpha=0.13, linewidth=0)
    axis.plot(
        sites,
        medians,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=3.1,
        linewidth=1.05,
        markeredgewidth=0.45,
        zorder=zorder,
    )


def _matched_run(runs: list[dict], key: tuple[str, str]) -> dict | None:
    return next((run for run in runs if run["_execution_key"] == key), None)


def plot_scaling(
    runs: list[dict],
    *,
    png_path: Path,
    pdf_path: Path,
    svg_path: Path,
) -> None:
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
        3,
        2,
        figsize=(NATURE_DOUBLE_COLUMN_MM / MM_PER_INCH, 158.0 / MM_PER_INCH),
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
        acceleration_axis = axes[2, column]
        runtime_axis.set_title(title)
        for run_index, run in enumerate(runs):
            results = _series_results(run, series)
            if not results:
                continue
            _, color = _execution_style(run, run_index)
            sites = np.asarray([result["input_sites"] for result in results])
            for method, (_, linestyle, marker) in TIMINGS.items():
                runtime_sets = [
                    1e3 * np.asarray(result["samples_seconds"][method], dtype=float)
                    for result in results
                ]
                _plot_summary(
                    runtime_axis,
                    sites,
                    runtime_sets,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                )
                speedup_sets = [paired_speedup_samples(result, method) for result in results]
                _plot_summary(
                    speedup_axis,
                    sites,
                    speedup_sets,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    central_values=np.asarray(
                        [result[f"{method}_speedup"] for result in results]
                    ),
                )

        # All pymatgen measurements are CPU runs from the same A3000 host. Pool
        # their repeats into one reference trace to avoid plotting it three times.
        pooled_pymatgen: dict[int, list[float]] = {}
        for run in runs:
            for result in _series_results(run, series):
                pooled_pymatgen.setdefault(result["input_sites"], []).extend(
                    result["samples_seconds"]["pymatgen"]
                )
        pooled_sites = np.asarray(sorted(pooled_pymatgen))
        pooled_samples = [
            1e3 * np.asarray(pooled_pymatgen[site_count], dtype=float)
            for site_count in pooled_sites
        ]
        _plot_summary(
            runtime_axis,
            pooled_sites,
            pooled_samples,
            color="0.3",
            linestyle=(0, (1.2, 1.6)),
            marker="D",
            zorder=2,
        )

        torch_cpu = _matched_run(runs, ("torch", "cpu"))
        torch_cuda = _matched_run(runs, ("torch", "cuda"))
        if torch_cpu is not None and torch_cuda is not None:
            cpu_results = {
                result["input_sites"]: result
                for result in _series_results(torch_cpu, series)
            }
            cuda_results = {
                result["input_sites"]: result
                for result in _series_results(torch_cuda, series)
            }
            matched_sites = np.asarray(sorted(cpu_results.keys() & cuda_results.keys()))
            _, cuda_color = EXECUTIONS[("torch", "cuda")]
            for method, (_, linestyle, marker) in TIMINGS.items():
                ratios = [
                    cpu_results[site_count][f"{method}_seconds"]
                    / cuda_results[site_count][f"{method}_seconds"]
                    for site_count in matched_sites
                ]
                acceleration_axis.plot(
                    matched_sites,
                    ratios,
                    color=cuda_color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3.1,
                    linewidth=1.05,
                    markeredgewidth=0.45,
                )
        else:
            acceleration_axis.text(
                0.5,
                0.5,
                "PyTorch CPU and CUDA records required",
                transform=acceleration_axis.transAxes,
                ha="center",
                va="center",
                color="0.4",
            )

        for axis in (runtime_axis, speedup_axis, acceleration_axis):
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.grid(which="major", color="0.9", linewidth=0.45)
        speedup_axis.axhline(1.0, color="0.45", linestyle=(0, (2, 2)), linewidth=0.7)
        acceleration_axis.axhline(
            1.0, color="0.45", linestyle=(0, (2, 2)), linewidth=0.7
        )
        acceleration_axis.set_xlabel("Supplied sites (atoms)")
        acceleration_axis.xaxis.set_major_formatter(ScalarFormatter())
        acceleration_axis.text(
            0.98,
            0.94,
            "CUDA faster above 1",
            transform=acceleration_axis.transAxes,
            ha="right",
            va="top",
            fontsize=5.3,
            color="0.3",
        )

    axes[0, 0].set_ylabel("Median runtime (ms)")
    axes[1, 0].set_ylabel("Speedup over pymatgen")
    axes[2, 0].set_ylabel("CUDA acceleration\n(PyTorch CPU / CUDA)")
    for label, axis in zip("abcdef", axes.flat):
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
    execution_handles = [
        Line2D([0], [0], color=color, linewidth=1.4, label=label)
        for label, color in (
            _execution_style(run, index) for index, run in enumerate(runs)
        )
    ]
    timing_handles = [
        Line2D(
            [0],
            [0],
            color="0.2",
            linestyle=linestyle,
            marker=marker,
            markersize=3.2,
            linewidth=1.05,
            label=label,
        )
        for label, linestyle, marker in TIMINGS.values()
    ]
    reference_handle = Line2D(
        [0],
        [0],
        color="0.3",
        linestyle=(0, (1.2, 1.6)),
        marker="D",
        markersize=3.0,
        linewidth=1.05,
        label="pymatgen CPU (pooled)",
    )
    figure.legend(
        handles=execution_handles + timing_handles + [reference_handle],
        loc="outside upper center",
        ncols=6,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.15,
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
