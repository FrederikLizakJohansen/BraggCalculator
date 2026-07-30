#!/usr/bin/env python3
"""Plot one-pass COD corpus throughput from matched CPU and CUDA records."""

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

EXECUTIONS = {
    ("numpy", "cpu"): ("NumPy CPU", "#0072B2"),
    ("torch", "cpu"): ("PyTorch CPU", "#E69F00"),
    ("torch", "cuda"): ("PyTorch CUDA", "#009E73"),
}
EXECUTION_ORDER = tuple(EXECUTIONS)
MODE_STYLES = {
    "xray": ("X-ray", "#D55E00", "o"),
    "neutron": ("Neutron", "#0072B2", "^"),
}
DEFAULT_INPUTS = (
    ROOT / "paper" / "data" / "cod_throughput_cpu_numpy_A3000.json",
    ROOT / "paper" / "data" / "cod_throughput_cpu_torch_A3000.json",
    ROOT / "paper" / "data" / "cod_throughput_gpu_torch_A3000.json",
)


def _result_map(run: dict) -> dict[tuple[str, str], dict]:
    mapped = {(result["id"], result["mode"]): result for result in run["results"]}
    if len(mapped) != len(run["results"]):
        raise ValueError("a throughput record contains duplicate structure-mode results")
    return mapped


def load_runs(paths: list[Path]) -> dict[tuple[str, str], dict]:
    """Load and cross-check the three matched publication records."""
    runs = {}
    for path in paths:
        run = json.loads(path.read_text())
        if run.get("schema_version") != 1:
            raise ValueError(f"{path} has an unsupported schema version")
        execution = run.get("execution", {})
        key = (
            execution.get("braggcalculator_backend"),
            execution.get("braggcalculator_device"),
        )
        if key not in EXECUTIONS:
            raise ValueError(f"{path} has unsupported execution {key!r}")
        if key in runs:
            raise ValueError(f"execution {key!r} occurs more than once")
        if run.get("corpus", {}).get("limited"):
            raise ValueError(f"{path} contains a limited corpus run")
        if run.get("timing_protocol", {}).get(
            "timed_evaluations_per_structure_and_mode"
        ) != 1:
            raise ValueError(f"{path} did not time each structure-mode pair exactly once")
        results = run.get("results", [])
        if not results or not all(result.get("passed") for result in results):
            raise ValueError(f"{path} contains missing or failed timed comparisons")
        if not run.get("summaries", {}).get("all", {}).get("passed"):
            raise ValueError(f"{path} has an invalid aggregate summary")
        for result in results:
            if result["mode"] not in MODE_STYLES:
                raise ValueError(f"{path} contains an unsupported radiation mode")
            if result["braggcalculator_seconds"] <= 0 or result["pymatgen_seconds"] <= 0:
                raise ValueError(f"{path} contains a non-positive runtime")
            expected_speedup = result["pymatgen_seconds"] / result[
                "braggcalculator_seconds"
            ]
            if not np.isclose(result["speedup"], expected_speedup, rtol=1e-12):
                raise ValueError(f"{path} contains an inconsistent speedup")
        runs[key] = run

    if set(runs) != set(EXECUTIONS):
        missing = set(EXECUTIONS) - set(runs)
        raise ValueError(f"missing required execution records: {sorted(missing)}")

    reference = runs[EXECUTION_ORDER[0]]
    reference_results = _result_map(reference)
    invariant_run_fields = (
        "git_revision",
        "modes",
        "position_atol_deg",
        "intensity_atol_percent",
    )
    invariant_result_fields = (
        "sha256",
        "crystal_system",
        "disordered",
        "input_sites",
        "reduced_sites",
        "braggcalculator_peaks",
        "pymatgen_peaks",
    )
    for key in EXECUTION_ORDER[1:]:
        run = runs[key]
        for field in invariant_run_fields:
            if run.get(field) != reference.get(field):
                raise ValueError(f"execution records disagree on {field}")
        if run["corpus"].get("manifest_sha256") != reference["corpus"].get(
            "manifest_sha256"
        ):
            raise ValueError("execution records use different corpus manifests")
        results = _result_map(run)
        if results.keys() != reference_results.keys():
            raise ValueError("execution records contain different structure-mode pairs")
        for result_key, reference_result in reference_results.items():
            for field in invariant_result_fields:
                if results[result_key].get(field) != reference_result.get(field):
                    raise ValueError(
                        f"execution records disagree on {field} for {result_key}"
                    )
    return runs


def corpus_metrics(runs: dict[tuple[str, str], dict]) -> dict:
    """Return the aggregate values displayed and quoted in the manuscript."""
    pymatgen_by_mode = {
        mode: float(
            np.median(
                [run["summaries"][mode]["pymatgen_total_seconds"] for run in runs.values()]
            )
        )
        for mode in MODE_STYLES
    }
    runtimes = {
        key: {
            mode: runs[key]["summaries"][mode]["braggcalculator_total_seconds"]
            for mode in MODE_STYLES
        }
        for key in EXECUTION_ORDER
    }
    speedups = {
        key: {
            mode: runs[key]["summaries"][mode]["total_corpus_speedup"]
            for mode in (*MODE_STYLES, "all")
        }
        for key in EXECUTION_ORDER
    }
    torch_cpu_total = runs[("torch", "cpu")]["summaries"]["all"][
        "braggcalculator_total_seconds"
    ]
    torch_cuda_total = runs[("torch", "cuda")]["summaries"]["all"][
        "braggcalculator_total_seconds"
    ]
    return {
        "pymatgen_by_mode": pymatgen_by_mode,
        "runtimes": runtimes,
        "speedups": speedups,
        "torch_cuda_acceleration": torch_cpu_total / torch_cuda_total,
    }


def _panel_label(axis: plt.Axes, label: str) -> None:
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


def plot_throughput(
    runs: dict[tuple[str, str], dict],
    *,
    png_path: Path,
    pdf_path: Path,
    svg_path: Path,
) -> None:
    metrics = corpus_metrics(runs)
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
            "svg.hashsalt": "braggcalculator-cod-throughput",
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
        figsize=(NATURE_DOUBLE_COLUMN_MM / MM_PER_INCH, 119.0 / MM_PER_INCH),
        layout="constrained",
    )
    runtime_axis, aggregate_axis, gpu_axis, acceleration_axis = axes.flat

    runtime_rows = [
        (
            "pymatgen CPU",
            "0.38",
            metrics["pymatgen_by_mode"],
        )
    ] + [
        (EXECUTIONS[key][0], EXECUTIONS[key][1], metrics["runtimes"][key])
        for key in EXECUTION_ORDER
    ]
    for row_index, (label, color, mode_values) in enumerate(runtime_rows):
        left = 0.0
        for mode in MODE_STYLES:
            value = mode_values[mode]
            runtime_axis.barh(
                row_index,
                value,
                left=left,
                height=0.62,
                color=color,
                alpha=0.96 if mode == "xray" else 0.62,
                hatch=None if mode == "xray" else "////",
                edgecolor="white",
                linewidth=0.45,
            )
            left += value
        runtime_axis.text(
            left + 1.7,
            row_index,
            f"{left:.1f} s",
            ha="left",
            va="center",
            fontsize=5.7,
            color="0.2",
        )
    runtime_axis.set_yticks(np.arange(len(runtime_rows)))
    runtime_axis.set_yticklabels([row[0] for row in runtime_rows])
    runtime_axis.invert_yaxis()
    runtime_axis.set_xlim(0, 108)
    runtime_axis.set_xlabel("Summed runtime (s)")
    runtime_axis.set_title("One pass: 70 structures × 2 radiation modes")
    runtime_axis.grid(axis="x", color="0.9", linewidth=0.45)

    mode_offsets = {"xray": -0.17, "neutron": 0.17, "all": 0.0}
    mode_markers = {"xray": "o", "neutron": "^", "all": "D"}
    x_values = np.arange(len(EXECUTION_ORDER))
    for x_value, key in zip(x_values, EXECUTION_ORDER):
        color = EXECUTIONS[key][1]
        values = metrics["speedups"][key]
        aggregate_axis.plot(
            [x_value, x_value],
            [min(values.values()), max(values.values())],
            color=color,
            alpha=0.45,
            linewidth=0.8,
            zorder=1,
        )
        for mode in ("xray", "neutron", "all"):
            aggregate_axis.scatter(
                x_value + mode_offsets[mode],
                values[mode],
                color=color,
                marker=mode_markers[mode],
                s=22 if mode == "all" else 18,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
        aggregate_axis.text(
            x_value,
            values["all"] + 0.75,
            f"{values['all']:.2f}×",
            ha="center",
            va="bottom",
            fontsize=5.7,
            color=color,
        )
    aggregate_axis.axhline(1.0, color="0.45", linestyle=(0, (2, 2)), linewidth=0.7)
    aggregate_axis.set_xticks(x_values)
    aggregate_axis.set_xticklabels([EXECUTIONS[key][0] for key in EXECUTION_ORDER])
    aggregate_axis.set_ylim(0, 18.2)
    aggregate_axis.set_ylabel("Speedup over pymatgen")
    aggregate_axis.set_title("Full-corpus speedup (ratio of total runtimes)")
    aggregate_axis.grid(axis="y", color="0.9", linewidth=0.45)

    gpu_run = runs[("torch", "cuda")]
    for mode, (label, color, marker) in MODE_STYLES.items():
        selected = [result for result in gpu_run["results"] if result["mode"] == mode]
        gpu_axis.scatter(
            [result["reduced_sites"] for result in selected],
            [result["speedup"] for result in selected],
            color=color,
            marker=marker,
            s=16,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.75,
            label=label,
            zorder=3,
        )
    gpu_axis.set_xscale("log")
    gpu_axis.set_yscale("log")
    gpu_axis.axhline(1.0, color="0.45", linestyle=(0, (2, 2)), linewidth=0.7)
    gpu_axis.set_xlabel("Primitive-cell sites")
    gpu_axis.set_ylabel("CUDA speedup over pymatgen")
    gpu_axis.set_title("Per-structure CUDA performance")
    gpu_axis.grid(which="major", color="0.9", linewidth=0.45)
    gpu_axis.text(
        0.97,
        0.06,
        "13.17× over the full corpus",
        transform=gpu_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color=EXECUTIONS[("torch", "cuda")][1],
    )

    cpu_results = _result_map(runs[("torch", "cpu")])
    cuda_results = _result_map(gpu_run)
    for mode, (label, color, marker) in MODE_STYLES.items():
        result_keys = sorted(key for key in cpu_results if key[1] == mode)
        acceleration_axis.scatter(
            [cpu_results[key]["reduced_sites"] for key in result_keys],
            [
                cpu_results[key]["braggcalculator_seconds"]
                / cuda_results[key]["braggcalculator_seconds"]
                for key in result_keys
            ],
            color=color,
            marker=marker,
            s=16,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.75,
            label=label,
            zorder=3,
        )
    acceleration_axis.set_xscale("log")
    acceleration_axis.set_yscale("log")
    acceleration_axis.axhline(
        1.0, color="0.45", linestyle=(0, (2, 2)), linewidth=0.7
    )
    acceleration_axis.set_xlabel("Primitive-cell sites")
    acceleration_axis.set_ylabel("PyTorch CPU / CUDA runtime")
    acceleration_axis.set_title("Direct CUDA acceleration")
    acceleration_axis.grid(which="major", color="0.9", linewidth=0.45)
    acceleration_axis.text(
        0.97,
        0.06,
        f"{metrics['torch_cuda_acceleration']:.2f}× over the full corpus",
        transform=acceleration_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color=EXECUTIONS[("torch", "cuda")][1],
    )

    for label, axis in zip("abcd", axes.flat):
        _panel_label(axis, label)
    mode_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle="none",
            markersize=3.7,
            label=label,
        )
        for label, color, marker in MODE_STYLES.values()
    ]
    mode_handles.append(
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="D",
            linestyle="none",
            markersize=3.7,
            label="Combined total",
        )
    )
    figure.legend(
        handles=mode_handles,
        loc="outside upper center",
        ncols=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.2,
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=450)
    figure.savefig(
        pdf_path,
        metadata={"Title": "COD corpus diffraction throughput", "CreationDate": None},
    )
    figure.savefig(
        svg_path,
        metadata={"Title": "COD corpus diffraction throughput", "Date": None},
    )
    svg_text = svg_path.read_text()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_runs(args.inputs)
    stem = "cod_throughput"
    plot_throughput(
        runs,
        png_path=args.output_dir / f"{stem}.png",
        pdf_path=args.output_dir / f"{stem}.pdf",
        svg_path=args.output_dir / f"{stem}.svg",
    )
    for extension in ("png", "pdf", "svg"):
        print(f"wrote {args.output_dir / f'{stem}.{extension}'}")


if __name__ == "__main__":
    main()
