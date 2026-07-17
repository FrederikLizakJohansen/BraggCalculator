#!/usr/bin/env python3
"""Generate the Milestone 7 linked workspace and project-lifecycle evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from braggcalculator import (
    BraggCalculator,
    OptimizationStage,
    ProjectStore,
    RefinementPolicy,
)
from braggcalculator.diagnostics import compare_calculators


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = HERE / "scientist_workspace_project"
DEFAULT_FIGURE = HERE / "scientist_agent_interface.png"


def _inputs(directory):
    lattice = Lattice.from_parameters(4.8, 5.4, 6.2, 79, 84, 73)
    model_a = Structure(
        lattice, ["Si", "O", "O", "Na"],
        [[0.13, 0.21, 0.34], [0.31, 0.47, 0.11], [0.72, 0.08, 0.59], [0.44, 0.76, 0.27]],
    )
    model_b = model_a.copy()
    model_b.translate_sites([1], [0.018, 0, 0], frac_coords=True)
    model_b.translate_sites([2], [0, -0.012, 0.008], frac_coords=True)
    paths = (directory / "candidate-a.cif", directory / "candidate-b.cif")
    CifWriter(model_a).write_file(paths[0])
    CifWriter(model_b).write_file(paths[1])
    calculator = BraggCalculator(
        primitive=False, two_theta_range=(18.0, 82.0), two_theta_step=0.08,
    ).load(model_a)
    coordinate, profile = calculator.pattern()
    background = 5.0 + 0.015 * (coordinate - coordinate.mean())
    intensity = 0.0012 * profile + background
    sigma = np.sqrt(np.maximum(intensity, 1.0))
    data = directory / "synthetic-pattern.xye"
    np.savetxt(
        data, np.column_stack([coordinate, intensity, sigma]),
        header="synthetic linked-interface pattern; columns: 2theta intensity sigma",
    )
    return data, paths, model_a, model_b


def _policy():
    return RefinementPolicy(
        refine_lattice=True,
        refine_coordinates=True,
        coordinate_restraint=0.05,
        background_degree=1,
        diagnostic_points=32,
        holdout_stride=8,
        stages=(
            OptimizationStage("scale/background", ("scale", "background"), 25, 0.025),
            OptimizationStage("profile/lattice", ("profile", "lattice"), 35, 0.008),
            OptimizationStage("coordinates", ("coordinates",), 35, 0.004),
            OptimizationStage(
                "joint", ("scale", "background", "profile", "lattice", "coordinates"),
                45, 0.002,
            ),
        ),
    )


def build_project(project_path=DEFAULT_PROJECT):
    project_path = Path(project_path)
    if project_path.exists():
        if project_path.name != "scientist_workspace_project":
            raise ValueError("refusing to replace an unexpected project directory")
        shutil.rmtree(project_path)
    with tempfile.TemporaryDirectory() as temporary:
        data, models, model_a, model_b = _inputs(Path(temporary))
        store = ProjectStore.create(
            project_path,
            dataset_path=data,
            model_paths=models,
            names=("reference motif", "oxygen-shift candidate"),
            wavelength=1.5406,
            title="Linked diffraction diagnostic workspace",
            policy=_policy(),
            metadata={
                "kind": "synthetic Milestone 7 interface evidence",
                "limitations": "Noiseless synthetic example; not experimental validation",
            },
        )
    first_document, first_result = store.run()
    final_document, final_result = store.run(resume=True)
    return store, first_document, first_result, final_document, final_result, model_a, model_b


def _plot(evidence, output):
    store, first_doc, first, final_doc, final, model_a, model_b = evidence
    figure, axes = plt.subplots(3, 2, figsize=(15, 14), constrained_layout=True)
    candidate = final.candidates[0]
    stride = max(1, len(final.dataset.coordinate) // 1600)
    axes[0, 0].plot(
        final.dataset.coordinate[::stride], final.dataset.intensity[::stride],
        color="black", lw=0.7, label="observed",
    )
    axes[0, 0].plot(
        final.dataset.coordinate[::stride], candidate.calculated[::stride],
        color="#0072B2", lw=0.7, label="calculated",
    )
    axes[0, 0].plot(
        final.dataset.coordinate[::stride], candidate.residual[::stride],
        color="#D55E00", lw=0.65, label="residual",
    )
    axes[0, 0].set(
        xlabel=r"$2\theta$ (degrees)", ylabel="intensity",
        title=f"Linked profile evidence; Rwp={candidate.r_wp:.5f}",
    )
    axes[0, 0].legend(fontsize=8)

    settings = dict(
        wavelength=1.5406, two_theta_range=(18.0, 82.0), primitive=False
    )
    calc_a = BraggCalculator(**settings).load(model_a)
    calc_b = BraggCalculator(**settings).load(model_b)
    mismatch = compare_calculators(calc_a, calc_b)
    axes[0, 1].add_patch(plt.Circle((0, 0), 1, fill=False, color="0.35"))
    points = axes[0, 1].scatter(
        mismatch.x, mismatch.y, c=mismatch.radius, cmap="viridis", s=18, alpha=0.75
    )
    axes[0, 1].axhline(0, color="0.85", lw=0.7)
    axes[0, 1].axvline(0, color="0.85", lw=0.7)
    axes[0, 1].set(
        aspect="equal", xlim=(-1.04, 1.04), ylim=(-1.04, 1.04),
        xlabel="amplitude coordinate", ylabel="phase coordinate",
        title=f"Interactive mismatch source; Dsf={mismatch.d_sf:.4f}",
    )
    figure.colorbar(points, ax=axes[0, 1], label="per-reflection radius")

    runs = final_doc["runs"]
    y = np.arange(len(runs))
    axes[1, 0].barh(y, [1] * len(runs), color=("#56B4E9", "#009E73"))
    axes[1, 0].set(
        yticks=y,
        yticklabels=[
            f"{run['run_id']}\nparent={run['parent_run_id'] or 'none'}\nresumed={run['resumed']}"
            for run in runs
        ],
        xticks=[], title="Persistent run lineage",
    )

    offsets = 0
    for run, color in zip(runs, ("#56B4E9", "#009E73")):
        encoded = json.loads((store.directory / run["result"]).read_text())
        loss = np.asarray(encoded["candidates"][0]["loss_history"])
        axes[1, 1].semilogy(np.arange(len(loss)) + offsets, loss, color=color, label=run["run_id"])
        offsets += len(loss)
    axes[1, 1].set(
        xlabel="stored optimizer step", ylabel="loss", title="Checkpoint-resumed trace segments"
    )
    axes[1, 1].legend()

    artifact_names = list(runs[-1]["artifacts"])
    axes[2, 0].barh(artifact_names, np.ones(len(artifact_names)), color="#0072B2")
    axes[2, 0].set(xlim=(0, 1.1), xticks=[], title="Auditable export bundle")

    operations = (
        "simulate", "compare", "create", "run", "resume", "status", "result",
        "sensitivity", "measurement",
    )
    colors = ["#009E73" if name not in {"run", "resume"} else "#E69F00" for name in operations]
    axes[2, 1].bar(np.arange(len(operations)), np.ones(len(operations)), color=colors)
    axes[2, 1].set(
        xticks=np.arange(len(operations)), xticklabels=operations, yticks=[], ylim=(0, 1.2),
        title="Same operations through Python, REST and MCP",
    )
    axes[2, 1].tick_params(axis="x", rotation=35)
    figure.suptitle("Scientist workspace and agent-safe project lifecycle", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args(argv)
    evidence = build_project(args.project)
    _plot(evidence, args.figure)
    store, _, _, final, result, *_ = evidence
    latest = final["runs"][-1]
    print(f"wrote {args.figure}")
    print(f"wrote {store.directory / latest['artifacts']['workspace_html']}")
    print(f"runs: {len(final['runs'])}; latest resumed={latest['resumed']}")
    print(result.conclusion)


if __name__ == "__main__":
    main()
