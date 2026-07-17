#!/usr/bin/env python3
"""Demonstrate recoverable and correlated occupancy/Biso refinement regimes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from braggcalculator import (
    BraggCalculator,
    DiffractionDataset,
    OptimizationStage,
    RefinementPolicy,
    RefinementSession,
)
from braggcalculator.backends import TorchBackend
from braggcalculator.experimental_profile import (
    axial_divergence_widths,
    render_split_pseudo_voigt,
    thompson_cox_hastings,
)


DEFAULT_FIGURE = Path(__file__).with_name("occupancy_adp_refinement.png")
DEFAULT_REPORT = Path(__file__).with_name("occupancy_adp_report.html")


def _structure():
    structure = Structure.from_spacegroup(
        "Pm-3m",
        Lattice.cubic(3.9),
        [{"Sr": 0.7, "Ca": 0.3}, "Ti", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
    )
    structure.add_site_property(
        "B",
        [
            0.4 if "Sr" in site.species else (0.3 if "Ti" in site.species else 0.7)
            for site in structure
        ],
    )
    return structure


def _inverse_softplus(value):
    return value if value > 20.0 else np.log(np.expm1(value))


def _target_parameters(calculator):
    occupancy_model = calculator.symmetry_occupancy_parameterization(mode="composition")
    b_iso_model = calculator.symmetry_b_iso_parameterization()
    occupancy_values = np.array([np.log(0.55 / 0.45)])
    b_iso_values = np.array([_inverse_softplus(value) for value in (0.6, 0.35, 1.1)])
    parameters = calculator.tensor_parameters()
    parameters["occupancies"] = occupancy_model.expand(occupancy_values, calculator.backend)
    parameters["b_iso"] = b_iso_model.expand(b_iso_values, calculator.backend)
    return parameters, occupancy_values, b_iso_values


def _synthetic_dataset(structure):
    calculator = BraggCalculator(
        primitive=False,
        two_theta_range=(20.0, 100.0),
        two_theta_step=0.02,
    ).load(structure)
    parameters, occupancy_values, b_iso_values = _target_parameters(calculator)
    centers, areas = calculator.line_components([calculator.wavelength], parameters=parameters)[0]
    grid = np.arange(20.0, 100.0001, 0.02)
    radians = np.radians(centers)
    fwhm, eta = thompson_cox_hastings(radians, 0.0025, 0.0, 0.0036, 0.01, 0.01, calculator.backend)
    low, high = axial_divergence_widths(fwhm, radians, 0.05, calculator.backend)
    intensity = (
        0.0015 * render_split_pseudo_voigt(grid, centers, areas, low, high, eta, calculator.backend)
        + 5.0
    )
    dataset = DiffractionDataset(
        coordinate=grid,
        intensity=intensity,
        sigma=np.sqrt(np.maximum(intensity, 1.0)),
        mask=np.ones(len(grid), dtype=bool),
        domain="two_theta",
        wavelength=calculator.wavelength,
        metadata={"kind": "synthetic mixed-site occupancy and Biso recovery"},
    )
    return dataset, occupancy_values, b_iso_values


def _controlled_recovery(structure, target_occupancy, target_b_iso):
    calculator = BraggCalculator(
        primitive=False,
        backend=TorchBackend(),
        two_theta_range=(20.0, 150.0),
    ).load(structure)
    occupancy_model = calculator.symmetry_occupancy_parameterization(mode="composition")
    b_iso_model = calculator.symmetry_b_iso_parameterization()
    target_occupancy = torch.as_tensor(target_occupancy, dtype=torch.float64)
    target_b_iso = torch.as_tensor(target_b_iso, dtype=torch.float64)

    def intensities(occupancy_values, b_iso_values):
        parameters = calculator.tensor_parameters()
        parameters["occupancies"] = occupancy_model.expand(occupancy_values, calculator.backend)
        parameters["b_iso"] = b_iso_model.expand(b_iso_values, calculator.backend)
        return calculator.line_components([calculator.wavelength], parameters=parameters)[0][1]

    target = intensities(target_occupancy, target_b_iso).detach()
    selected = target > target.max() * 1e-5

    def loss(occupancy_values, b_iso_values):
        difference = intensities(occupancy_values, b_iso_values)[selected] - target[selected]
        return torch.mean((difference / (torch.sqrt(target[selected]) + 1.0)) ** 2)

    occupancy_values = occupancy_model.initial_values(calculator.backend, requires_grad=True)
    b_iso_values = b_iso_model.initial_values(calculator.backend, requires_grad=True)
    history = []
    optimizer = torch.optim.Adam([occupancy_values], lr=0.02)
    for _ in range(500):
        optimizer.zero_grad()
        value = loss(occupancy_values, target_b_iso)
        value.backward()
        optimizer.step()
        history.append(float(value.detach()))
    optimizer = torch.optim.Adam([b_iso_values], lr=0.02)
    for _ in range(700):
        optimizer.zero_grad()
        value = loss(occupancy_values, b_iso_values)
        value.backward()
        optimizer.step()
        history.append(float(value.detach()))
    optimizer = torch.optim.Adam([occupancy_values, b_iso_values], lr=0.01)
    for _ in range(500):
        optimizer.zero_grad()
        value = loss(occupancy_values, b_iso_values)
        value.backward()
        optimizer.step()
        history.append(float(value.detach()))

    recovered = intensities(occupancy_values, b_iso_values).detach().cpu().numpy()
    return {
        "occupancy": occupancy_model.physical_groups(occupancy_values.detach().cpu().numpy()),
        "b_iso": b_iso_model.physical_groups(b_iso_values.detach().cpu().numpy()),
        "target_intensity": target.cpu().numpy(),
        "recovered_intensity": recovered,
        "history": np.asarray(history),
    }


def _joint_session(dataset, structure, report):
    stages = (
        OptimizationStage("scale/background", ("scale", "background"), 80, 0.03),
        OptimizationStage("composition", ("occupancies",), 180, 0.03),
        OptimizationStage("Biso", ("b_iso",), 220, 0.02),
        OptimizationStage(
            "joint",
            ("scale", "background", "occupancies", "b_iso"),
            350,
            0.008,
        ),
    )
    policy = RefinementPolicy(
        refine_lattice=False,
        occupancy_mode="composition",
        occupancy_restraint=0.0,
        refine_b_iso=True,
        b_iso_restraint=0.0,
        background_degree=0,
        diagnostic_points=64,
        stages=stages,
    )
    session = RefinementSession(dataset, [structure], names=["mixed Sr/Ca perovskite"])
    result = session.run(policy)
    session.write_html(result, report)
    return result.candidates[0]


def _plot(dataset, controlled, joint, output):
    target_ca = 0.55
    controlled_ca = controlled["occupancy"][0]["species"]["Ca"]
    joint_ca = joint.physical_parameters["occupancy_groups"][0]["species"]["Ca"]
    target_b = np.array([0.6, 0.35, 1.1])
    controlled_b = np.array([item["B_iso"] for item in controlled["b_iso"]])
    joint_b = np.array(
        [item["B_iso"] for item in joint.physical_parameters["isotropic_displacement_groups"]]
    )

    figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    axes[0, 0].plot(dataset.coordinate, dataset.intensity, color="black", lw=0.8, label="target")
    axes[0, 0].plot(
        dataset.coordinate, joint.calculated, color="#0072B2", lw=0.8, label="joint fit"
    )
    axes[0, 0].set(
        xlabel=r"$2\theta$ (degrees)", ylabel="intensity", title="Whole-profile joint fit"
    )
    axes[0, 0].legend()

    axes[0, 1].plot(dataset.coordinate, joint.residual, color="#D55E00", lw=0.8)
    axes[0, 1].axhline(0, color="black", lw=0.6)
    axes[0, 1].set(
        xlabel=r"$2\theta$ (degrees)",
        ylabel="observed - calculated",
        title=f"Residual, Rwp={joint.r_wp:.4f}",
    )

    labels = ("initial", "target", "controlled", "joint")
    ca_values = (0.30, target_ca, controlled_ca, joint_ca)
    axes[1, 0].bar(labels, ca_values, color=("#999999", "#009E73", "#56B4E9", "#E69F00"))
    axes[1, 0].set(ylim=(0, 0.7), ylabel="Ca site fraction", title="Shared-site composition")
    for index, value in enumerate(ca_values):
        axes[1, 0].text(index, value + 0.015, f"{value:.3f}", ha="center")

    x = np.arange(3)
    width = 0.24
    axes[1, 1].bar(x - width, target_b, width, label="target", color="#009E73")
    axes[1, 1].bar(x, controlled_b, width, label="controlled", color="#56B4E9")
    axes[1, 1].bar(x + width, joint_b, width, label="joint", color="#E69F00")
    axes[1, 1].set(
        xticks=x,
        xticklabels=("A site", "Ti", "O"),
        ylabel=r"$B_{iso}$ ($\AA^2$)",
        title="Orbit-shared isotropic displacement",
    )
    axes[1, 1].legend()

    axes[2, 0].semilogy(np.maximum(controlled["history"], 1e-30), color="#0072B2")
    axes[2, 0].axvline(500, color="black", ls="--", lw=0.8)
    axes[2, 0].axvline(1200, color="black", ls="--", lw=0.8)
    axes[2, 0].set(
        xlabel="optimization step",
        ylabel="weighted intensity loss",
        title="Controlled staged recovery",
    )

    names = joint.identifiability["parameter_names"]
    correlation = np.asarray(joint.identifiability["correlation"])
    image = axes[2, 1].imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    short = [name.replace("occupancies.", "occ.").replace("b_iso.", "B.") for name in names]
    axes[2, 1].set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=short,
        yticklabels=short,
        title="Joint local parameter correlation",
    )
    axes[2, 1].tick_params(axis="x", rotation=65, labelsize=8)
    axes[2, 1].tick_params(axis="y", labelsize=8)
    figure.colorbar(image, ax=axes[2, 1], shrink=0.8)
    figure.suptitle("Symmetry-constrained occupancy and Biso refinement", fontsize=16)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_occupancy_adp_demo(output=DEFAULT_FIGURE, report=DEFAULT_REPORT):
    """Run the recovery and ambiguity examples and write their artifacts."""
    output = Path(output)
    report = Path(report)
    structure = _structure()
    dataset, target_occupancy, target_b_iso = _synthetic_dataset(structure)
    controlled = _controlled_recovery(structure, target_occupancy, target_b_iso)
    joint = _joint_session(dataset, structure, report)
    _plot(dataset, controlled, joint, output)
    return dataset, controlled, joint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    _, controlled, joint = run_occupancy_adp_demo(args.output, args.report)

    controlled_ca = controlled["occupancy"][0]["species"]["Ca"]
    joint_ca = joint.physical_parameters["occupancy_groups"][0]["species"]["Ca"]
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    print("target Ca fraction: 0.550000")
    print(f"controlled recovered Ca fraction: {controlled_ca:.6f}")
    print(f"joint-profile recovered Ca fraction: {joint_ca:.6f}")
    print(f"joint-profile Rwp: {joint.r_wp:.5f}")
    for warning in joint.warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
