#!/usr/bin/env python3
"""Demonstrate anisotropic-U recovery and chemically restrained refinement."""

from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from pymatgen.core import Lattice, Structure

from braggcalculator import (
    BraggCalculator,
    DiffractionDataset,
    OptimizationStage,
    RefinementPolicy,
    RefinementSession,
    StructuralRestraintSet,
)
from braggcalculator.backends import TorchBackend
from braggcalculator.experimental_profile import (
    axial_divergence_widths,
    render_split_pseudo_voigt,
    thompson_cox_hastings,
)


DEFAULT_FIGURE = Path(__file__).with_name("anisotropic_restraint_refinement.png")
DEFAULT_REPORT = Path(__file__).with_name("anisotropic_restraint_report.html")


def _anisotropic_problem():
    structure = Structure(
        Lattice.tetragonal(4.2, 6.1),
        ["Si"],
        [[0, 0, 0]],
        site_properties={"U_cart": [np.eye(3) * 0.006]},
    )
    target_u = np.asarray([np.diag([0.004, 0.004, 0.014])])
    calculator = BraggCalculator(
        primitive=False,
        two_theta_range=(20.0, 110.0),
        two_theta_step=0.04,
    ).load(structure)
    parameters = calculator.tensor_parameters()
    parameters["u_cart"] = target_u
    centers, areas = calculator.line_components([calculator.wavelength], parameters=parameters)[0]
    grid = np.arange(20.0, 110.0001, 0.04)
    radians = np.radians(centers)
    fwhm, eta = thompson_cox_hastings(radians, 0.0025, 0.0, 0.0036, 0.01, 0.01, calculator.backend)
    low, high = axial_divergence_widths(fwhm, radians, 0.05, calculator.backend)
    intensity = (
        0.002 * render_split_pseudo_voigt(grid, centers, areas, low, high, eta, calculator.backend)
        + 3.0
    )
    dataset = DiffractionDataset(
        coordinate=grid,
        intensity=intensity,
        sigma=np.sqrt(np.maximum(intensity, 1.0)),
        mask=np.ones(len(grid), dtype=bool),
        domain="two_theta",
        wavelength=calculator.wavelength,
        metadata={"kind": "synthetic tetragonal anisotropic-displacement recovery"},
    )
    stages = (
        OptimizationStage("scale/background", ("scale", "background"), 60, 0.03),
        OptimizationStage("anisotropic displacement", ("u_aniso",), 200, 0.025),
        OptimizationStage("joint", ("scale", "background", "u_aniso"), 150, 0.008),
    )
    policy = RefinementPolicy(
        refine_lattice=False,
        refine_u_aniso=True,
        u_aniso_restraint=0.0,
        background_degree=0,
        diagnostic_points=32,
        structural_restraints={"composition": [{"species": "Si", "target": 1.0, "sigma": 0.01}]},
        stages=stages,
    )
    candidate = (
        RefinementSession(dataset, [structure], names=["tetragonal Si"]).run(policy).candidates[0]
    )
    recovered_u = np.asarray(
        candidate.physical_parameters["anisotropic_displacement_groups"][0]["U_cart"]
    )
    return dataset, candidate, target_u[0], recovered_u


def _geometry_problem():
    lattice = Lattice.from_parameters(10.0, 10.5, 11.0, 88, 92, 89)
    matrix = np.asarray(lattice.matrix)
    center = np.array([0.5, 0.5, 0.5])
    target_bond = 1.62
    target_angle = 109.5
    angle = np.radians(target_angle)
    vectors = (
        np.array([target_bond, 0.0, 0.0]),
        target_bond * np.array([np.cos(angle), np.sin(angle), 0.0]),
    )
    target = np.stack(
        [
            center + vectors[0] @ np.linalg.inv(matrix),
            center,
            center + vectors[1] @ np.linalg.inv(matrix),
        ]
    )
    starting = target + np.array([[0.035, -0.025, 0.012], [0.0, 0.0, 0.0], [-0.030, 0.025, -0.015]])
    structure = Structure(lattice, ["O", "Si", "O"], starting)
    calculator = BraggCalculator(
        primitive=False,
        backend=TorchBackend(),
        q_range=(0.5, 4.0),
        q_step=0.02,
    ).load(structure)
    coordinate_model = calculator.symmetry_coordinate_parameterization()
    target_parameters = calculator.tensor_parameters()
    target_parameters["frac_coords"] = torch.as_tensor(target, dtype=torch.float64)
    target_intensity = calculator.fq(parameters=target_parameters).detach()
    selected = torch.argsort(target_intensity, descending=True)[:8]
    restraints = StructuralRestraintSet.from_dict(
        calculator,
        {
            "composition": [{"species": "O", "target": 2.0, "sigma": 0.01}],
            "bonds": [
                {"sites": [0, 1], "target": target_bond, "sigma": 0.02},
                {"sites": [1, 2], "target": target_bond, "sigma": 0.02},
            ],
            "angles": [
                {
                    "sites": [0, 1, 2],
                    "target_degrees": target_angle,
                    "sigma_degrees": 1.5,
                }
            ],
            "minimum_distances": [{"sites": [0, 2], "minimum": 2.50, "sigma": 0.05}],
        },
    )
    return calculator, coordinate_model, restraints, target, target_intensity, selected


def _geometry_metrics(frac_coords, lattice):
    cartesian = np.asarray(frac_coords) @ np.asarray(lattice)
    left = cartesian[0] - cartesian[1]
    right = cartesian[2] - cartesian[1]
    lengths = (np.linalg.norm(left), np.linalg.norm(right))
    cosine = np.dot(left, right) / (lengths[0] * lengths[1])
    return np.array([lengths[0], lengths[1], np.degrees(np.arccos(np.clip(cosine, -1, 1)))])


def _refine_geometry(restraint_weight):
    calculator, model, restraints, target, target_intensity, selected = _geometry_problem()
    values = model.initial_values(calculator.backend, requires_grad=True)
    optimizer = torch.optim.Adam([values], lr=0.015)
    history = []
    for _ in range(1000):
        optimizer.zero_grad()
        parameters = calculator.tensor_parameters()
        parameters["frac_coords"] = model.expand(values, calculator.backend)
        calculated = calculator.fq(parameters=parameters)
        data_loss = torch.mean(
            (
                (calculated[selected] - target_intensity[selected])
                / (torch.sqrt(target_intensity[selected]) + 1.0)
            )
            ** 2
        )
        restraint_loss, terms = restraints.loss(
            parameters["lattice"],
            parameters["frac_coords"],
            parameters["occupancies"],
            calculator.backend,
        )
        loss = data_loss + restraint_weight * restraint_loss
        loss.backward()
        optimizer.step()
        history.append((float(data_loss.detach()), float(restraint_loss.detach())))
    final_coordinates = model.expand(values, calculator.backend).detach().cpu().numpy()
    final_parameters = calculator.tensor_parameters()
    final_parameters["frac_coords"] = torch.as_tensor(final_coordinates, dtype=torch.float64)
    _, final_terms = restraints.loss(
        final_parameters["lattice"],
        final_parameters["frac_coords"],
        final_parameters["occupancies"],
        calculator.backend,
    )
    return {
        "coordinates": final_coordinates,
        "target_coordinates": target,
        "starting_coordinates": np.asarray(calculator._symm["structure"].frac_coords),
        "lattice": np.asarray(calculator._symm["lattice"]),
        "metrics": _geometry_metrics(final_coordinates, calculator._symm["lattice"]),
        "history": np.asarray(history),
        "restraint_contributions": {
            name: float(value.detach()) for name, value in final_terms.items()
        },
        "specification": restraints.specification(),
    }


def _draw_ellipse(axis, tensor, color, label, linestyle="-"):
    section = tensor[np.ix_([0, 2], [0, 2])]
    values, vectors = np.linalg.eigh(section)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    patch = Ellipse(
        (0, 0),
        width=2 * np.sqrt(values[0]),
        height=2 * np.sqrt(values[1]),
        angle=angle,
        fill=False,
        color=color,
        lw=2,
        ls=linestyle,
        label=label,
    )
    axis.add_patch(patch)


def _plot(dataset, candidate, target_u, recovered_u, unrestrained, restrained, output):
    figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    axes[0, 0].plot(dataset.coordinate, dataset.intensity, color="black", lw=0.8, label="target")
    axes[0, 0].plot(
        dataset.coordinate, candidate.calculated, color="#0072B2", lw=0.8, label="refined"
    )
    axes[0, 0].set(
        xlabel=r"$2\theta$ (degrees)",
        ylabel="intensity",
        title="Anisotropic whole-profile recovery",
    )
    axes[0, 0].legend()
    axes[0, 1].plot(dataset.coordinate, candidate.residual, color="#D55E00", lw=0.8)
    axes[0, 1].axhline(0, color="black", lw=0.6)
    axes[0, 1].set(
        xlabel=r"$2\theta$ (degrees)",
        ylabel="observed - calculated",
        title=f"Profile residual, Rwp={candidate.r_wp:.5f}",
    )

    initial_u = np.eye(3) * 0.006
    _draw_ellipse(axes[1, 0], initial_u, "#999999", "initial", "--")
    _draw_ellipse(axes[1, 0], target_u, "#009E73", "target")
    _draw_ellipse(axes[1, 0], recovered_u, "#56B4E9", "recovered", ":")
    limit = 1.2 * np.sqrt(max(np.linalg.eigvalsh(target_u).max(), 0.006))
    axes[1, 0].set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        aspect="equal",
        xlabel=r"$x$ displacement ($\AA$)",
        ylabel=r"$z$ displacement ($\AA$)",
        title="One-sigma x-z displacement ellipse",
    )
    axes[1, 0].legend()

    eigen_target = np.linalg.eigvalsh(target_u)
    eigen_recovered = np.linalg.eigvalsh(recovered_u)
    x = np.arange(3)
    axes[1, 1].bar(x - 0.18, eigen_target, 0.36, color="#009E73", label="target")
    axes[1, 1].bar(x + 0.18, eigen_recovered, 0.36, color="#56B4E9", label="recovered")
    axes[1, 1].set(
        xticks=x,
        xticklabels=("U1", "U2", "U3"),
        ylabel=r"eigenvalue ($\AA^2$)",
        title="Positive tensor eigenvalues",
    )
    axes[1, 1].legend()

    colors = {"target": "#009E73", "unrestrained": "#D55E00", "restrained": "#0072B2"}
    for name, values in (
        ("target", restrained["target_coordinates"]),
        ("unrestrained", unrestrained["coordinates"]),
        ("restrained", restrained["coordinates"]),
    ):
        cartesian = values @ restrained["lattice"]
        local = cartesian - cartesian[1]
        axes[2, 0].plot(
            local[[0, 1, 2], 0], local[[0, 1, 2], 1], "o-", color=colors[name], label=name
        )
    axes[2, 0].set(
        xlabel=r"local x ($\AA$)",
        ylabel=r"local y ($\AA$)",
        aspect="equal",
        title="Sparse-data local geometry",
    )
    axes[2, 0].legend()

    target_metrics = np.array([1.62, 1.62, 109.5])
    ratios = np.vstack(
        [
            _geometry_metrics(restrained["starting_coordinates"], restrained["lattice"])
            / target_metrics,
            unrestrained["metrics"] / target_metrics,
            restrained["metrics"] / target_metrics,
        ]
    )
    labels = ("start", "unrestrained", "restrained")
    width = 0.24
    for index, metric in enumerate(("Si-O 1", "Si-O 2", "O-Si-O")):
        axes[2, 1].bar(
            np.arange(3) + (index - 1) * width,
            ratios[:, index],
            width,
            label=metric,
        )
    axes[2, 1].axhline(1, color="black", lw=0.8)
    axes[2, 1].set(
        xticks=np.arange(3),
        xticklabels=labels,
        ylabel="value / chemical target",
        title=(
            "Geometry despite comparable sparse-data fits\n"
            f"data loss: {unrestrained['history'][-1, 0]:.1e} vs "
            f"{restrained['history'][-1, 0]:.1e}"
        ),
    )
    axes[2, 1].legend()
    figure.suptitle("Anisotropic displacement and structural-restraint diagnostics", fontsize=16)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_html(candidate, target_u, recovered_u, unrestrained, restrained, figure, report):
    encoded = base64.b64encode(Path(figure).read_bytes()).decode("ascii")
    target_metrics = np.array([1.62, 1.62, 109.5])
    rows = []
    for name, metrics in (
        ("Target", target_metrics),
        ("Unrestrained", unrestrained["metrics"]),
        ("Restrained", restrained["metrics"]),
    ):
        rows.append(
            f"<tr><th>{name}</th>" + "".join(f"<td>{value:.6f}</td>" for value in metrics) + "</tr>"
        )
    warnings = "".join(f"<li>{html.escape(value)}</li>" for value in candidate.warnings)
    content = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Anisotropic displacement and restraint diagnostic</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}
img{{width:100%}}table{{border-collapse:collapse}}th,td{{padding:.4rem;border-bottom:1px solid #ddd}}
code,pre{{background:#f4f4f4;padding:.2rem}}</style></head><body>
<h1>Anisotropic displacement and structural-restraint diagnostic</h1>
<p>The tetragonal profile refines to Rwp={candidate.r_wp:.6f}. The recovered U eigenvalues are
<code>{np.linalg.eigvalsh(recovered_u).tolist()}</code>, compared with target
<code>{np.linalg.eigvalsh(target_u).tolist()}</code>.</p>
<img alt="six-panel anisotropic displacement and restraint diagnostic" src="data:image/png;base64,{encoded}">
<h2>Local geometry</h2><table><tr><th>Run</th><th>Si-O 1 (A)</th><th>Si-O 2 (A)</th><th>O-Si-O (deg)</th></tr>
{"".join(rows)}</table>
<p>The unrestrained sparse-reflection fit reaches data loss
<code>{unrestrained["history"][-1, 0]:.6g}</code> while retaining chemically incorrect geometry.
The restrained fit reaches data loss <code>{restrained["history"][-1, 0]:.6g}</code> and the
declared geometry simultaneously. Restraints are prior information, not extra observations.</p>
<h2>Restraint specification</h2><pre>{html.escape(str(restrained["specification"]))}</pre>
<h2>Warnings</h2><ul>{warnings}</ul></body></html>"""
    Path(report).write_text(content, encoding="utf-8")


def run_anisotropic_restraint_demo(output=DEFAULT_FIGURE, report=DEFAULT_REPORT):
    output = Path(output)
    report = Path(report)
    dataset, candidate, target_u, recovered_u = _anisotropic_problem()
    unrestrained = _refine_geometry(0.0)
    restrained = _refine_geometry(0.1)
    _plot(dataset, candidate, target_u, recovered_u, unrestrained, restrained, output)
    _write_html(candidate, target_u, recovered_u, unrestrained, restrained, output, report)
    return candidate, target_u, recovered_u, unrestrained, restrained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    candidate, target_u, recovered_u, unrestrained, restrained = run_anisotropic_restraint_demo(
        args.output, args.report
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    print(f"anisotropic Rwp: {candidate.r_wp:.6f}")
    print(f"maximum U error: {np.max(np.abs(recovered_u - target_u)):.6g} A^2")
    print(f"unrestrained geometry: {unrestrained['metrics']}")
    print(f"restrained geometry: {restrained['metrics']}")
    print(f"unrestrained sparse-data loss: {unrestrained['history'][-1, 0]:.6g}")
    print(f"restrained sparse-data loss: {restrained['history'][-1, 0]:.6g}")


if __name__ == "__main__":
    main()
