#!/usr/bin/env python3
"""Demonstrate rigid-body pose recovery and physical phase-mixture diagnostics."""

from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from braggcalculator import (
    BraggCalculator,
    DiffractionDataset,
    OptimizationStage,
    PhaseMixturePolicy,
    PhaseMixtureSession,
)
from braggcalculator.backends import TorchBackend
from braggcalculator.experimental_profile import caglioti_fwhm, render_pseudo_voigt


DEFAULT_FIGURE = Path(__file__).with_name("rigid_multiphase_refinement.png")
DEFAULT_REPORT = Path(__file__).with_name("rigid_multiphase_report.html")


def _rigid_problem():
    structure = Structure(
        Lattice.from_parameters(8.2, 9.1, 10.3, 81, 87, 76),
        ["Si", "O", "O", "Na"],
        [[0.32, 0.41, 0.52], [0.46, 0.40, 0.50], [0.28, 0.55, 0.48], [0.8, 0.1, 0.2]],
    )
    calculator = BraggCalculator(
        primitive=False,
        backend=TorchBackend(),
        q_range=(0.5, 4.5),
        q_step=0.02,
    ).load(structure)
    model = calculator.rigid_body_parameterization(
        [{"name": "silicate", "sites": [0, 1, 2]}],
        translation_scale=0.1,
        rotation_scale_degrees=5.0,
    )
    target_values = torch.tensor([0.6, -0.4, 0.2, 0.8, -0.5, 0.6], dtype=torch.float64)
    target_parameters = calculator.tensor_parameters()
    target_parameters["frac_coords"] = model.expand(target_values, calculator.backend)
    target_intensity = calculator.fq(parameters=target_parameters).detach()
    normalization = torch.sqrt(torch.clamp(target_intensity, min=0.0)) + 1.0

    values = model.initial_values(calculator.backend, requires_grad=True)
    optimizer = torch.optim.Adam([values], lr=0.03)
    history = []
    for _ in range(900):
        optimizer.zero_grad()
        parameters = calculator.tensor_parameters()
        parameters["frac_coords"] = model.expand(values, calculator.backend)
        calculated = calculator.fq(parameters=parameters)
        loss = torch.mean(((calculated - target_intensity) / normalization) ** 2)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    recovered_values = values.detach().cpu().numpy()
    recovered_fractional = model.expand(values, calculator.backend).detach().cpu().numpy()
    target_fractional = model.expand(target_values, calculator.backend).detach().cpu().numpy()
    initial_cartesian = np.asarray(structure.cart_coords)
    target_cartesian = target_fractional @ structure.lattice.matrix
    recovered_cartesian = recovered_fractional @ structure.lattice.matrix
    pairs = ((0, 1), (0, 2), (1, 2))
    initial_distances = np.array(
        [
            np.linalg.norm(initial_cartesian[left] - initial_cartesian[right])
            for left, right in pairs
        ]
    )
    target_distances = np.array(
        [np.linalg.norm(target_cartesian[left] - target_cartesian[right]) for left, right in pairs]
    )
    recovered_distances = np.array(
        [
            np.linalg.norm(recovered_cartesian[left] - recovered_cartesian[right])
            for left, right in pairs
        ]
    )
    return {
        "model": model,
        "target_values": target_values.numpy(),
        "recovered_values": recovered_values,
        "initial_cartesian": initial_cartesian,
        "target_cartesian": target_cartesian,
        "recovered_cartesian": recovered_cartesian,
        "initial_distances": initial_distances,
        "target_distances": target_distances,
        "recovered_distances": recovered_distances,
        "history": np.asarray(history),
    }


def _phases():
    nacl = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.6402), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    cscl = Structure(Lattice.cubic(4.12), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return nacl, cscl


def _unit_profile(structure, grid, wavelength):
    calculator = BraggCalculator(
        wavelength=wavelength,
        two_theta_range=(float(grid[0]), float(grid[-1])),
        two_theta_step=float(np.median(np.diff(grid))),
        primitive=False,
    ).load(structure)
    centers, areas = calculator.line_components([wavelength], domain="two_theta")[0]
    widths = caglioti_fwhm(np.radians(centers), 0.0025, 1e-6, 0.0064, calculator.backend)
    profile = render_pseudo_voigt(grid, centers, areas, widths, 0.5, calculator.backend)
    return profile / (np.sum(profile) * np.median(np.diff(grid)))


def _mixture_problem():
    phases = _phases()
    names = ("NaCl", "CsCl")
    wavelength = 1.5406
    grid = np.arange(20.0, 90.0001, 0.06)
    unit_profiles = [_unit_profile(phase, grid, wavelength) for phase in phases]
    target_fractions = np.array([0.72, 0.28])
    signal_area = 2500.0
    background = 5.0
    intensity = (
        signal_area
        * sum(fraction * profile for fraction, profile in zip(target_fractions, unit_profiles))
        + background
    )
    dataset = DiffractionDataset(
        coordinate=grid,
        intensity=intensity,
        sigma=np.full(len(grid), 2.0),
        mask=np.ones(len(grid), dtype=bool),
        domain="two_theta",
        wavelength=wavelength,
        metadata={"kind": "synthetic NaCl/CsCl physical mixture"},
    )
    stages = (
        OptimizationStage("scale/background", ("scale", "background"), 40, 0.03),
        OptimizationStage("phase fractions", ("phase_fractions",), 100, 0.025),
        OptimizationStage("joint", ("scale", "background", "phase_fractions"), 140, 0.008),
    )
    policy = PhaseMixturePolicy(
        initial_fractions=(0.5, 0.5),
        refine_profile=False,
        profile_model="legacy",
        background_degree=0,
        diagnostic_points=16,
        stages=stages,
    )
    result = PhaseMixtureSession(dataset, phases, names=names).run(policy)

    weak_fractions = np.array([0.9997, 0.0003])
    weak_components = [
        signal_area * fraction * profile for fraction, profile in zip(weak_fractions, unit_profiles)
    ]
    weak_sigma = np.linalg.norm(weak_components[1]) / 2.0
    weak_dataset = DiffractionDataset(
        coordinate=grid,
        intensity=weak_components[0] + weak_components[1] + background,
        sigma=np.full(len(grid), weak_sigma),
        mask=np.ones(len(grid), dtype=bool),
        domain="two_theta",
        wavelength=wavelength,
        metadata={"kind": "synthetic sub-detectable CsCl trace phase"},
    )
    weak_policy = PhaseMixturePolicy(
        initial_fractions=tuple(weak_fractions),
        refine_profile=False,
        profile_model="legacy",
        background_degree=0,
        diagnostic_points=0,
        stages=(OptimizationStage("scale/background", ("scale", "background"), 30, 0.02),),
    )
    weak_result = PhaseMixtureSession(weak_dataset, phases, names=names).run(weak_policy)
    return dataset, result, target_fractions, weak_result


def _plot(rigid, dataset, mixture, target_fractions, weak, output):
    figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    colors = {"initial": "#999999", "target": "#009E73", "recovered": "#0072B2"}
    pivot = rigid["initial_cartesian"][:3].mean(axis=0)
    for name, coordinates, linestyle in (
        ("initial", rigid["initial_cartesian"], "--"),
        ("target", rigid["target_cartesian"], "-"),
        ("recovered", rigid["recovered_cartesian"], ":"),
    ):
        local = coordinates[:3] - pivot
        axes[0, 0].plot(
            local[:, 0],
            local[:, 1],
            marker="o",
            color=colors[name],
            ls=linestyle,
            label=name,
        )
    axes[0, 0].set(
        aspect="equal",
        xlabel=r"local x ($\AA$)",
        ylabel=r"local y ($\AA$)",
        title="Declared SiO2 group: pose changes, shape does not",
    )
    axes[0, 0].legend()

    pair_labels = ("Si-O1", "Si-O2", "O1-O2")
    distance_error = np.vstack(
        [
            rigid["target_distances"] - rigid["initial_distances"],
            rigid["recovered_distances"] - rigid["initial_distances"],
        ]
    )
    x = np.arange(3)
    axes[0, 1].bar(x - 0.18, distance_error[0], 0.36, color="#009E73", label="target")
    axes[0, 1].bar(x + 0.18, distance_error[1], 0.36, color="#0072B2", label="recovered")
    axes[0, 1].set(
        xticks=x,
        xticklabels=pair_labels,
        ylabel=r"distance change ($\AA$)",
        title="Rigid-body invariant: all internal distances",
    )
    axes[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0, 1].legend()

    physical_target = np.r_[0.1 * rigid["target_values"][:3], 5.0 * rigid["target_values"][3:]]
    physical_recovered = np.r_[
        0.1 * rigid["recovered_values"][:3], 5.0 * rigid["recovered_values"][3:]
    ]
    normalized = np.vstack([physical_target, physical_recovered]) / np.where(
        np.abs(physical_target) > 0, physical_target, 1
    )
    labels = ("tx", "ty", "tz", "rx", "ry", "rz")
    axes[1, 0].bar(np.arange(6) - 0.18, normalized[0], 0.36, color="#009E73", label="target")
    axes[1, 0].bar(np.arange(6) + 0.18, normalized[1], 0.36, color="#0072B2", label="recovered")
    axes[1, 0].axhline(1.0, color="black", lw=0.7)
    axes[1, 0].set(
        xticks=np.arange(6),
        xticklabels=labels,
        ylabel="value / target",
        title=f"Six pose modes recovered; final loss={rigid['history'][-1]:.2e}",
    )
    axes[1, 0].legend()

    axes[1, 1].plot(dataset.coordinate, dataset.intensity, color="black", lw=0.8, label="target")
    axes[1, 1].plot(dataset.coordinate, mixture.calculated, color="#0072B2", lw=0.8, label="fit")
    for name, component in mixture.component_profiles.items():
        axes[1, 1].plot(dataset.coordinate, component + 5.0, lw=0.7, alpha=0.8, label=name)
    axes[1, 1].set(
        xlabel=r"$2\theta$ (degrees)",
        ylabel="intensity",
        title=f"One profile, two physical phases; Rwp={mixture.r_wp:.5f}",
    )
    axes[1, 1].legend(ncol=2)

    recovered_fractions = np.array([mixture.phase_fractions[name] for name in mixture.phase_names])
    axes[2, 0].bar(np.arange(2) - 0.22, [0.5, 0.5], 0.22, color="#999999", label="initial")
    axes[2, 0].bar(np.arange(2), target_fractions, 0.22, color="#009E73", label="target")
    axes[2, 0].bar(
        np.arange(2) + 0.22,
        recovered_fractions,
        0.22,
        color="#0072B2",
        label="recovered",
    )
    axes[2, 0].set(
        xticks=np.arange(2),
        xticklabels=mixture.phase_names,
        ylabel="integrated profile-area fraction",
        ylim=(0, 0.82),
        title="Positive simplex: fractions sum exactly to one",
    )
    axes[2, 0].legend()

    scores = [weak.phase_detectability[name] for name in weak.phase_names]
    bars = axes[2, 1].bar(weak.phase_names, scores, color=("#0072B2", "#D55E00"))
    axes[2, 1].axhline(3.0, color="black", ls="--", label="approx. 3-sigma threshold")
    axes[2, 1].set(
        ylabel=r"component norm, $\sqrt{\sum(I/\sigma)^2}$",
        yscale="log",
        title="0.03% trace phase: numerical value, no experimental support",
    )
    axes[2, 1].bar_label(bars, fmt="%.2f")
    axes[2, 1].legend()
    figure.suptitle("Rigid-body and multi-phase diffraction refinement", fontsize=16)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_html(rigid, mixture, target_fractions, weak, figure, report):
    encoded = base64.b64encode(Path(figure).read_bytes()).decode("ascii")
    target_pose = rigid["model"].physical_groups(rigid["target_values"])[0]
    recovered_pose = rigid["model"].physical_groups(rigid["recovered_values"])[0]
    fraction_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{target:.6f}</td>"
        f"<td>{mixture.phase_fractions[name]:.6f}</td>"
        f"<td>{mixture.phase_detectability[name]:.2f}</td></tr>"
        for name, target in zip(mixture.phase_names, target_fractions)
    )
    warnings = "".join(f"<li>{html.escape(item)}</li>" for item in weak.warnings)
    content = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Rigid-body and multi-phase refinement diagnostic</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}
img{{width:100%}}table{{border-collapse:collapse}}th,td{{padding:.4rem;border-bottom:1px solid #ddd}}
code,pre{{background:#f4f4f4;padding:.2rem}}</style></head><body>
<h1>Rigid-body and multi-phase refinement diagnostic</h1>
<p>The rigid SiO2 group recovers a known translation and rotation with final weighted
intensity loss <code>{rigid["history"][-1]:.6g}</code>. Its maximum internal-distance change is
<code>{np.max(np.abs(rigid["recovered_distances"] - rigid["initial_distances"])):.3e} A</code>.</p>
<img alt="six-panel rigid-body and phase-mixture diagnostic" src="data:image/png;base64,{encoded}">
<h2>Rigid pose</h2><p>Target translation/rotation: <code>{html.escape(str(target_pose))}</code>.<br>
Recovered translation/rotation: <code>{html.escape(str(recovered_pose))}</code>.</p>
<h2>Physical mixture</h2><p>Rwp=<code>{mixture.r_wp:.6f}</code>. These are integrated
profile-area fractions over the fitted range. They are not quantitative mass fractions.</p>
<table><tr><th>Phase</th><th>Target</th><th>Recovered</th><th>Detectability</th></tr>
{fraction_rows}</table>
<h2>Trace-phase guardrail</h2><p>The synthetic trace case contains 0.03% CsCl by profile area.
The noiseless numerical model can carry that value, but the supplied uncertainty makes its
component norm {weak.phase_detectability["CsCl"]:.3f}, below the approximate threshold.</p>
<ul>{warnings}</ul></body></html>"""
    Path(report).write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    rigid = _rigid_problem()
    dataset, mixture, target_fractions, weak = _mixture_problem()
    _plot(rigid, dataset, mixture, target_fractions, weak, args.figure)
    _write_html(rigid, mixture, target_fractions, weak, args.figure, args.report)
    print(f"wrote {args.figure}")
    print(f"wrote {args.report}")
    print(f"rigid final loss={rigid['history'][-1]:.6g}")
    print(f"rigid recovered raw modes={rigid['recovered_values'].tolist()}")
    print(f"mixture Rwp={mixture.r_wp:.6f}; fractions={mixture.phase_fractions}")
    print(f"trace-phase detectability={weak.phase_detectability['CsCl']:.6f}")
    for warning in weak.warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
