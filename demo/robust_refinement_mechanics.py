#!/usr/bin/env python3
"""Executable evidence for guarded differentiable-refinement mechanics."""

from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from braggcalculator import (
    OptimizationStage,
    damped_gauss_newton,
    recommend_parameter_groups,
    staged_optimize,
)


DEFAULT_FIGURE = Path(__file__).with_name("robust_refinement_mechanics.png")
DEFAULT_REPORT = Path(__file__).with_name("robust_refinement_mechanics_report.html")


def _poisson_example():
    rng = np.random.default_rng(20260717)
    coordinate = np.linspace(-3.0, 3.0, 61)
    shape = np.exp(-0.5 * (coordinate / 0.7) ** 2)
    truth = 2.0 + 3.0 * shape
    observed = rng.poisson(truth)
    amplitudes = np.linspace(0.05, 7.0, 350)
    expected = 2.0 + amplitudes[:, None] * shape
    poisson = 2 * np.sum(
        expected
        - observed
        + np.where(observed > 0, observed * np.log(np.maximum(observed, 1) / expected), 0),
        axis=1,
    )
    gaussian = np.sum(
        (expected - observed) ** 2 / np.maximum(observed + 1.0, 1.0), axis=1
    )
    return amplitudes, poisson, gaussian, float(amplitudes[np.argmin(poisson)]), float(
        amplitudes[np.argmin(gaussian)]
    )


def _continuation_example():
    grid = torch.linspace(-6.0, 6.0, 500, dtype=torch.float64)
    target = torch.exp(-0.5 * ((grid - 2.0) / 0.20) ** 2)

    def run(stages):
        center = torch.tensor(-1.8, dtype=torch.float64, requires_grad=True)
        width = 1.0

        def prepare(stage):
            nonlocal width
            width = 0.20 * stage.width_multiplier

        def objective():
            profile = torch.exp(-0.5 * ((grid - center) / width) ** 2)
            return torch.mean((profile - target) ** 2)

        result = staged_optimize(
            objective, {"center": center}, stages, before_stage=prepare
        )
        return result

    direct = run([OptimizationStage("physical only", ("center",), 100, 0.08)])
    continued = run(
        [
            OptimizationStage("wide", ("center",), 100, 0.08, width_multiplier=8.0),
            OptimizationStage("medium", ("center",), 80, 0.05, width_multiplier=3.0),
            OptimizationStage(
                "physical L-BFGS",
                ("center",),
                40,
                0.8,
                optimizer="lbfgs",
                width_multiplier=1.0,
            ),
        ]
    )
    return direct, continued


def _gauss_newton_example():
    target = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)

    def residual(values):
        coordinate = torch.tensor([1.0, 2.0, 3.0], dtype=values.dtype)
        return values[0] * coordinate.pow(values[1]) - target

    return damped_gauss_newton(
        residual, [0.35, 1.2], damping=0.02, trust_radius=0.45, max_steps=30
    )


def _release_example():
    return recommend_parameter_groups(
        {"lattice": 10.0, "occupancy": 4.0, "ADP": 0.05, "duplicate mode": 8.0},
        {"lattice": 3.0, "occupancy": 0.01, "ADP": 2.0, "duplicate mode": 2.0},
        {("duplicate mode", "lattice"): 0.995},
    )


def _rollback_example():
    parameter = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
    return staged_optimize(
        lambda: (parameter - 1.0).square(),
        {"parameter": parameter},
        [OptimizationStage("release candidate", ("parameter",), 50, 0.12)],
        validation_objective=lambda: (parameter + 1.0).square(),
    )


def _restart_example():
    seeds = (11, 17, 23, 29, 31, 37)
    attempts = []
    for seed in seeds:
        start = np.random.default_rng(seed).normal(0.0, 1.4)
        value = torch.tensor(start, dtype=torch.float64, requires_grad=True)
        result = staged_optimize(
            lambda: (value.square() - 1.0).square() + 0.08 * value,
            {"value": value},
            [
                OptimizationStage("Adam", ("value",), 90, 0.04),
                OptimizationStage("L-BFGS", ("value",), 30, 0.7, optimizer="lbfgs"),
            ],
        )
        final = float(result.final_values["value"])
        score = (final**2 - 1.0) ** 2 + 0.08 * final
        attempts.append((seed, start, final, score, result.convergence_classification))
    return attempts


def _build_results():
    return {
        "poisson": _poisson_example(),
        "continuation": _continuation_example(),
        "gauss_newton": _gauss_newton_example(),
        "release": _release_example(),
        "rollback": _rollback_example(),
        "restarts": _restart_example(),
    }


def _plot(results, output):
    figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)

    amplitude, poisson, gaussian, poisson_minimum, gaussian_minimum = results["poisson"]
    axes[0, 0].plot(amplitude, poisson - poisson.min(), label="Poisson deviance")
    axes[0, 0].plot(amplitude, gaussian - gaussian.min(), label="Gaussian WLS")
    axes[0, 0].axvline(3.0, color="black", ls="--", label="generating amplitude")
    axes[0, 0].set(
        xlabel="peak amplitude (counts)",
        ylabel="objective above minimum",
        ylim=(0, 25),
        title=f"Low counts: Poisson {poisson_minimum:.2f}, WLS {gaussian_minimum:.2f}",
    )
    axes[0, 0].legend()

    direct, continued = results["continuation"]
    axes[0, 1].semilogy(direct.loss, label=f"physical only → {direct.final_values['center']:.2f}")
    axes[0, 1].semilogy(
        continued.loss, label=f"wide → physical → {continued.final_values['center']:.2f}"
    )
    axes[0, 1].set(
        xlabel="objective evaluation",
        ylabel="mean squared profile error",
        title="Peak-width continuation recovers a distant narrow peak",
    )
    axes[0, 1].legend()

    gauss_newton = results["gauss_newton"]
    steps = np.arange(len(gauss_newton.loss))
    axes[1, 0].semilogy(steps, gauss_newton.loss, "o-", label="loss")
    axes[1, 0].semilogy(steps, gauss_newton.damping, "s--", label="damping")
    axes[1, 0].semilogy(steps, gauss_newton.trust_radius, "^--", label="trust radius")
    axes[1, 0].set(
        xlabel="Gauss–Newton iteration",
        title=f"Trust mechanics → a={gauss_newton.values[0]:.3f}, p={gauss_newton.values[1]:.3f}",
    )
    axes[1, 0].legend()

    release = results["release"]
    colors = ["#009E73" if item.accepted else "#D55E00" for item in release]
    axes[1, 1].bar([item.group for item in release], [item.sensitivity for item in release], color=colors)
    axes[1, 1].set(
        ylabel="whitened sensitivity (relative units)",
        yscale="log",
        title="Release gate: green accepted, orange rejected",
    )
    axes[1, 1].tick_params(axis="x", rotation=20)

    rollback = results["rollback"].stage_outcomes[0]
    axes[2, 0].bar(
        ["training before", "training candidate", "validation before", "validation candidate"],
        [
            rollback.training_before,
            rollback.training_after,
            rollback.validation_before,
            rollback.validation_after,
        ],
        color=("#56B4E9", "#56B4E9", "#E69F00", "#E69F00"),
    )
    axes[2, 0].set(
        ylabel="loss",
        title="Validation worsens, so the released parameter is restored",
    )
    axes[2, 0].tick_params(axis="x", rotation=18)

    restarts = results["restarts"]
    classifications = sorted({item[4] for item in restarts})
    palette = {name: color for name, color in zip(classifications, ("#0072B2", "#D55E00"))}
    for classification in classifications:
        subset = [item for item in restarts if item[4] == classification]
        axes[2, 1].scatter(
            [item[0] for item in subset],
            [item[2] for item in subset],
            s=80,
            color=palette[classification],
            label=classification.replace("_", " "),
        )
    axes[2, 1].axhline(-1.0, color="black", lw=0.7, ls="--")
    axes[2, 1].axhline(1.0, color="black", lw=0.7, ls="--")
    axes[2, 1].set(
        xlabel="recorded random seed",
        ylabel="final parameter (two local basins)",
        title="Deterministic multistart retains every outcome",
    )
    axes[2, 1].legend(fontsize=8)

    figure.suptitle("Robust differentiable-refinement mechanics", fontsize=17)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_report(results, figure_path, report_path):
    encoded = base64.b64encode(figure_path.read_bytes()).decode("ascii")
    release_rows = "".join(
        f"<tr><td>{html.escape(item.group)}</td><td>{item.accepted}</td>"
        f"<td>{item.sensitivity:.4g}</td><td>{item.residual_support:.4g}</td>"
        f"<td>{html.escape(item.reason)}</td></tr>" for item in results["release"]
    )
    restart_rows = "".join(
        f"<tr><td>{seed}</td><td>{start:.5f}</td><td>{final:.5f}</td>"
        f"<td>{score:.6g}</td><td>{html.escape(classification)}</td></tr>"
        for seed, start, final, score, classification in results["restarts"]
    )
    report_path.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Robust refinement mechanics</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;line-height:1.5}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}}code{{background:#eee;padding:.1rem .3rem}}</style></head><body>
<h1>Robust differentiable-refinement mechanics</h1>
<p>This deterministic executable gate exercises the same optimization primitives exposed by BraggCalculator. It shows why each guard exists; it is not a claim that these synthetic examples establish crystallographic correctness.</p>
<img alt="six robust refinement demonstrations" src="data:image/png;base64,{encoded}">
<h2>Adaptive release evidence</h2><table><tr><th>Group</th><th>Accepted</th><th>Sensitivity</th><th>Residual support</th><th>Reason</th></tr>{release_rows}</table>
<h2>Deterministic restarts</h2><table><tr><th>Seed</th><th>Start</th><th>Final</th><th>Objective</th><th>Classification</th></tr>{restart_rows}</table>
<h2>Interpretation</h2><ul><li>Poisson deviance uses the count observation model; Gaussian WLS is retained for calibrated continuous uncertainties.</li><li>Broad-profile continuation supplies gradients before the physical narrow peaks overlap.</li><li>Damped Gauss–Newton accepts steps using actual versus predicted improvement.</li><li>Validation rollback restores the exact pre-stage snapshot.</li><li>Release decisions and restart outcomes remain machine-readable rather than being hidden behind a single final score.</li></ul>
<p>Seeds: low-count example 20260717; restart seeds 11, 17, 23, 29, 31, 37.</p>
</body></html>""",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    results = _build_results()
    _plot(results, arguments.figure)
    _write_report(results, arguments.figure, arguments.report)
    print(f"figure: {arguments.figure}")
    print(f"report: {arguments.report}")
    print(f"Gauss-Newton: {results['gauss_newton'].convergence_classification}")
    print(f"rollback: {results['rollback'].stage_outcomes[0].reason}")


if __name__ == "__main__":
    main()
