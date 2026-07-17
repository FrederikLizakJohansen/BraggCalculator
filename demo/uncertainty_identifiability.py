#!/usr/bin/env python3
"""Demonstrate rank-aware information and bootstrap interval diagnostics."""

from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator, parametric_bootstrap
from braggcalculator.experimental_profile import render_pseudo_voigt
from braggcalculator.sensitivity import analyze_jacobian


DEFAULT_FIGURE = Path(__file__).with_name("uncertainty_identifiability.png")
DEFAULT_REPORT = Path(__file__).with_name("uncertainty_identifiability_report.html")


def _phase_profiles():
    phases = (
        Structure.from_spacegroup(
            "Fm-3m",
            Lattice.cubic(5.6402),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        ),
        Structure(Lattice.cubic(4.12), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    )
    coordinate = np.linspace(20.0, 80.0, 180)
    profiles = []
    for structure in phases:
        calculator = BraggCalculator(
            primitive=False,
            wavelength=1.5406,
            two_theta_range=(20.0, 80.0),
        ).load(structure)
        centers, areas = calculator.line_components([calculator.wavelength])[0]
        profile = render_pseudo_voigt(
            coordinate,
            centers,
            areas,
            np.full(len(centers), 0.55),
            0.35,
            calculator.backend,
        )
        profile /= np.trapezoid(profile, coordinate)
        profiles.append(profile)
    return coordinate, np.asarray(profiles)


def _linear_fraction_problem(*, correlated, target, desired_standard_error):
    coordinate, profiles = _phase_profiles()
    signal_area = 100.0
    background = 10.0
    base = background + signal_area * profiles[0]
    derivative = signal_area * (profiles[1] - profiles[0])
    if correlated:
        indices = np.arange(len(coordinate))
        correlation = 0.55 ** np.abs(indices[:, None] - indices[None, :])
        base_information = derivative @ np.linalg.solve(correlation, derivative)
        sigma = desired_standard_error * np.sqrt(base_information)
        covariance = sigma**2 * correlation
        precision_derivative = np.linalg.solve(covariance, derivative)
        noise = {"covariance": covariance}
    else:
        base_information = derivative @ derivative
        sigma = desired_standard_error * np.sqrt(base_information)
        precision_derivative = derivative / sigma**2
        covariance = None
        noise = {"sigma": np.full(len(coordinate), sigma)}
    denominator = derivative @ precision_derivative

    def estimator(values):
        return np.array([precision_derivative @ (values - base) / denominator])

    expected = base + target * derivative
    return {
        "coordinate": coordinate,
        "profiles": profiles,
        "expected": expected,
        "estimator": estimator,
        "noise": noise,
        "covariance": covariance,
        "sigma": sigma,
        "derivative": derivative,
    }


def _rank_problem():
    coordinate = np.linspace(-1.0, 1.0, 60)
    shape = np.exp(-4.0 * coordinate**2)
    jacobian = np.column_stack([shape, shape])
    data_only = analyze_jacobian(
        jacobian,
        weights=np.ones(len(coordinate)),
        parameter_scales=[0.05, 0.1],
        parameter_names=["occupancy", "Biso"],
    )
    with_prior = analyze_jacobian(
        jacobian,
        weights=np.ones(len(coordinate)),
        parameter_scales=[0.05, 0.1],
        parameter_names=["occupancy", "Biso"],
        prior_precision=np.diag([0.0, 25.0]),
    )
    posterior_singular = np.sqrt(
        np.maximum(np.linalg.eigvalsh(with_prior.posterior_normal_matrix)[::-1], 0.0)
    )
    return data_only, with_prior, posterior_singular


def _bootstrap_problems():
    well = _linear_fraction_problem(correlated=True, target=0.28, desired_standard_error=0.02)
    well_result = parametric_bootstrap(
        well["expected"],
        well["expected"],
        well["estimator"],
        covariance=well["covariance"],
        draws=900,
        bounds=[(0.0, 1.0)],
        parameter_names=("CsCl profile-area fraction",),
        seed=31415,
    )

    coverage_problem = _linear_fraction_problem(
        correlated=False, target=0.28, desired_standard_error=0.025
    )
    outer_rng = np.random.default_rng(2718)
    intervals = []
    for repeat in range(200):
        observed = coverage_problem["expected"] + coverage_problem[
            "sigma"
        ] * outer_rng.standard_normal(len(coverage_problem["expected"]))
        fitted_fraction = float(np.clip(coverage_problem["estimator"](observed)[0], 0.0, 1.0))
        fitted = (
            10.0
            + 100.0 * coverage_problem["profiles"][0]
            + fitted_fraction * coverage_problem["derivative"]
        )
        result = parametric_bootstrap(
            observed,
            fitted,
            coverage_problem["estimator"],
            sigma=coverage_problem["noise"]["sigma"],
            draws=160,
            confidence_level=0.9,
            bounds=[(0.0, 1.0)],
            seed=5000 + repeat,
        )
        intervals.append(
            (result.point_estimate[0], result.lower[0], result.upper[0], result.contains([0.28])[0])
        )
    intervals = np.asarray(intervals, dtype=np.float64)

    boundary = _linear_fraction_problem(
        correlated=False, target=0.003, desired_standard_error=0.012
    )
    boundary_result = parametric_bootstrap(
        boundary["expected"],
        boundary["expected"],
        boundary["estimator"],
        sigma=boundary["noise"]["sigma"],
        draws=900,
        bounds=[(0.0, 1.0)],
        parameter_names=("trace CsCl profile-area fraction",),
        seed=1618,
    )
    return well, well_result, intervals, boundary_result


def _plot(
    data_only, with_prior, posterior_singular, well, well_result, intervals, boundary, output
):
    figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    covariance = well["covariance"]
    correlation = covariance / np.sqrt(np.diag(covariance)[:, None] * np.diag(covariance)[None, :])
    image = axes[0, 0].imshow(correlation[:40, :40], vmin=-1, vmax=1, cmap="coolwarm")
    axes[0, 0].set(
        xlabel="profile bin",
        ylabel="profile bin",
        title="Declared correlated observation model (first 40 bins)",
    )
    figure.colorbar(image, ax=axes[0, 0], label="correlation")

    x = np.arange(2)
    axes[0, 1].bar(x - 0.18, data_only.singular_values, 0.36, label="diffraction data")
    axes[0, 1].bar(x + 0.18, posterior_singular, 0.36, label="data + Biso prior")
    axes[0, 1].set(
        xticks=x,
        xticklabels=("mode 1", "mode 2"),
        yscale="log",
        ylabel="singular value",
        title=f"Data rank {data_only.rank}; posterior rank {with_prior.posterior_rank}",
    )
    axes[0, 1].legend()

    null = data_only.null_space_vectors[0]
    axes[1, 0].bar(("occupancy", "Biso"), null, color=("#0072B2", "#D55E00"))
    axes[1, 0].axhline(0, color="black", lw=0.7)
    axes[1, 0].set(
        ylabel="scaled null-vector coefficient",
        title="Unmeasured combination: the data cannot separate these columns",
    )

    samples = well_result.bootstrap_estimates[:, 0]
    axes[1, 1].hist(samples, bins=35, color="#56B4E9", alpha=0.9)
    axes[1, 1].axvline(0.28, color="#009E73", lw=2, label="target")
    axes[1, 1].axvspan(well_result.lower[0], well_result.upper[0], color="#E69F00", alpha=0.25)
    axes[1, 1].set(
        xlabel="CsCl profile-area fraction",
        ylabel="bootstrap replicates",
        title=(
            f"Correlated-noise 95% interval: {well_result.lower[0]:.3f}–{well_result.upper[0]:.3f}"
        ),
    )
    axes[1, 1].legend()

    for index, (estimate, lower, upper, covered) in enumerate(intervals[:60]):
        color = "#0072B2" if covered else "#D55E00"
        axes[2, 0].plot([lower, upper], [index, index], color=color, lw=1.5)
        axes[2, 0].plot(estimate, index, ".", color=color)
    coverage = np.mean(intervals[:, 3])
    axes[2, 0].axvline(0.28, color="black", ls="--", label="target")
    axes[2, 0].set(
        xlabel="CsCl profile-area fraction",
        ylabel="displayed synthetic experiment",
        title=f"Repeated 90% intervals: empirical coverage {coverage:.1%}",
    )
    axes[2, 0].legend()

    boundary_samples = boundary.bootstrap_estimates[:, 0]
    axes[2, 1].hist(boundary_samples, bins=np.linspace(0, 0.045, 31), color="#D55E00")
    axes[2, 1].axvline(0.003, color="#009E73", lw=2, label="target")
    axes[2, 1].set(
        xlabel="trace CsCl profile-area fraction",
        ylabel="bootstrap replicates",
        title=(
            f"Boundary pile-up: {boundary.boundary_hits[0, 0]}/{boundary.successful_draws} at zero"
        ),
    )
    axes[2, 1].legend()
    figure.suptitle("Calibrated uncertainty and identifiability diagnostics", fontsize=16)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_html(data_only, with_prior, well, intervals, boundary, figure, report):
    encoded = base64.b64encode(Path(figure).read_bytes()).decode("ascii")
    coverage = np.mean(intervals[:, 3])
    null = {
        name: float(value)
        for name, value in zip(data_only.parameter_names, data_only.null_space_vectors[0])
    }
    content = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Uncertainty and identifiability diagnostic</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}
img{{width:100%}}table{{border-collapse:collapse}}th,td{{padding:.4rem;border-bottom:1px solid #ddd}}
code,pre{{background:#f4f4f4;padding:.2rem}}</style></head><body>
<h1>Calibrated uncertainty and identifiability diagnostic</h1>
<p>The diffraction Jacobian has rank {data_only.rank}/{len(data_only.parameter_names)}. A Biso
prior makes the posterior rank {with_prior.posterior_rank}/{len(data_only.parameter_names)}, but
does not make the data identify the null combination <code>{html.escape(str(null))}</code>.</p>
<img alt="six-panel uncertainty and identifiability diagnostic" src="data:image/png;base64,{encoded}">
<h2>Correlated-noise bootstrap</h2><p>The 95% percentile interval for the target 0.28 profile-area
fraction is <code>[{well.lower[0]:.6f}, {well.upper[0]:.6f}]</code> from
{well.successful_draws} successful draws (seed {well.seed}).</p>
<h2>Repeated-synthetic gate</h2><p>Independent-noise 90% bootstrap intervals cover the generating
fraction in <code>{coverage:.1%}</code> of {len(intervals)} repeated experiments. This is an empirical finite-run
check, not proof of universal coverage.</p>
<h2>Bounded trace phase</h2><p>{boundary.boundary_hits[0, 0]} of
{boundary.successful_draws} replicates hit the zero bound. The interval is
<code>[{boundary.lower[0]:.6f}, {boundary.upper[0]:.6f}]</code>; a symmetric Gaussian error bar
would hide this boundary behavior.</p>
<h2>Scope</h2><p>All intervals are conditional on the fixed phase profiles, Gaussian noise model,
known scale/background and supplied parameter bounds. They are not certification uncertainties.</p>
</body></html>"""
    Path(report).write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    data_only, with_prior, posterior_singular = _rank_problem()
    well_problem, well, intervals, boundary = _bootstrap_problems()
    _plot(
        data_only,
        with_prior,
        posterior_singular,
        well_problem,
        well,
        intervals,
        boundary,
        args.figure,
    )
    _write_html(data_only, with_prior, well, intervals, boundary, args.figure, args.report)
    print(f"wrote {args.figure}")
    print(f"wrote {args.report}")
    print(f"data rank={data_only.rank}; posterior rank={with_prior.posterior_rank}")
    print(f"95% correlated-noise interval={well.lower[0]:.6f}..{well.upper[0]:.6f}")
    print(f"90% repeated-synthetic coverage={np.mean(intervals[:, 3]):.3f}")
    print(f"trace lower-bound hits={boundary.boundary_hits[0, 0]}/{boundary.successful_draws}")


if __name__ == "__main__":
    main()
