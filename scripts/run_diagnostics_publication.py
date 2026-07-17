#!/usr/bin/env python3
"""Regenerate and verify the Milestone 8 diagnostics publication package."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator, diagnose_structures
from braggcalculator.diagnostics import match_reflections
from braggcalculator.profiles import GaussianProfileQ
from braggcalculator.publication import (
    PUBLICATION_SCHEMA,
    WEIGHTING_SCHEMES,
    compare_weighting_schemes,
    cyclic_difference_multiset,
    cyclic_sets_dihedrally_equivalent,
    profile_metric_suite,
    publication_gate_summary,
    verify_input_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "publication_diagnostics"
CASES = DATA / "cases"
OUTPUT = ROOT / "paper" / "diagnostics"
FIGURES = OUTPUT / "figures"
Q_RANGE = (0.4, 6.0)
Q_STEP = 0.005
FIGURE_CREATOR = "BraggCalculator diagnostics benchmark 1.0.0"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _save_figure(figure, stem):
    figure.savefig(
        FIGURES / f"{stem}.png",
        dpi=220,
        metadata={"Software": FIGURE_CREATOR},
    )
    figure.savefig(
        FIGURES / f"{stem}.pdf",
        dpi=220,
        metadata={"Creator": FIGURE_CREATOR, "CreationDate": None, "ModDate": None},
    )
    svg_path = FIGURES / f"{stem}.svg"
    figure.savefig(
        svg_path,
        dpi=220,
        metadata={"Creator": FIGURE_CREATOR, "Date": None},
    )
    normalized_svg = "\n".join(
        line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
    )
    svg_path.write_text(normalized_svg + "\n", encoding="utf-8")


def _calculator(structure, *, q_stop=6.0, fwhm=0.08):
    return BraggCalculator(
        primitive=False,
        symprec=1e-5,
        q_range=(Q_RANGE[0], q_stop),
        q_step=Q_STEP,
        profile_q=GaussianProfileQ(fwhm),
    ).load(structure)


def _profile_pair(path_a, path_b, *, fwhm):
    calculator_a = _calculator(path_a, fwhm=fwhm)
    calculator_b = _calculator(path_b, fwhm=fwhm)
    coordinate, profile_a = calculator_a.pattern(domain="q")
    other_coordinate, profile_b = calculator_b.pattern(domain="q")
    np.testing.assert_allclose(coordinate, other_coordinate, rtol=0, atol=1e-14)
    return np.asarray(coordinate), np.asarray(profile_a), np.asarray(profile_b)


def _complex_pair(path_a, path_b, *, q_stop=6.0):
    calculator_a = _calculator(path_a, q_stop=q_stop)
    calculator_b = _calculator(path_b, q_stop=q_stop)
    table_a = calculator_a.reflection_table(domain="q")
    table_b = calculator_b.reflection_table(domain="q")
    match = match_reflections(table_a.hkl, table_b.hkl)
    return (
        match.hkl,
        np.asarray(table_a.q)[match.indices_a],
        np.asarray(table_a.structure_factor)[match.indices_a],
        np.asarray(table_b.structure_factor)[match.indices_b],
    )


def _weighting_results(path_a, path_b, *, q_stop=6.0):
    hkl, q, factor_a, factor_b = _complex_pair(path_a, path_b, q_stop=q_stop)
    return compare_weighting_schemes(hkl, q, factor_a, factor_b, shell_width=0.5)


def _equivalent_structures():
    reference = Structure.from_file(CASES / "realistic-compatible-a.cif")
    shift = np.array([0.125, 0.25, 0.375])
    reversed_sites = list(reversed(reference))
    shifted = Structure(
        reference.lattice,
        [site.species for site in reversed_sites],
        [(site.frac_coords + shift) % 1.0 for site in reversed_sites],
    )
    wrapped = Structure(
        reference.lattice,
        [site.species for site in reference],
        [site.frac_coords + [1.0, -1.0, 2.0] for site in reference],
    )
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = Structure(
        Lattice(reference.lattice.matrix @ rotation),
        [site.species for site in reference],
        [site.frac_coords for site in reference],
    )
    return reference, {"permuted_origin": shifted, "wrapped": wrapped}, rotated


def run_benchmark():
    verified = verify_input_manifest(DATA, DATA / "manifest.json")
    pair_declarations = {
        "exact_homometric": (
            CASES / "homometric-a.cif", CASES / "homometric-b.cif", 0.08
        ),
        "near_homometric": (
            CASES / "homometric-a.cif", CASES / "near-homometric-b.cif", 0.08
        ),
        "realistic_compatible": (
            CASES / "realistic-compatible-a.cif",
            CASES / "realistic-compatible-b.cif",
            0.08,
        ),
        "resolution_broad": (
            CASES / "resolution-cubic.cif", CASES / "resolution-strained.cif", 0.20
        ),
        "resolution_high": (
            CASES / "resolution-cubic.cif", CASES / "resolution-strained.cif", 0.015
        ),
    }
    cases = {}
    profile_arrays = {}
    for name, (path_a, path_b, fwhm) in pair_declarations.items():
        coordinate, profile_a, profile_b = _profile_pair(path_a, path_b, fwhm=fwhm)
        metrics = profile_metric_suite(
            profile_a,
            profile_b,
            coordinate_step=Q_STEP,
            cross_correlation_tolerance=max(fwhm / 2.0, Q_STEP),
        )
        weighting = None
        if name not in {"resolution_broad", "resolution_high"}:
            weighting = _weighting_results(path_a, path_b)
        cases[name] = {
            "structure_a": str(path_a.relative_to(ROOT)),
            "structure_b": str(path_b.relative_to(ROOT)),
            "profile_fwhm_q": fwhm,
            "profile_metrics": metrics,
            "weighting": weighting,
        }
        profile_arrays[name] = (coordinate, profile_a, profile_b)

    reference, equivalent, rotated = _equivalent_structures()
    invariance = {}
    for name, candidate in equivalent.items():
        calculator_a = _calculator(reference)
        calculator_b = _calculator(candidate)
        table_a = calculator_a.reflection_table(domain="q")
        table_b = calculator_b.reflection_table(domain="q")
        match = match_reflections(table_a.hkl, table_b.hkl)
        invariance[name] = compare_weighting_schemes(
            match.hkl,
            np.asarray(table_a.q)[match.indices_a],
            np.asarray(table_a.structure_factor)[match.indices_a],
            np.asarray(table_b.structure_factor)[match.indices_b],
        )
    rotated_result = diagnose_structures(
        reference,
        rotated,
        q_range=Q_RANGE,
        q_step=0.02,
        profile_fwhm_q=0.08,
        pair_r_max=5.0,
    )
    invariance["rotated_cartesian_setting"] = {
        "relationship": asdict(rotated_result.relationship),
        "d_sf": rotated_result.mismatch.d_sf,
    }

    qmax_stability = {}
    for q_stop in (3.0, 4.0, 5.0, 6.0):
        qmax_stability[str(q_stop)] = _weighting_results(
            CASES / "homometric-a.cif", CASES / "near-homometric-b.cif", q_stop=q_stop
        )
    stability_cv = {}
    for scheme in WEIGHTING_SCHEMES:
        values = np.array(
            [qmax_stability[key][scheme]["d_sf"] for key in qmax_stability]
        )
        stability_cv[scheme] = float(values.std(ddof=0) / values.mean())

    exact = cases["exact_homometric"]
    cyclic_a = (0, 3, 4, 5)
    cyclic_b = (0, 1, 3, 4)
    difference_a = cyclic_difference_multiset(cyclic_a, 8)
    difference_b = cyclic_difference_multiset(cyclic_b, 8)
    construction = {
        "modulus": 8,
        "set_a": cyclic_a,
        "set_b": cyclic_b,
        "directed_difference_multiset_a": difference_a,
        "directed_difference_multiset_b": difference_b,
        "difference_multisets_equal": difference_a == difference_b,
        "dihedrally_equivalent": cyclic_sets_dihedrally_equivalent(cyclic_a, cyclic_b, 8),
    }
    broad = cases["resolution_broad"]["profile_metrics"]["cosine"]
    high = cases["resolution_high"]["profile_metrics"]["cosine"]
    invariant_values = [
        declaration[scheme]["d_sf"]
        for key, declaration in invariance.items()
        if key != "rotated_cartesian_setting"
        for scheme in WEIGHTING_SCHEMES
    ]
    gates = {
        "input_manifest_verified": len(verified) == 7,
        "homometric_construction_verified": (
            construction["difference_multisets_equal"]
            and not construction["dihedrally_equivalent"]
        ),
        "representation_invariance": max(invariant_values) < 1e-10,
        "rotated_setting_invariance": rotated_result.mismatch.d_sf < 1e-10,
        "homometric_intensity_equality": exact["profile_metrics"]["cosine"] > 1 - 1e-12,
        "homometric_phase_detection": (
            exact["weighting"]["shell_balanced_intensity"]["d_phase"] > 0.1
        ),
        "extinction_stable_amplitude": (
            exact["weighting"]["shell_balanced_intensity"]["d_amplitude"] < 1e-10
        ),
        "resolution_transition": broad - high > 0.04,
        "baseline_metrics_bounded": all(
            -1 <= value <= 1
            for case in cases.values()
            for value in case["profile_metrics"].values()
        ),
        "external_expert_review": None,
    }
    explanations = {
        "exact_homometric": (
            "The ideal powder profiles and all intensity-only baselines agree, while the "
            "phase-aware disk remains displaced vertically. The ambiguity is caused by "
            "phase loss, not peak broadening."
        ),
        "near_homometric": (
            "A controlled site perturbation weakly breaks intensity equality, but ordinary "
            "profile metrics remain close to one. Complex mismatch exposes the larger model "
            "difference before powder information loss."
        ),
        "resolution_limited": (
            "The strained model is almost indistinguishable with 0.20 inverse-angstrom "
            "FWHM, but its similarity falls at 0.015 inverse-angstrom FWHM. The ambiguity is "
            "therefore resolution-limited under the broad experiment."
        ),
        "weighting": (
            "Uniform reflection weighting gives numerical extinctions undue influence in "
            "the exact homometric pair. Intensity-based weights suppress that artifact; "
            "shell balancing additionally prevents one resolution region from taking all "
            "weight. The benchmark reports every scheme rather than hiding this choice."
        ),
    }
    return {
        "schema": PUBLICATION_SCHEMA,
        "benchmark_version": "1.0.0",
        "input_hashes": verified,
        "fixed_configuration": {
            "radiation": "xray",
            "wavelength_angstrom": 1.5406,
            "q_range_inverse_angstrom": Q_RANGE,
            "q_step_inverse_angstrom": Q_STEP,
            "shell_width_inverse_angstrom": 0.5,
        },
        "cases": cases,
        "homometric_construction": construction,
        "invariance": invariance,
        "qmax_stability": qmax_stability,
        "qmax_stability_coefficient_of_variation": stability_cv,
        "explanations": explanations,
        "gates": gates,
        "release_status": publication_gate_summary(gates),
        "scope_warning": (
            "Synthetic benchmark evidence only. External crystallographer review is unsigned, "
            "and the package does not validate arbitrary structure solution."
        ),
    }, profile_arrays


def _write_table(result):
    path = OUTPUT / "metric_table.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "case",
                "cosine",
                "pearson",
                "jensen_shannon",
                "gaussian_cross_correlation",
                "d_sf_shell_balanced",
                "d_amplitude_shell_balanced",
                "d_phase_shell_balanced",
            ]
        )
        for name, case in result["cases"].items():
            weighting = (case["weighting"] or {}).get("shell_balanced_intensity", {})
            writer.writerow(
                [
                    name,
                    case["profile_metrics"]["cosine"],
                    case["profile_metrics"]["pearson"],
                    case["profile_metrics"]["jensen_shannon"],
                    case["profile_metrics"]["gaussian_cross_correlation"],
                    weighting.get("d_sf"),
                    weighting.get("d_amplitude"),
                    weighting.get("d_phase"),
                ]
            )


def _write_review_packet(result):
    declarations = (
        ("blind-01", "exact_homometric"),
        ("blind-02", "near_homometric"),
        ("blind-03", "resolution_limited"),
        ("blind-04", "weighting"),
    )
    packet = {
        "schema": PUBLICATION_SCHEMA,
        "instructions": (
            "For each explanation, score scientific correctness, usefulness and whether the "
            "claimed information-loss mechanism follows from the supplied evidence."
        ),
        "reviewer_identity": None,
        "reviewer_affiliation": None,
        "review_date": None,
        "cases": [
            {
                "blind_id": blind,
                "explanation": result["explanations"][key],
                "scientific_correctness_1_to_5": None,
                "diagnostic_usefulness_1_to_5": None,
                "mechanism_supported": None,
                "comments": None,
            }
            for blind, key in declarations
        ],
        "signature": None,
    }
    key = {"schema": PUBLICATION_SCHEMA, "mapping": dict(declarations)}
    (OUTPUT / "expert_review_packet.json").write_text(
        json.dumps(packet, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "expert_review_key.json").write_text(
        json.dumps(key, indent=2) + "\n", encoding="utf-8"
    )


def _environment():
    packages = {}
    for name in ("braggcalculator", "numpy", "pymatgen", "matplotlib", "scipy", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "schema": PUBLICATION_SCHEMA,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "source_revision_at_generation": revision,
        "note": "The input manifest, results and analysis script are the reproducibility authority.",
    }


def _write_artifact_manifest():
    relative_paths = [
        "results.json",
        "metric_table.csv",
        "expert_review_packet.json",
        "expert_review_key.json",
        "requirements-lock.txt",
        "figures/diagnostic_benchmark.png",
        "figures/diagnostic_benchmark.pdf",
        "figures/diagnostic_benchmark.svg",
        "figures/weighting_invariance.png",
        "figures/weighting_invariance.pdf",
        "figures/weighting_invariance.svg",
    ]
    records = []
    for relative in relative_paths:
        path = OUTPUT / relative
        records.append(
            {
                "path": relative,
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    sources = []
    for path in (
        ROOT / "braggcalculator" / "publication.py",
        ROOT / "scripts" / "run_diagnostics_publication.py",
        DATA / "manifest.json",
    ):
        sources.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema": PUBLICATION_SCHEMA,
        "benchmark_version": "1.0.0",
        "generated_artifacts": records,
        "analysis_sources": sources,
        "environment_excluded_from_hash_gate": (
            "environment.json intentionally records the reproducing machine"
        ),
    }
    (OUTPUT / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _plot_main(result, profiles):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 150})
    plt.rcParams["svg.hashsalt"] = "braggcalculator-diagnostics-v1"
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.8), constrained_layout=True)
    colors = {"a": "#1565a7", "b": "#d56b1f", "phase": "#7756a5"}

    coordinate, profile_a, profile_b = profiles["exact_homometric"]
    axes[0, 0].plot(coordinate, profile_a / profile_a.max(), color=colors["a"], lw=1.0)
    axes[0, 0].plot(
        coordinate, profile_b / profile_b.max(), color=colors["b"], lw=0.8, ls="--"
    )
    axes[0, 0].set(title="a  Exact homometry: identical powder profile", xlabel="Q / Å⁻¹")
    axes[0, 0].text(0.02, 0.92, "cosine = 1.000000", transform=axes[0, 0].transAxes)

    hkl, q, factor_a, factor_b = _complex_pair(
        CASES / "homometric-a.cif", CASES / "homometric-b.cif"
    )
    weights = compare_weighting_schemes(hkl, q, factor_a, factor_b)[
        "shell_balanced_intensity"
    ]
    from braggcalculator.publication import mismatch_weights
    from braggcalculator.diagnostics import mismatch_disk

    disk = mismatch_disk(
        hkl,
        factor_a,
        factor_b,
        weights=mismatch_weights(
            q, np.abs(factor_a), np.abs(factor_b), scheme="shell_balanced_intensity"
        ),
        optimize_origin=True,
    )
    axes[0, 1].add_patch(plt.Circle((0, 0), 1, fill=False, color="#536777", lw=0.8))
    axes[0, 1].axhline(0, color="#d8dee5", lw=0.6)
    axes[0, 1].axvline(0, color="#d8dee5", lw=0.6)
    size = 5 + 80 * disk.weights / disk.weights.max()
    axes[0, 1].scatter(disk.x, disk.y, s=size, c=q, cmap="viridis", alpha=0.65)
    axes[0, 1].set(
        title="b  Phase-aware mismatch disk",
        xlabel="amplitude coordinate",
        ylabel="phase coordinate",
        xlim=(-1.05, 1.05),
        ylim=(-1.05, 1.05),
        aspect="equal",
    )
    axes[0, 1].text(
        0.03,
        0.04,
        f"Damp={weights['d_amplitude']:.3g}\nDphase={weights['d_phase']:.3f}",
        transform=axes[0, 1].transAxes,
    )

    schemes = list(WEIGHTING_SCHEMES)
    exact_weights = result["cases"]["exact_homometric"]["weighting"]
    x = np.arange(len(schemes))
    axes[0, 2].bar(
        x - 0.18,
        [exact_weights[name]["d_amplitude"] for name in schemes],
        width=0.36,
        label="amplitude",
        color=colors["a"],
    )
    axes[0, 2].bar(
        x + 0.18,
        [exact_weights[name]["d_phase"] for name in schemes],
        width=0.36,
        label="phase",
        color=colors["phase"],
    )
    axes[0, 2].set_xticks(x, ["uniform", "intensity", "sqrt-I", "shell-I"], rotation=20)
    axes[0, 2].set(title="c  Weighting changes interpretation", ylabel="RMS disk component")
    axes[0, 2].legend(frameon=False)

    for label, style in (("resolution_broad", "-"), ("resolution_high", "--")):
        coordinate, first, second = profiles[label]
        difference = np.abs(first / first.max() - second / second.max())
        axes[1, 0].plot(coordinate, difference, ls=style, lw=1.0, label=label.split("_")[1])
    axes[1, 0].set(
        title="d  Resolution exposes strained-cell differences",
        xlabel="Q / Å⁻¹",
        ylabel="|normalized difference|",
    )
    axes[1, 0].legend(frameon=False)

    case_names = list(result["cases"])
    metric_names = ["cosine", "pearson", "jensen_shannon", "gaussian_cross_correlation"]
    matrix = np.array(
        [[result["cases"][case]["profile_metrics"][metric] for metric in metric_names]
         for case in case_names]
    )
    image = axes[1, 1].imshow(matrix, vmin=0.9, vmax=1.0, cmap="magma")
    axes[1, 1].set_xticks(range(4), ["cos", "Pearson", "JS", "weighted CC"], rotation=25)
    axes[1, 1].set_yticks(range(len(case_names)), [name.replace("_", " ") for name in case_names])
    axes[1, 1].set(title="e  Intensity/profile baselines saturate")
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046)

    layers = ["complex", "intensity", "broad profile", "high-res profile"]
    values = [
        1 - exact_weights["shell_balanced_intensity"]["d_sf"],
        result["cases"]["exact_homometric"]["profile_metrics"]["cosine"],
        result["cases"]["resolution_broad"]["profile_metrics"]["cosine"],
        result["cases"]["resolution_high"]["profile_metrics"]["cosine"],
    ]
    axes[1, 2].barh(layers, values, color=[colors["phase"], colors["a"], "#16866a", "#b58a18"])
    axes[1, 2].set(xlim=(0, 1.02), title="f  Similarity depends on information level")
    for index, value in enumerate(values):
        axes[1, 2].text(value - 0.01, index, f"{value:.3f}", ha="right", va="center", color="white")

    _save_figure(figure, "diagnostic_benchmark")
    plt.close(figure)


def _plot_robustness(result):
    plt.rcParams["svg.hashsalt"] = "braggcalculator-diagnostics-v1"
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), constrained_layout=True)
    schemes = list(WEIGHTING_SCHEMES)
    transformations = [key for key in result["invariance"] if key != "rotated_cartesian_setting"]
    for transform in transformations:
        values = [max(result["invariance"][transform][scheme]["d_sf"], 1e-16) for scheme in schemes]
        axes[0].plot(schemes, values, marker="o", label=transform)
    rotated = max(result["invariance"]["rotated_cartesian_setting"]["d_sf"], 1e-16)
    axes[0].axhline(rotated, color="#7756a5", ls="--", label="rotated setting")
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(title="a  Representation invariance", ylabel="Dsf (log scale)")
    axes[0].legend(frameon=False, fontsize=7)

    qmax = [float(value) for value in result["qmax_stability"]]
    for scheme in schemes:
        axes[1].plot(
            qmax,
            [result["qmax_stability"][str(value)][scheme]["d_sf"] for value in qmax],
            marker="o",
            label=scheme,
        )
    axes[1].set(title="b  Reflection-range sensitivity", xlabel="Qmax / Å⁻¹", ylabel="Dsf")
    axes[1].legend(frameon=False, fontsize=6)

    gates = result["gates"]
    gate_names = [name.replace("_", " ") for name in gates]
    gate_values = [1 if value is True else 0.5 if value is None else 0 for value in gates.values()]
    colors = ["#16866a" if value is True else "#b58a18" if value is None else "#b83c35" for value in gates.values()]
    axes[2].barh(gate_names, gate_values, color=colors)
    axes[2].set(xlim=(0, 1.05), title="c  Publication release gates", xticks=[0, 0.5, 1], xticklabels=["fail", "pending", "pass"])
    _save_figure(figure, "weighting_invariance")
    plt.close(figure)


def write_outputs(result, profiles):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(
        json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8"
    )
    environment = _environment()
    (OUTPUT / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    locked = [
        f"{name}=={value}"
        for name, value in environment["packages"].items()
        if name != "braggcalculator" and value is not None
    ]
    (OUTPUT / "requirements-lock.txt").write_text(
        "# Exact packages used for the frozen benchmark environment.\n"
        + "\n".join(locked)
        + "\n",
        encoding="utf-8",
    )
    _write_table(result)
    _write_review_packet(result)
    _plot_main(result, profiles)
    _plot_robustness(result)
    _write_artifact_manifest()


def verify_gates(result):
    failed = [name for name, value in result["gates"].items() if value is False]
    if failed:
        raise SystemExit(f"publication benchmark gates failed: {', '.join(failed)}")
    print(f"release status: {result['release_status']}")
    for name, value in result["gates"].items():
        print(f"  {name}: {'PASS' if value is True else 'PENDING' if value is None else 'FAIL'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="fail when a numerical gate fails")
    args = parser.parse_args(argv)
    result, profiles = run_benchmark()
    write_outputs(result, profiles)
    if args.verify:
        verify_gates(result)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
