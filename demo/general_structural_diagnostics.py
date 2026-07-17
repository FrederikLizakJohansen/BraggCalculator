#!/usr/bin/env python3
"""Generate the Milestone 5 relationship-aware diagnostics evidence report."""

from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Lattice, Structure

from braggcalculator import (
    BraggCalculator,
    classify_structural_relationship,
    diagnose_structures,
    suggest_measurements,
)
from braggcalculator.profiles import GaussianProfileQ


DEFAULT_FIGURE = Path(__file__).with_name("general_structural_diagnostics.png")
DEFAULT_REPORT = Path(__file__).with_name("general_structural_diagnostics_report.html")


def _models():
    compatible_a = Structure(
        Lattice.from_parameters(4.2, 5.1, 6.3, 78, 82, 73),
        ["Si", "O", "O"],
        [[0.13, 0.21, 0.34], [0.31, 0.47, 0.11], [0.72, 0.08, 0.59]],
    )
    compatible_b = compatible_a.copy()
    compatible_b.translate_sites([1], [0.04, 0.0, 0.0], frac_coords=True)
    compatible_b.translate_sites([2], [0.0, -0.025, 0.015], frac_coords=True)
    equivalent = Structure(
        compatible_a.lattice,
        [site.species for site in reversed(compatible_a)],
        [
            (site.frac_coords + np.array([0.125, 0.25, 0.375])) % 1.0
            for site in reversed(compatible_a)
        ],
    )
    parent = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    ordered = parent.copy()
    ordered.make_supercell([2, 1, 1])
    ordered.replace(1, "P")
    unrelated = Structure(
        Lattice.hexagonal(3.05, 5.37),
        ["Si", "O", "O"],
        [[0, 0, 0], [1 / 3, 2 / 3, 0.22], [2 / 3, 1 / 3, 0.78]],
    )
    split_a = Structure(
        Lattice.cubic(4.0), ["Si", "O"], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    split_b = Structure(
        Lattice.cubic(4.04), ["Si", "O"], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    return {
        "compatible_a": compatible_a,
        "compatible_b": compatible_b,
        "equivalent": equivalent,
        "parent": parent,
        "ordered": ordered,
        "unrelated": unrelated,
        "split_a": split_a,
        "split_b": split_b,
    }


def _build_results():
    models = _models()
    relationships = {
        "equivalent": classify_structural_relationship(
            models["compatible_a"], models["equivalent"]
        ),
        "compatible": classify_structural_relationship(
            models["compatible_a"], models["compatible_b"]
        ),
        "commensurate": classify_structural_relationship(models["parent"], models["ordered"]),
        "unrelated": classify_structural_relationship(
            models["compatible_a"], models["unrelated"]
        ),
    }
    compatible = diagnose_structures(
        models["compatible_a"],
        models["compatible_b"],
        q_range=(0.5, 5.0),
        q_step=0.025,
        profile_fwhm_q=0.18,
        count_scale=30.0,
        pair_r_max=6.0,
        site_groups={"Si": [0], "O(1)": [1], "O(2)": [2]},
        counterfactual_groups={"O(1)": [1], "O(2)": [2], "both oxygen sites": [1, 2]},
    )
    commensurate = diagnose_structures(
        models["parent"],
        models["ordered"],
        q_range=(0.5, 5.0),
        q_step=0.025,
        profile_fwhm_q=0.10,
        count_scale=30.0,
        pair_r_max=6.0,
    )
    unrelated = diagnose_structures(
        models["compatible_a"],
        models["unrelated"],
        q_range=(0.5, 5.0),
        q_step=0.025,
        profile_fwhm_q=0.15,
        count_scale=30.0,
        pair_r_max=6.0,
    )
    measurements = suggest_measurements(
        models["split_a"],
        models["split_b"],
        [
            {
                "name": "standard Cu X-ray",
                "radiation": "xray",
                "wavelength": 1.5406,
                "q_range": (0.5, 5.0),
                "q_step": 0.01,
                "fwhm_q": 0.25,
                "count_scale": 1000.0,
                "background_density": 1.0,
            },
            {
                "name": "high-resolution Cu X-ray",
                "radiation": "xray",
                "wavelength": 1.5406,
                "q_range": (0.5, 5.0),
                "q_step": 0.01,
                "fwhm_q": 0.04,
                "count_scale": 1000.0,
                "background_density": 1.0,
            },
            {
                "name": "high-resolution Mo X-ray",
                "radiation": "xray",
                "wavelength": 0.7107,
                "q_range": (0.5, 5.0),
                "q_step": 0.01,
                "fwhm_q": 0.04,
                "count_scale": 1000.0,
                "background_density": 1.0,
            },
            {
                "name": "neutron (declared exposure)",
                "radiation": "neutron",
                "wavelength": 1.8,
                "q_range": (0.5, 5.0),
                "q_step": 0.01,
                "fwhm_q": 0.08,
                "count_scale": 100000.0,
                "background_density": 1.0,
            },
        ],
    )
    return models, relationships, compatible, commensurate, unrelated, measurements


def _supercell_lines(models):
    calculators = []
    for structure in (models["parent"], models["ordered"]):
        calculators.append(
            BraggCalculator(
                primitive=False,
                q_range=(0.5, 5.0),
                q_step=0.025,
                profile_q=GaussianProfileQ(0.1),
            ).load(structure)
        )
    return tuple(calculator.line_pattern(domain="q", scaled=True) for calculator in calculators)


def _plot(results, output):
    models, relationships, compatible, commensurate, unrelated, measurements = results
    figure, axes = plt.subplots(4, 2, figsize=(15, 17), constrained_layout=True)

    relationship_names = tuple(relationships)
    regimes = [{"I": 1, "II": 2, "III": 3}[relationships[name].regime] for name in relationship_names]
    colors = ["#009E73", "#56B4E9", "#E69F00", "#D55E00"]
    axes[0, 0].bar(relationship_names, regimes, color=colors)
    axes[0, 0].set(
        yticks=(1, 2, 3),
        yticklabels=("I: direct", "II: common cell", "III: powder/PDF"),
        ylim=(0, 3.4),
        title="Relationship gate selects valid mathematics",
    )
    axes[0, 0].tick_params(axis="x", rotation=15)

    ladder_names = ("complex", "intensity", "ideal_powder", "profile", "radial_pair")
    ladder_values = [compatible.similarities[name] for name in ladder_names]
    axes[0, 1].plot(range(4), ladder_values[:4], "o-", lw=2, label="diffraction ladder")
    axes[0, 1].scatter(
        [4], [ladder_values[4]], marker="D", s=55, color="#D55E00", label="alternative real-space view"
    )
    axes[0, 1].axvline(3.5, color="0.6", ls=":", lw=1)
    axes[0, 1].set(
        xticks=range(len(ladder_names)),
        xticklabels=("complex F", "|F|²", "ideal powder", "broadened", "radial PDF"),
        ylabel="cosine / bounded similarity",
        ylim=(0, 1.03),
        title=f"Information ladder: {compatible.dominant_information_loss.replace('_', ' ')}",
    )
    axes[0, 1].tick_params(axis="x", rotation=18)
    axes[0, 1].legend(fontsize=8)

    (parent_q, parent_i), (ordered_q, ordered_i) = _supercell_lines(models)
    axes[1, 0].vlines(parent_q, 0, parent_i, color="#0072B2", lw=1.5, label="parent")
    axes[1, 0].vlines(ordered_q, 0, -ordered_i, color="#D55E00", lw=1.2, label="ordered 2× cell")
    superstructure = commensurate.superstructure
    if superstructure is not None:
        axes[1, 0].scatter(
            superstructure.q,
            np.full(len(superstructure.q), -105),
            marker="|",
            color="black",
            label="non-parent indices",
        )
    axes[1, 0].axhline(0, color="black", lw=0.6)
    axes[1, 0].set(
        xlabel=r"Q ($\AA^{-1}$)",
        ylabel="scaled line intensity",
        title=(
            "Ordered supercell: "
            f"{100 * superstructure.intensity_fraction:.3f}% in superstructure reflections"
        ),
    )
    axes[1, 0].legend(fontsize=8)

    peak_groups = compatible.peak_groups_a[:8]
    site_names = tuple(peak_groups[0].site_effects)
    matrix = np.asarray(
        [[group.site_effects[name] for group in peak_groups] for name in site_names]
    )
    image = axes[1, 1].imshow(matrix, aspect="auto", cmap="magma")
    axes[1, 1].set(
        yticks=range(len(site_names)),
        yticklabels=site_names,
        xticks=range(len(peak_groups)),
        xticklabels=[f"{group.q_center:.2f}" for group in peak_groups],
        xlabel=r"peak-group Q ($\AA^{-1}$)",
        title="Non-additive site-removal effect by resolved peak group",
    )
    figure.colorbar(image, ax=axes[1, 1], label="|I − I(without site)| / I")

    counterfactuals = compatible.counterfactuals
    x = np.arange(len(counterfactuals))
    axes[2, 0].bar(
        x - 0.18,
        [item.effect_norm for item in counterfactuals],
        0.36,
        label="relative effect norm",
    )
    axes[2, 0].bar(
        x + 0.18,
        [item.alignment_fraction for item in counterfactuals],
        0.36,
        label="alignment with A→B difference",
    )
    axes[2, 0].set(
        xticks=x,
        xticklabels=[item.name for item in counterfactuals],
        ylim=(0, 1.1),
        title="Counterfactual site substitutions (interference-aware)",
    )
    axes[2, 0].tick_params(axis="x", rotation=15)
    axes[2, 0].legend(fontsize=8)

    pair = unrelated.pair_distribution
    axes[2, 1].plot(pair.radius, pair.distribution_a, label="triclinic SiO₂ model")
    axes[2, 1].plot(pair.radius, pair.distribution_b, label="hexagonal SiO₂ model")
    axes[2, 1].set(
        xlabel=r"pair distance ($\AA$)",
        ylabel="normalized scattering-weighted PDF",
        title=f"Unrelated lattices retain radial comparison: S={pair.similarity:.3f}",
    )
    axes[2, 1].legend(fontsize=8)

    names = [item.name for item in measurements]
    discrimination = [item.total_discrimination for item in measurements]
    axes[3, 0].barh(names[::-1], discrimination[::-1], color="#0072B2")
    axes[3, 0].set(
        xlabel=r"expected count-model separation ($\Delta\chi^2$ approximation)",
        xscale="log",
        title="Declared experiment configurations are ranked quantitatively",
    )

    coordinate = unrelated.profile_discrimination.coordinate
    profile_a = unrelated.profile_discrimination.expected_a
    profile_b = unrelated.profile_discrimination.expected_b
    axes[3, 1].plot(coordinate, profile_a, label="triclinic expected counts")
    axes[3, 1].plot(coordinate, profile_b, label="hexagonal expected counts")
    axes[3, 1].fill_between(
        coordinate,
        profile_a,
        profile_b,
        color="#E69F00",
        alpha=0.25,
        label="measurable difference",
    )
    axes[3, 1].set(
        xlabel=r"Q ($\AA^{-1}$)",
        ylabel="expected counts / bin",
        title="Regime III: powder evidence without invented hkl phases",
    )
    axes[3, 1].legend(fontsize=8)

    figure.suptitle("General structural diagnostics for materials characterization", fontsize=18)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_report(results, figure_path, report_path):
    _, relationships, compatible, commensurate, unrelated, measurements = results
    encoded = base64.b64encode(figure_path.read_bytes()).decode("ascii")
    relationship_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{item.regime}</td>"
        f"<td>{html.escape(item.classification)}</td><td>{item.complex_comparison_allowed}</td>"
        f"<td>{html.escape(item.reason)}</td></tr>" for name, item in relationships.items()
    )
    similarity_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{'not valid' if value is None else f'{value:.6f}'}</td></tr>"
        for name, value in compatible.similarities.items()
    )
    measurement_rows = "".join(
        f"<tr><td>{rank}</td><td>{html.escape(item.name)}</td><td>{item.radiation}</td>"
        f"<td>{item.wavelength:.4g}</td><td>{item.fwhm_q:.4g}</td>"
        f"<td>{item.total_discrimination:.6g}</td><td>{item.most_informative_q:.4g}</td></tr>"
        for rank, item in enumerate(measurements, 1)
    )
    counterfactual_rows = "".join(
        f"<tr><td>{html.escape(item.name)}</td><td>{list(item.site_indices)}</td>"
        f"<td>{item.effect_norm:.6f}</td><td>{item.alignment_fraction:.6f}</td>"
        f"<td>{item.largest_effect_coordinate:.4f}</td></tr>"
        for item in compatible.counterfactuals
    )
    superstructure = commensurate.superstructure
    report_path.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>General structural diagnostics</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:2rem auto;line-height:1.5}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}}code{{background:#eee;padding:.1rem .3rem}}</style></head><body>
<h1>General structural diagnostics</h1><p>This deterministic report demonstrates relationship-gated diagnostics. Complex phase comparisons are emitted only when a reciprocal mapping exists. Counterfactual and site-removal effects are non-additive because structure-factor interference remains in the calculation.</p>
<img alt="general structural diagnostics evidence" src="data:image/png;base64,{encoded}">
<h2>Relationship classification</h2><table><tr><th>Pair</th><th>Regime</th><th>Class</th><th>Complex valid</th><th>Evidence</th></tr>{relationship_rows}</table>
<h2>Compatible-pair information ladder</h2><p><strong>Classification:</strong> {html.escape(compatible.dominant_information_loss)} — {html.escape(compatible.explanation)}</p><table><tr><th>Level</th><th>Similarity</th></tr>{similarity_rows}</table>
<h2>Superstructure</h2><p>Transformation <code>{html.escape(str(superstructure.transformation.tolist()))}</code>; {len(superstructure.hkl)} calculated non-parent reciprocal points; intensity fraction {superstructure.intensity_fraction:.6g}.</p>
<h2>Counterfactual attribution</h2><table><tr><th>Group</th><th>Sites</th><th>Relative effect</th><th>Alignment</th><th>Largest-effect Q</th></tr>{counterfactual_rows}</table>
<h2>Unrelated pair</h2><p>Complex similarity: <strong>not valid / absent</strong>. Powder expected separation: {unrelated.profile_discrimination.total_discrimination:.6g}. Radial-pair similarity: {unrelated.pair_distribution.similarity:.6f}.</p>
<h2>Experiment ranking</h2><table><tr><th>Rank</th><th>Name</th><th>Radiation</th><th>Wavelength Å</th><th>FWHM Q</th><th>Expected separation</th><th>Top Q</th></tr>{measurement_rows}</table>
<p>Experiment scores assume the separately declared count scale and background for each configuration. They are predictions under the kinematic single-phase forward model, Gaussian Q-profile and symmetric mean-count Poisson variance approximation; they are not universal instrument rankings.</p>
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
    _, _, compatible, commensurate, unrelated, measurements = results
    print(f"figure: {arguments.figure}")
    print(f"report: {arguments.report}")
    print(f"information loss: {compatible.dominant_information_loss}")
    print(f"superstructure fraction: {commensurate.superstructure.intensity_fraction:.6g}")
    print(f"unrelated complex metric: {unrelated.similarities['complex']}")
    print(f"best experiment: {measurements[0].name}")


if __name__ == "__main__":
    main()
