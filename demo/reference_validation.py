#!/usr/bin/env python3
"""Generate the Milestone 6 reference-validation matrix and evidence report."""

from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Lattice, Structure

from benchmarks.reference_cases import reference_structures
from braggcalculator import (
    BraggCalculator,
    DiffractionDataset,
    ValidationCase,
    ValidationMatrix,
    ValidationMetric,
    load_reference_sources,
    suggest_measurements,
    validate_line_oracle,
    validate_public_sources,
)
from braggcalculator.profiles import GaussianProfileQ
from demo.characterize_nist_lab6 import characterize
from demo.general_structural_diagnostics import _models
from demo.refine_anisotropic_restraints import _anisotropic_problem
from demo.refine_occupancy_adp import (
    _controlled_recovery,
    _joint_session,
    _structure,
    _synthetic_dataset,
)
from demo.refine_rigid_multiphase import _mixture_problem, _rigid_problem
from demo.refine_staged import run_staged_example
from demo.refine_symmetry_coordinates import run_refinement
from demo.refine_symmetry_lattice import run_lattice_refinement


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE = Path(__file__).with_name("reference_validation.png")
DEFAULT_REPORT = Path(__file__).with_name("reference_validation_report.html")
DEFAULT_JSON = Path(__file__).with_name("reference_validation_results.json")


def _maximum_metric(name, value, unit, passed, warned, explanation=""):
    return ValidationMetric(
        name, float(value), unit, "maximum", float(passed), float(warned),
        explanation=explanation,
    )


def _synthetic_recoveries():
    cases = []
    summary = {}

    lattice = run_lattice_refinement()
    lattice_error = lattice["maximum_error"]
    cases.append(
        ValidationCase(
            "recovery:lattice", "synthetic_recovery",
            "Two symmetry-allowed tetragonal metric modes",
            (_maximum_metric("maximum_cell_error", lattice_error, "A or degrees", 5e-5, 5e-4),),
        )
    )
    summary["lattice"] = lattice_error / 5e-5

    target, recovered, _, _ = run_refinement()
    coordinate_error = np.max(np.abs(recovered - target))
    cases.append(
        ValidationCase(
            "recovery:coordinates", "synthetic_recovery",
            "Symmetry-linked general-position coordinates",
            (_maximum_metric("maximum_fractional_error", coordinate_error, "fractional", 1e-5, 1e-4),),
        )
    )
    summary["coordinates"] = coordinate_error / 1e-5

    staged = run_staged_example()
    nuisance_tolerances = {"scale": 2e-3, "background": 1.0, "zero_shift": 2e-5, "fwhm": 2e-4}
    nuisance_metrics = []
    ratios = []
    for name, tolerance in nuisance_tolerances.items():
        error = abs(staged["recovered_physical"][name] - staged["target_physical"][name])
        nuisance_metrics.append(_maximum_metric(f"{name}_error", error, "", tolerance, 10 * tolerance))
        ratios.append(error / tolerance)
    cases.append(
        ValidationCase(
            "recovery:profile-nuisance", "synthetic_recovery",
            "Scale, background, zero shift and profile width in a staged joint fit",
            tuple(nuisance_metrics),
        )
    )
    summary["profile/nuisance"] = max(ratios)

    structure = _structure()
    dataset, target_occupancy, target_b_iso = _synthetic_dataset(structure)
    controlled = _controlled_recovery(structure, target_occupancy, target_b_iso)
    occupancy_error = abs(controlled["occupancy"][0]["species"]["Ca"] - 0.55)
    b_error = np.max(
        np.abs(np.array([item["B_iso"] for item in controlled["b_iso"]]) - [0.6, 0.35, 1.1])
    )
    cases.append(
        ValidationCase(
            "recovery:occupancy-biso", "synthetic_recovery",
            "Staged shared-site composition and isotropic displacement recovery",
            (
                _maximum_metric("Ca_fraction_error", occupancy_error, "fraction", 1e-5, 1e-4),
                _maximum_metric("maximum_Biso_error", b_error, "A^2", 1e-5, 1e-4),
            ),
        )
    )
    summary["occupancy"] = occupancy_error / 1e-5
    summary["Biso"] = b_error / 1e-5

    _, u_candidate, target_u, recovered_u = _anisotropic_problem()
    u_error = np.max(np.abs(recovered_u - target_u))
    cases.append(
        ValidationCase(
            "recovery:anisotropic-u", "synthetic_recovery",
            "Positive anisotropic displacement tensor from a whole profile",
            (
                _maximum_metric("maximum_U_error", u_error, "A^2", 1e-4, 5e-4),
                _maximum_metric("Rwp", u_candidate.r_wp, "", 1e-3, 1e-2),
            ),
        )
    )
    summary["Uaniso"] = u_error / 1e-4

    rigid = _rigid_problem()
    rigid_error = np.max(np.abs(rigid["recovered_values"] - rigid["target_values"]))
    distance_error = np.max(np.abs(rigid["recovered_distances"] - rigid["initial_distances"]))
    cases.append(
        ValidationCase(
            "recovery:rigid-pose", "synthetic_recovery",
            "Six rigid-body translation/rotation modes with invariant internal distances",
            (
                _maximum_metric("maximum_raw_pose_error", rigid_error, "raw mode", 2e-4, 2e-3),
                _maximum_metric("maximum_internal_distance_change", distance_error, "A", 1e-10, 1e-8),
            ),
        )
    )
    summary["rigid pose"] = max(rigid_error / 2e-4, distance_error / 1e-10)

    _, mixture, target_fractions, weak = _mixture_problem()
    recovered_fractions = np.array(
        [mixture.phase_fractions[name] for name in mixture.phase_names]
    )
    fraction_error = np.max(np.abs(recovered_fractions - target_fractions))
    cases.append(
        ValidationCase(
            "recovery:phase-fractions", "synthetic_recovery",
            "Positive two-phase profile-area fractions constrained to sum to one",
            (
                _maximum_metric("maximum_fraction_error", fraction_error, "fraction", 5e-4, 5e-3),
                _maximum_metric("fraction_sum_error", abs(recovered_fractions.sum() - 1), "", 1e-12, 1e-9),
            ),
        )
    )
    summary["phase fraction"] = fraction_error / 5e-4
    joint = _joint_session(
        dataset, structure, Path(__file__).with_name("occupancy_adp_report.html")
    )
    return tuple(cases), summary, weak, joint


def _difficult_cases(weak, joint):
    models = _models()
    resolution = suggest_measurements(
        models["split_a"], models["split_b"],
        [
            {"name": "broad", "q_range": (0.5, 5.0), "q_step": 0.02, "fwhm_q": 0.25,
             "count_scale": 1000.0},
            {"name": "resolved", "q_range": (0.5, 5.0), "q_step": 0.02, "fwhm_q": 0.04,
             "count_scale": 1000.0},
        ],
    )
    by_name = {item.name: item for item in resolution}
    resolution_ratio = (
        by_name["resolved"].total_discrimination / by_name["broad"].total_discrimination
    )

    base = Structure(
        Lattice.cubic(7.0), ["C", "H", "H"],
        [[0.25, 0.25, 0.25], [0.35, 0.25, 0.25], [0.25, 0.35, 0.25]],
    )
    shifted = base.copy()
    shifted.translate_sites([1], [0.04, 0.0, 0.0], frac_coords=True)

    def normalized_difference(mode):
        profiles = []
        for structure in (base, shifted):
            calculator = BraggCalculator(
                mode=mode, primitive=False, q_range=(0.5, 5.0), q_step=0.02,
                profile_q=GaussianProfileQ(0.08),
            ).load(structure)
            profiles.append(np.asarray(calculator.pattern(domain="q")[1]))
        normalized = [profile / np.linalg.norm(profile) for profile in profiles]
        return float(np.linalg.norm(normalized[1] - normalized[0]))

    xray_h = normalized_difference("xray")
    neutron_h = normalized_difference("neutron")
    contrast_ratio = neutron_h / xray_h
    trace_score = weak.phase_detectability["CsCl"]
    names = joint.identifiability["parameter_names"]
    correlation = np.asarray(joint.identifiability["correlation"])
    occupancy_indices = [i for i, name in enumerate(names) if name.startswith("occupancies.")]
    b_indices = [i for i, name in enumerate(names) if name.startswith("b_iso.")]
    occupancy_b_correlation = float(
        np.max(np.abs(correlation[np.ix_(occupancy_indices, b_indices)]))
    )
    correlation_warning = float(
        any("Occupancy and Biso" in warning for warning in joint.warnings)
    )
    cases = (
        ValidationCase(
            "difficult:overlap", "difficult_case",
            "Resolution-aware discrimination increases when nearby features are resolved",
            (ValidationMetric("resolved_to_broad_information", resolution_ratio, "ratio", "minimum", 2.0, 1.1),),
        ),
        ValidationCase(
            "difficult:weak-hydrogen", "difficult_case",
            "Radiation choice changes sensitivity to a displaced hydrogen site",
            (ValidationMetric("neutron_to_xray_profile_change", contrast_ratio, "ratio", "minimum", 1.2, 1.0),),
            assumptions=("Profiles are unit-norm; this is contrast sensitivity, not a count-time comparison",),
        ),
        ValidationCase(
            "difficult:trace-phase", "difficult_case",
            "A numerically present 0.03% phase remains below the declared detectability threshold",
            (_maximum_metric("trace_phase_detectability", trace_score, "sigma-like norm", 3.0, 4.0),),
            warnings=("A recovered numerical phase fraction is not by itself evidence of detection",),
        ),
        ValidationCase(
            "difficult:occupancy-biso-correlation", "difficult_case",
            "The local Jacobian exposes occupancy/displacement ambiguity and emits a guardrail",
            (
                ValidationMetric(
                    "maximum_cross_group_correlation", occupancy_b_correlation, "", "minimum",
                    0.85, 0.75,
                ),
                ValidationMetric(
                    "warning_emitted", correlation_warning, "boolean", "minimum", 1.0, 1.0,
                ),
            ),
            warnings=("Passing this case means detecting ambiguity, not identifying both parameters",),
        ),
        ValidationCase(
            "difficult:preferred-orientation", "difficult_case",
            "Preferred-orientation refinement is not implemented in the current forward model",
            declared_status="unsupported",
            warnings=("Do not interpret systematic family-dependent residuals as structural until texture is modelled",),
        ),
        ValidationCase(
            "difficult:time-of-flight", "difficult_case",
            "Neutron time-of-flight peak positions and profiles require a separate instrument model",
            declared_status="unsupported",
        ),
    )
    values = {
        "resolution information ratio": resolution_ratio,
        "neutron/H contrast ratio": contrast_ratio,
        "trace phase detectability": trace_score,
        "occupancy/B correlation": occupancy_b_correlation,
    }
    return cases, values


def build_validation_matrix(*, nist_report: Path):
    sources = load_reference_sources(ROOT / "data/reference_validation/manifest.json")
    cases = list(validate_line_oracle(reference_structures()))
    cases.extend(validate_public_sources(ROOT, sources))

    nist = characterize(nist_report)
    candidate = nist.candidates[0]
    refined_a = float(candidate.physical_parameters["lattice"][0][0])
    reference_a = float(nist.dataset.metadata["reference_lattice_angstrom"])
    expanded_uncertainty = float(
        nist.dataset.metadata["reference_lattice_expanded_uncertainty_angstrom"]
    )
    cases.append(
        ValidationCase(
            "reference-parameter:nist-srm660c", "reference_parameter",
            "LaB6 lattice refined against NIST SRM 660c laboratory data",
            (
                _maximum_metric(
                    "absolute_lattice_error", abs(refined_a - reference_a), "A",
                    expanded_uncertainty, 5 * expanded_uncertainty,
                    "Pass is the certificate's expanded uncertainty; warn allows model development",
                ),
                _maximum_metric("Rwp", candidate.r_wp, "", 0.15, 0.25),
            ),
            source_identifiers=("NIST-SRM-660c-scan-100a",),
            assumptions=tuple(nist.dataset.metadata["model_limitations"]),
        )
    )

    recoveries, recovery_summary, weak, joint = _synthetic_recoveries()
    cases.extend(recoveries)
    difficult, difficult_values = _difficult_cases(weak, joint)
    cases.extend(difficult)
    cases.append(
        ValidationCase(
            "reference-software:full-profile", "reference_software",
            "Direct final-profile and covariance reproduction from a frozen GSAS-II project",
            declared_status="pending_review",
            warnings=(
                "The corpus is ingested and pymatgen line patterns are matched, but no frozen GSAS-II final project is yet reproduced.",
            ),
        )
    )
    matrix = ValidationMatrix(
        cases=tuple(cases),
        sources=sources,
        required_categories=(
            "line_oracle", "public_data", "reference_parameter", "synthetic_recovery",
            "difficult_case", "reference_software",
        ),
        expert_review_status="pending_review",
        expert_review_checklist=(
            "Check whether each generated conclusion is crystallographically justified by its cited metric.",
            "Review the NIST mismatch and decide whether the compact profile model is fit for the intended claim.",
            "Review weak-scatterer, overlap, trace-phase and occupancy/ADP warnings for false reassurance.",
            "Sign and date the frozen JSON artifact without changing numerical results.",
        ),
        metadata={
            "scope": "constant-wavelength powder diffraction",
            "nist_refined_a_angstrom": refined_a,
            "nist_reference_a_angstrom": reference_a,
            "nist_rwp": candidate.r_wp,
        },
    )
    evidence = {
        "nist": nist,
        "recovery_summary": recovery_summary,
        "difficult_values": difficult_values,
    }
    return matrix, evidence


def _plot(matrix, evidence, output):
    figure, axes = plt.subplots(3, 2, figsize=(15, 15), constrained_layout=True)
    oracle = [case for case in matrix.cases if case.category == "line_oracle"]
    labels = [case.identifier.removeprefix("line-oracle:") for case in oracle]
    position = [case.metrics[1].value for case in oracle]
    intensity = [case.metrics[2].value for case in oracle]
    x = np.arange(len(labels))
    axes[0, 0].semilogy(x, np.maximum(position, 1e-17), "o", label="2theta error (deg)")
    axes[0, 0].semilogy(x, np.maximum(intensity, 1e-17), "s", label="scaled intensity error")
    axes[0, 0].axhline(1e-9, color="0.4", ls="--", lw=0.8)
    axes[0, 0].set(xticks=x, xticklabels=labels, title="Independent pymatgen line oracle", ylabel="maximum absolute error")
    axes[0, 0].tick_params(axis="x", rotation=65, labelsize=7)
    axes[0, 0].legend(fontsize=8)

    for offset, source in enumerate(matrix.sources):
        reader = (
            DiffractionDataset.from_xye
            if "xye" in source.data_format.lower()
            else DiffractionDataset.from_gsas_constant_step
        )
        dataset = reader(
            ROOT / source.relative_path, wavelength=source.wavelength_angstrom,
            radiation=source.radiation,
        )
        scaled = dataset.intensity / max(np.percentile(dataset.intensity, 99.5), 1)
        label = f"{source.material} ({source.radiation})"
        axes[0, 1].plot(dataset.coordinate, scaled + offset * 1.25, lw=0.55, label=label)
    axes[0, 1].set(xlabel="2theta (degrees)", yticks=[], title="Checksummed public X-ray/neutron corpus")
    axes[0, 1].legend(fontsize=8)

    recovery = evidence["recovery_summary"]
    bars = axes[1, 0].bar(list(recovery), list(recovery.values()), color="#0072B2")
    axes[1, 0].axhline(1, color="#D55E00", ls="--", label="pass limit")
    axes[1, 0].set(yscale="log", ylabel="error / pass limit", title="All refinable-family synthetic recoveries")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].legend()
    axes[1, 0].bar_label(bars, fmt="%.2g", fontsize=7)

    nist = evidence["nist"]
    candidate = nist.candidates[0]
    stride = max(1, len(nist.dataset.coordinate) // 2500)
    axes[1, 1].plot(nist.dataset.coordinate[::stride], nist.dataset.intensity[::stride], color="black", lw=0.45, label="NIST observed")
    axes[1, 1].plot(nist.dataset.coordinate[::stride], candidate.calculated[::stride], color="#0072B2", lw=0.45, label="refined")
    axes[1, 1].set(xlabel="2theta (degrees)", ylabel="counts", title=f"NIST SRM 660c: Rwp={candidate.r_wp:.4f}")
    axes[1, 1].legend(fontsize=8)

    difficult = evidence["difficult_values"]
    axes[2, 0].bar(list(difficult), list(difficult.values()), color="#56B4E9")
    axes[2, 0].set(yscale="log", ylabel="declared diagnostic ratio/norm", title="Difficult-case evidence (different metrics)")
    axes[2, 0].tick_params(axis="x", rotation=35)

    status_names = ("pass", "warn", "pending_review", "unsupported", "fail")
    colors = ("#009E73", "#E69F00", "#56B4E9", "#D55E00", "#CC0000")
    counts = [matrix.status_counts[name] for name in status_names]
    axes[2, 1].bar(status_names, counts, color=colors)
    axes[2, 1].set(ylabel="validation cases", title=f"Matrix status: {matrix.overall_status.replace('_', ' ')}")
    axes[2, 1].tick_params(axis="x", rotation=25)
    figure.suptitle("Reference validation: passes remain separate from limitations", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_html(matrix, figure, report):
    encoded = base64.b64encode(Path(figure).read_bytes()).decode("ascii")
    rows = []
    for case in matrix.cases:
        metric_text = "; ".join(
            f"{metric.name}={metric.value:.6g} {metric.unit} [{metric.status}]"
            if metric.value is not None else f"{metric.name} [{metric.status}]"
            for metric in case.metrics
        ) or "capability/review gate"
        rows.append(
            f"<tr class='{case.status}'><td>{html.escape(case.status)}</td>"
            f"<td>{html.escape(case.category)}</td><td>{html.escape(case.identifier)}</td>"
            f"<td>{html.escape(metric_text)}</td></tr>"
        )
    sources = "".join(
        f"<tr><td>{html.escape(source.identifier)}</td><td>{html.escape(source.material)}</td>"
        f"<td>{html.escape(source.instrument)}</td><td>{source.wavelength_angstrom:g} A</td>"
        f"<td><a href='{html.escape(source.source_url)}'>source</a></td>"
        f"<td><code>{source.sha256}</code></td></tr>" for source in matrix.sources
    )
    checklist = "".join(f"<li>{html.escape(item)}</li>" for item in matrix.expert_review_checklist)
    content = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>BraggCalculator reference validation</title><style>
body{{font-family:system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem}}
img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:.86rem}}
th,td{{padding:.4rem;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}}
code{{font-size:.75rem}}.pass td:first-child{{color:#087f5b}}.warn td:first-child,
.pending_review td:first-child{{color:#a15c00}}.unsupported td:first-child,.fail td:first-child{{color:#b42318}}
</style></head><body><h1>Reference-validation matrix</h1>
<p><strong>Overall status: {html.escape(matrix.overall_status)}</strong>. A pass means only that
the declared numerical gate passed. Unsupported physics and pending human review remain visible
and cannot be averaged away.</p><img alt='six-panel reference-validation evidence' src='data:image/png;base64,{encoded}'>
<h2>Cases</h2><table><tr><th>Status</th><th>Category</th><th>Case</th><th>Evidence</th></tr>{''.join(rows)}</table>
<h2>Public inputs</h2><table><tr><th>ID</th><th>Material</th><th>Instrument</th><th>Wavelength</th><th>Origin</th><th>SHA-256</th></tr>{sources}</table>
<h2>External expert review: pending</h2><ol>{checklist}</ol>
<p>The direct GSAS-II final-profile/covariance reproduction is also pending. Pymatgen currently
serves as the independent line-pattern oracle; NIST SRM 660c supplies the certified lattice gate.</p>
</body></html>"""
    Path(report).write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args(argv)
    matrix, evidence = build_validation_matrix(nist_report=Path(__file__).with_name("nist_lab6_report.html"))
    matrix.write_json(args.json)
    _plot(matrix, evidence, args.figure)
    _write_html(matrix, args.figure, args.report)
    print(f"wrote {args.figure}")
    print(f"wrote {args.report}")
    print(f"wrote {args.json}")
    print(f"overall status: {matrix.overall_status}; counts: {matrix.status_counts}")
    for case in matrix.cases:
        if case.status != "pass":
            print(f"{case.status}: {case.identifier}")


if __name__ == "__main__":
    main()
