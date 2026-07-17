"""Relationship-aware diagnostics for arbitrary periodic structural models."""

from __future__ import annotations

from fractions import Fraction
from math import pi, sqrt
from typing import Mapping, Sequence

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher

from .backends import NumpyBackend
from .core import BraggCalculator
from .diagnostics import compare_profile_counts, mismatch_disk
from .factors import neutron_b_coherent
from .io import to_pmg_structure
from .profiles import FWHM_TO_SIGMA, GaussianProfileQ
from .results import (
    CounterfactualAttribution,
    ExperimentRecommendation,
    PairDistributionComparison,
    PeakGroupDiagnostic,
    ReflectionMatch,
    StructuralDiagnosticsResult,
    StructuralRelationship,
    SuperstructureResult,
)


def classify_structural_relationship(
    model_a,
    model_b,
    *,
    lattice_rtol: float = 1e-6,
    lattice_atol: float = 1e-7,
    maximum_rational_denominator: int = 8,
) -> StructuralRelationship:
    """Classify a structural pair before selecting mathematically valid diagnostics."""
    structure_a = _prepared_structure(model_a)
    structure_b = _prepared_structure(model_b)
    lattice_a = np.asarray(structure_a.lattice.matrix, dtype=np.float64)
    lattice_b = np.asarray(structure_b.lattice.matrix, dtype=np.float64)
    volume_ratio = float(structure_b.volume / structure_a.volume)

    matcher = StructureMatcher(
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
        allow_subset=False,
        ltol=max(lattice_rtol, 1e-7),
        stol=1e-4,
        angle_tol=0.01,
    )
    lattice_mapping = _best_lattice_mapping(
        structure_a.lattice, structure_b.lattice, ltol=max(lattice_rtol, 1e-7), atol=0.01
    )
    if matcher.fit(structure_a, structure_b):
        matched_transformation = matcher.get_transformation(structure_a, structure_b)
        transform = np.asarray(matched_transformation[0], dtype=np.int64)
        direction = "b_to_a"
        return StructuralRelationship(
            regime="I",
            classification="equivalent",
            transformation=transform,
            transformation_direction=direction,
            volume_ratio=volume_ratio,
            complex_comparison_allowed=True,
            reason="StructureMatcher found the same periodic decorated structure.",
        )

    if lattice_mapping is not None and np.isclose(
        abs(np.linalg.det(lattice_mapping[2])), 1.0
    ):
        return StructuralRelationship(
            regime="I",
            classification="lattice_compatible",
            transformation=np.asarray(lattice_mapping[2], dtype=np.int64),
            transformation_direction="a_to_b",
            volume_ratio=volume_ratio,
            complex_comparison_allowed=True,
            reason="The direct lattice matrices coincide but the decorated structures differ.",
        )

    transform, direction = _integer_lattice_transform(
        lattice_a, lattice_b, lattice_rtol, lattice_atol
    )
    if transform is not None:
        determinant = abs(float(np.linalg.det(transform)))
        regime = "I" if np.isclose(determinant, 1.0) else "II"
        classification = "lattice_compatible" if regime == "I" else "commensurate"
        return StructuralRelationship(
            regime=regime,
            classification=classification,
            transformation=transform,
            transformation_direction=direction,
            volume_ratio=volume_ratio,
            complex_comparison_allowed=True,
            reason=(
                "An integer direct-cell transformation establishes a common reciprocal "
                "representation."
            ),
        )

    rational = _rational_transform(
        lattice_b @ np.linalg.inv(lattice_a),
        maximum_rational_denominator,
        lattice_atol,
    )
    if rational is not None:
        return StructuralRelationship(
            regime="II",
            classification="commensurate",
            transformation=rational,
            transformation_direction="common_rational",
            volume_ratio=volume_ratio,
            complex_comparison_allowed=True,
            reason=(
                "The cell transformation is rational within tolerance; reciprocal vectors "
                "can be matched, but there is no one-way integer parent/supercell mapping."
            ),
        )

    return StructuralRelationship(
        regime="III",
        classification="unrelated",
        transformation=None,
        transformation_direction=None,
        volume_ratio=volume_ratio,
        complex_comparison_allowed=False,
        reason=(
            "No bounded integer or rational cell transformation was found; direct hkl-phase "
            "comparison is disabled."
        ),
    )


def radial_pair_distribution(
    model,
    *,
    radiation: str = "xray",
    r_max: float = 10.0,
    step: float = 0.02,
    broadening: float = 0.08,
    neutron_scattering_lengths: Mapping[str | int, float | str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a periodic scattering-weighted radial Patterson/PDF signal."""
    if radiation not in {"xray", "neutron"}:
        raise ValueError("radiation must be 'xray' or 'neutron'")
    if r_max <= 0 or step <= 0 or broadening <= 0:
        raise ValueError("r_max, step, and broadening must be positive")
    structure = _prepared_structure(model)
    radius = np.arange(0.0, r_max + 0.5 * step, step)
    weights = _site_scattering_weights(
        structure, radiation, neutron_scattering_lengths=neutron_scattering_lengths
    )
    distribution = np.zeros_like(radius)
    sigma = broadening * FWHM_TO_SIGMA
    normalization = 1.0 / (sigma * sqrt(2.0 * pi))
    for center_index, site in enumerate(structure):
        for neighbor in structure.get_neighbors(site, r_max):
            distance = float(neighbor.nn_distance)
            if distance <= 1e-12:
                continue
            pair_weight = 0.5 * weights[center_index] * weights[int(neighbor.index)]
            distribution += (
                pair_weight
                * normalization
                * np.exp(-0.5 * ((radius - distance) / sigma) ** 2)
                / max(distance**2, 1e-12)
            )
    norm = np.linalg.norm(distribution)
    if norm:
        distribution /= norm
    return radius, distribution


def compare_pair_distributions(
    model_a,
    model_b,
    *,
    radiation: str = "xray",
    r_max: float = 10.0,
    step: float = 0.02,
    broadening: float = 0.08,
    neutron_scattering_lengths: Mapping[str | int, float | str] | None = None,
) -> PairDistributionComparison:
    """Compare periodic radial Patterson/PDF signals on a shared grid."""
    radius, distribution_a = radial_pair_distribution(
        model_a,
        radiation=radiation,
        r_max=r_max,
        step=step,
        broadening=broadening,
        neutron_scattering_lengths=neutron_scattering_lengths,
    )
    other_radius, distribution_b = radial_pair_distribution(
        model_b,
        radiation=radiation,
        r_max=r_max,
        step=step,
        broadening=broadening,
        neutron_scattering_lengths=neutron_scattering_lengths,
    )
    if not np.array_equal(radius, other_radius):  # pragma: no cover - shared declaration
        raise RuntimeError("pair-distribution grids differ")
    similarity = _cosine_similarity(distribution_a, distribution_b)
    return PairDistributionComparison(
        radius=radius,
        distribution_a=distribution_a,
        distribution_b=distribution_b,
        similarity=similarity,
        radiation=radiation,
        broadening=float(broadening),
        r_max=float(r_max),
    )


def peak_group_attribution(
    calculator,
    *,
    fwhm_q: float,
    site_groups: Mapping[str, Sequence[int]] | None = None,
    overlap_factor: float = 1.0,
    maximum_groups: int | None = 20,
) -> tuple[PeakGroupDiagnostic, ...]:
    """Group reciprocal points by resolution and estimate non-additive site effects."""
    calculator._ensure_loaded()
    if fwhm_q <= 0 or overlap_factor <= 0:
        raise ValueError("fwhm_q and overlap_factor must be positive")
    groups = _resolved_site_groups(calculator, site_groups)
    table = calculator.reflection_table(domain="q")
    q = _as_numpy(table.q)
    intensity = np.maximum(_as_numpy(table.intensity), 0.0)
    order = np.argsort(q)
    q_sorted = q[order]
    boundaries = np.r_[
        0,
        np.flatnonzero(np.diff(q_sorted) > overlap_factor * fwhm_q) + 1,
        len(q_sorted),
    ]
    counterfactual_intensity = {}
    for name, sites in groups.items():
        parameters = calculator.tensor_parameters()
        occupancies = _as_numpy(parameters["occupancies"]).copy()
        contribution_sites = np.asarray(calculator._symm["site_indices"], dtype=np.int64)
        occupancies[np.isin(contribution_sites, sites)] = 0.0
        parameters["occupancies"] = occupancies
        counterfactual_intensity[name] = np.maximum(
            _as_numpy(calculator.reflection_table(domain="q", parameters=parameters).intensity),
            0.0,
        )

    diagnostics = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        indices = order[start:stop]
        values = intensity[indices]
        total = float(values.sum())
        if total <= 0:
            continue
        probabilities = values / total
        positive = probabilities > 0
        effective = float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))
        effects = {
            name: float(np.sum(np.abs(values - removed[indices])) / total)
            for name, removed in counterfactual_intensity.items()
        }
        diagnostics.append(
            PeakGroupDiagnostic(
                q_center=float(np.sum(q[indices] * values) / total),
                q_min=float(q[indices].min()),
                q_max=float(q[indices].max()),
                integrated_intensity=total,
                effective_reflections=effective,
                hkl=table.hkl[indices].copy(),
                reflection_intensity=values.copy(),
                site_effects=effects,
            )
        )
    diagnostics.sort(key=lambda item: item.integrated_intensity, reverse=True)
    if maximum_groups is not None:
        diagnostics = diagnostics[:maximum_groups]
    return tuple(diagnostics)


def counterfactual_site_substitutions(
    calculator_a,
    calculator_b,
    groups: Mapping[str, Sequence[int]],
    *,
    domain: str = "q",
) -> tuple[CounterfactualAttribution, ...]:
    """Replace declared prepared sites in A by B and compare profile directions."""
    calculator_a._ensure_loaded()
    calculator_b._ensure_loaded()
    if not np.allclose(
        calculator_a._symm["lattice"], calculator_b._symm["lattice"], rtol=1e-7, atol=1e-8
    ):
        raise ValueError("site substitution requires the same prepared lattice")
    if not np.array_equal(calculator_a._symm["Z"], calculator_b._symm["Z"]) or not np.array_equal(
        calculator_a._symm["site_indices"], calculator_b._symm["site_indices"]
    ):
        raise ValueError("site substitution requires contribution-wise species/site mapping")
    coordinate_a, profile_a = calculator_a.pattern(domain=domain)
    coordinate_b, profile_b = calculator_b.pattern(domain=domain)
    coordinate = _as_numpy(coordinate_a)
    if not np.allclose(coordinate, _as_numpy(coordinate_b), atol=1e-12, rtol=0):
        raise ValueError("calculator profile grids must coincide")
    base = _as_numpy(profile_a)
    target_change = _as_numpy(profile_b) - base
    target_norm = float(np.linalg.norm(target_change))
    parameters_a = {name: _as_numpy(value).copy() for name, value in calculator_a.tensor_parameters().items()}
    parameters_b = {name: _as_numpy(value).copy() for name, value in calculator_b.tensor_parameters().items()}
    contribution_sites = np.asarray(calculator_a._symm["site_indices"], dtype=np.int64)
    results = []
    for name, sites in groups.items():
        selected_sites = tuple(sorted({int(item) for item in sites}))
        mask = np.isin(contribution_sites, selected_sites)
        hybrid = {key: value.copy() for key, value in parameters_a.items()}
        for key in ("frac_coords", "occupancies", "b_iso", "u_cart"):
            if key in hybrid and key in parameters_b:
                hybrid[key][mask] = parameters_b[key][mask]
        _, hybrid_profile = calculator_a.pattern(domain=domain, parameters=hybrid)
        change = _as_numpy(hybrid_profile) - base
        change_norm = float(np.linalg.norm(change))
        denominator = change_norm**2 * target_norm**2
        alignment = (
            float(np.dot(change, target_change) ** 2 / denominator) if denominator > 0 else 0.0
        )
        results.append(
            CounterfactualAttribution(
                name=str(name),
                site_indices=selected_sites,
                effect_norm=change_norm / max(target_norm, 1e-30),
                alignment_fraction=float(np.clip(alignment, 0.0, 1.0)),
                largest_effect_coordinate=float(coordinate[int(np.argmax(np.abs(change)))]),
                profile_change=change,
            )
        )
    return tuple(results)


def identify_superstructure_reflections(
    calculator_a,
    calculator_b,
    relationship: StructuralRelationship | None = None,
) -> SuperstructureResult | None:
    """Identify calculated supercell reflections absent from the parent reciprocal lattice."""
    relation = relationship or classify_structural_relationship(calculator_a, calculator_b)
    if relation.classification != "commensurate" or relation.transformation is None:
        return None
    if relation.transformation_direction not in {"a_to_b", "b_to_a"}:
        return None
    if relation.transformation_direction == "a_to_b":
        parent_name, supercell_name = "A", "B"
        supercell_calculator = calculator_b
    else:
        parent_name, supercell_name = "B", "A"
        supercell_calculator = calculator_a
    transformation = np.asarray(relation.transformation, dtype=np.float64)
    if not np.allclose(transformation, np.rint(transformation), atol=1e-8):
        return None
    transformation = np.rint(transformation).astype(np.int64)
    table = supercell_calculator.reflection_table(domain="q")
    parent_indices = table.hkl @ np.linalg.inv(transformation).T
    fundamental = np.all(np.isclose(parent_indices, np.rint(parent_indices), atol=1e-8), axis=1)
    intensity = np.maximum(_as_numpy(table.intensity), 0.0)
    visible = intensity > max(float(intensity.max()) * 1e-10, 0.0)
    selected = ~fundamental & visible
    total = float(intensity.sum())
    return SuperstructureResult(
        parent=parent_name,
        supercell=supercell_name,
        transformation=transformation,
        hkl=table.hkl[selected].copy(),
        q=_as_numpy(table.q)[selected].copy(),
        intensity=intensity[selected].copy(),
        intensity_fraction=float(intensity[selected].sum() / total) if total > 0 else 0.0,
    )


def diagnose_structures(
    model_a,
    model_b,
    *,
    radiation: str = "xray",
    wavelength: float = 1.5406,
    q_range: tuple[float, float] = (0.3, 8.0),
    q_step: float = 0.01,
    profile_fwhm_q: float = 0.08,
    count_scale: float = 100.0,
    background_density: float = 1.0,
    pair_r_max: float = 10.0,
    pair_broadening: float = 0.08,
    site_groups: Mapping[str, Sequence[int]] | None = None,
    counterfactual_groups: Mapping[str, Sequence[int]] | None = None,
) -> StructuralDiagnosticsResult:
    """Run the strongest valid diagnostics for an arbitrary structural pair."""
    relationship = classify_structural_relationship(model_a, model_b)
    structure_a = _prepared_structure(model_a)
    structure_b = _prepared_structure(model_b)
    calculator_a = _diagnostic_calculator(
        structure_a, radiation, wavelength, q_range, q_step, profile_fwhm_q
    )
    calculator_b = _diagnostic_calculator(
        structure_b, radiation, wavelength, q_range, q_step, profile_fwhm_q
    )
    discrimination = compare_profile_counts(
        calculator_a,
        calculator_b,
        domain="q",
        count_scale=count_scale,
        background_density=background_density,
    )
    pair = compare_pair_distributions(
        structure_a,
        structure_b,
        radiation=radiation,
        r_max=pair_r_max,
        step=min(0.02, pair_broadening / 3.0),
        broadening=pair_broadening,
    )

    table_a = calculator_a.reflection_table(domain="q")
    table_b = calculator_b.reflection_table(domain="q")
    reciprocal_match = None
    mismatch = None
    intensity_similarity = None
    complex_similarity = None
    if relationship.complex_comparison_allowed:
        try:
            reciprocal_match = _match_hkl_from_relationship(
                table_a.hkl, table_b.hkl, relationship
            )
            if reciprocal_match is None:
                reciprocal_match = _match_reciprocal_vectors(
                    table_a.hkl,
                    calculator_a._symm["lattice"],
                    table_b.hkl,
                    calculator_b._symm["lattice"],
                )
        except ValueError:
            reciprocal_match = None
        if reciprocal_match is not None:
            normalization_a = _structure_factor_normalization(calculator_a)
            normalization_b = _structure_factor_normalization(calculator_b)
            factors_a = (
                _as_numpy(table_a.structure_factor)[reciprocal_match.indices_a]
                / normalization_a
            )
            factors_b = (
                _as_numpy(table_b.structure_factor)[reciprocal_match.indices_b]
                / normalization_b
            )
            mismatch_result = mismatch_disk(
                reciprocal_match.hkl,
                factors_a,
                factors_b,
                optimize_origin=(relationship.regime == "I"),
            )
            mismatch = type(mismatch_result)(
                match=reciprocal_match,
                alignment=mismatch_result.alignment,
                structure_factor_a=mismatch_result.structure_factor_a,
                amplitude_a=mismatch_result.amplitude_a,
                amplitude_b=mismatch_result.amplitude_b,
                phase_difference=mismatch_result.phase_difference,
                phase_defined=mismatch_result.phase_defined,
                x=mismatch_result.x,
                y=mismatch_result.y,
                radius=mismatch_result.radius,
                weights=mismatch_result.weights,
                d_sf=mismatch_result.d_sf,
                d_amplitude=mismatch_result.d_amplitude,
                d_phase=mismatch_result.d_phase,
                identity_error=mismatch_result.identity_error,
                epsilon=mismatch_result.epsilon,
                phase_threshold=mismatch_result.phase_threshold,
            )
            complex_similarity = float(np.clip(1.0 - mismatch.d_sf, 0.0, 1.0))
            intensity_similarity = _cosine_similarity(
                np.abs(factors_a) ** 2, np.abs(factors_b) ** 2
            )

    grid_a, profile_a = calculator_a.pattern(domain="q")
    grid_b, profile_b = calculator_b.pattern(domain="q")
    grid = _as_numpy(grid_a)
    if not np.allclose(grid, _as_numpy(grid_b), rtol=0, atol=1e-12):  # pragma: no cover
        raise RuntimeError("diagnostic profile grids differ")
    ideal_width = max(q_step * 1.5, min(profile_fwhm_q / 6.0, 0.015))
    ideal_a = _render_sticks(grid, _as_numpy(table_a.q), _as_numpy(table_a.intensity), ideal_width)
    ideal_b = _render_sticks(grid, _as_numpy(table_b.q), _as_numpy(table_b.intensity), ideal_width)
    similarities = {
        "complex": complex_similarity,
        "intensity": intensity_similarity,
        "ideal_powder": _cosine_similarity(ideal_a, ideal_b),
        "profile": _cosine_similarity(_as_numpy(profile_a), _as_numpy(profile_b)),
        "radial_pair": pair.similarity,
    }
    dominant, explanation = _classify_information_loss(similarities, relationship)
    peak_groups_a = peak_group_attribution(
        calculator_a, fwhm_q=profile_fwhm_q, site_groups=site_groups
    )
    peak_groups_b = peak_group_attribution(
        calculator_b, fwhm_q=profile_fwhm_q, site_groups=site_groups
    )
    counterfactuals = ()
    if counterfactual_groups and relationship.regime == "I":
        counterfactuals = counterfactual_site_substitutions(
            calculator_a, calculator_b, counterfactual_groups
        )
    superstructure = identify_superstructure_reflections(
        calculator_a, calculator_b, relationship
    )
    return StructuralDiagnosticsResult(
        relationship=relationship,
        similarities=similarities,
        dominant_information_loss=dominant,
        explanation=explanation,
        mismatch=mismatch,
        profile_discrimination=discrimination,
        pair_distribution=pair,
        superstructure=superstructure,
        peak_groups_a=peak_groups_a,
        peak_groups_b=peak_groups_b,
        counterfactuals=counterfactuals,
    )


def suggest_measurements(
    model_a,
    model_b,
    configurations: Sequence[Mapping[str, object]],
) -> tuple[ExperimentRecommendation, ...]:
    """Rank declared experiments by expected measured-count discrimination."""
    if not configurations:
        raise ValueError("configurations must be non-empty")
    structure_a = _prepared_structure(model_a)
    structure_b = _prepared_structure(model_b)
    recommendations = []
    for index, declaration in enumerate(configurations):
        name = str(declaration.get("name", f"experiment_{index + 1}"))
        radiation = str(declaration.get("radiation", "xray"))
        wavelength = float(declaration.get("wavelength", 1.5406))
        q_range = tuple(float(item) for item in declaration.get("q_range", (0.3, 8.0)))
        q_step = float(declaration.get("q_step", 0.01))
        fwhm_q = float(declaration.get("fwhm_q", 0.08))
        count_scale = float(declaration.get("count_scale", 100.0))
        background_density = float(declaration.get("background_density", 1.0))
        calculator_a = _diagnostic_calculator(
            structure_a, radiation, wavelength, q_range, q_step, fwhm_q
        )
        calculator_b = _diagnostic_calculator(
            structure_b, radiation, wavelength, q_range, q_step, fwhm_q
        )
        result = compare_profile_counts(
            calculator_a,
            calculator_b,
            domain="q",
            count_scale=count_scale,
            background_density=background_density,
        )
        most_informative = (
            float(result.coordinate[int(np.argmax(result.pointwise_discrimination))])
            if result.pointwise_discrimination is not None
            else float("nan")
        )
        recommendations.append(
            ExperimentRecommendation(
                name=name,
                radiation=radiation,
                wavelength=wavelength,
                q_range=(q_range[0], q_range[1]),
                fwhm_q=fwhm_q,
                total_discrimination=result.total_discrimination,
                most_informative_q=most_informative,
                assumptions={
                    "q_step": q_step,
                    "count_scale": count_scale,
                    "background_density": background_density,
                    "variance": "symmetric mean-count Poisson approximation",
                },
            )
        )
    recommendations.sort(key=lambda item: item.total_discrimination, reverse=True)
    return tuple(recommendations)


def _classify_information_loss(similarities, relationship):
    if relationship.regime == "III":
        return (
            "unrelated_lattices",
            "No reciprocal phase correspondence exists; comparison begins at powder and "
            "radial-pair levels.",
        )
    if similarities["complex"] is None or similarities["intensity"] is None:
        return (
            "no_common_reflections",
            "The relationship permits a common reciprocal representation, but the declared "
            "Q range contains no matched reciprocal vectors.",
        )
    transitions = {
        "phase_loss": (similarities["intensity"] or 0.0) - (similarities["complex"] or 0.0),
        "powder_averaging": (similarities["ideal_powder"] or 0.0)
        - (similarities["intensity"] or 0.0),
        "peak_overlap": (similarities["profile"] or 0.0)
        - (similarities["ideal_powder"] or 0.0),
    }
    dominant = max(transitions, key=transitions.get)
    if transitions[dominant] < 0.03:
        if min(value for value in similarities.values() if value is not None) > 0.95:
            return "intrinsically_similar", "The models remain similar at every evaluated level."
        return (
            "differences_remain_visible",
            "No information-losing transition increases similarity by more than 0.03.",
        )
    explanations = {
        "phase_loss": "Discarding calculated phases produces the largest similarity increase.",
        "powder_averaging": "Collapsing reciprocal directions onto Q produces the largest similarity increase.",
        "peak_overlap": "The declared profile width merges the strongest remaining differences.",
    }
    return dominant, explanations[dominant]


def _prepared_structure(model):
    if isinstance(model, BraggCalculator):
        model._ensure_loaded()
        return model._symm["structure"].copy()
    return to_pmg_structure(model).copy()


def _integer_lattice_transform(lattice_a, lattice_b, rtol, atol):
    forward = lattice_b @ np.linalg.inv(lattice_a)
    rounded = np.rint(forward)
    if abs(np.linalg.det(rounded)) > 0.5 and np.allclose(forward, rounded, rtol=rtol, atol=atol):
        return rounded.astype(np.int64), "a_to_b"
    reverse = lattice_a @ np.linalg.inv(lattice_b)
    rounded = np.rint(reverse)
    if abs(np.linalg.det(rounded)) > 0.5 and np.allclose(reverse, rounded, rtol=rtol, atol=atol):
        return rounded.astype(np.int64), "b_to_a"
    return None, None


def _best_lattice_mapping(lattice_a, lattice_b, *, ltol, atol):
    mappings = tuple(lattice_a.find_all_mappings(lattice_b, ltol=ltol, atol=atol))
    if not mappings:
        return None
    identity = np.eye(3)
    return min(mappings, key=lambda item: np.linalg.norm(np.asarray(item[2]) - identity))


def _rational_transform(matrix, maximum_denominator, tolerance):
    rational = np.empty_like(matrix)
    for index, value in np.ndenumerate(matrix):
        fraction = Fraction(float(value)).limit_denominator(maximum_denominator)
        rational[index] = fraction.numerator / fraction.denominator
        if abs(rational[index] - value) > tolerance:
            return None
    if abs(np.linalg.det(rational)) < 1e-10:
        return None
    return rational


def _diagnostic_calculator(structure, radiation, wavelength, q_range, q_step, fwhm_q):
    return BraggCalculator(
        mode=radiation,
        wavelength=wavelength,
        q_range=q_range,
        q_step=q_step,
        profile_q=GaussianProfileQ(fwhm_q=fwhm_q),
        primitive=False,
    ).load(structure)


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _cosine_similarity(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.clip(np.dot(left, right) / denominator, 0.0, 1.0))


def _site_scattering_weights(structure, radiation, *, neutron_scattering_lengths=None):
    atomic_numbers = []
    occupancies = []
    for site in structure:
        atomic_numbers.append([species.Z for species in site.species])
        occupancies.append([float(value) for value in site.species.values()])
    if radiation == "xray":
        return np.asarray(
            [sum(z * occ for z, occ in zip(numbers, values)) for numbers, values in zip(atomic_numbers, occupancies)]
        )
    backend = NumpyBackend()
    result = []
    for numbers, values in zip(atomic_numbers, occupancies):
        lengths = neutron_b_coherent(
            np.asarray(numbers), backend, overrides=neutron_scattering_lengths
        )
        result.append(float(np.dot(lengths, values)))
    return np.asarray(result)


def _resolved_site_groups(calculator, groups):
    site_count = len(calculator._symm["structure"])
    if groups is None:
        result = {}
        for index, orbit in enumerate(calculator._symm["orbit_indices"]):
            sites = tuple(int(item) for item in orbit)
            symbols = sorted({calculator._symm["structure"][item].species_string for item in sites})
            result[f"orbit {index} ({'/'.join(symbols)})"] = sites
        return result
    result = {}
    for name, sites in groups.items():
        values = tuple(sorted({int(item) for item in sites}))
        if not values or min(values) < 0 or max(values) >= site_count:
            raise ValueError(f"site group {name!r} contains an invalid prepared-site index")
        result[str(name)] = values
    return result


def _render_sticks(grid, positions, intensities, fwhm):
    sigma = fwhm * FWHM_TO_SIGMA
    result = np.zeros_like(grid, dtype=np.float64)
    normalization = 1.0 / (sigma * sqrt(2.0 * pi))
    for start in range(0, len(positions), 256):
        stop = min(start + 256, len(positions))
        distance = grid[:, None] - positions[None, start:stop]
        result += normalization * np.sum(
            intensities[None, start:stop] * np.exp(-0.5 * (distance / sigma) ** 2), axis=1
        )
    return result


def _match_reciprocal_vectors(hkl_a, lattice_a, hkl_b, lattice_b, tolerance=1e-7):
    reciprocal_a = np.asarray(hkl_a) @ np.linalg.inv(np.asarray(lattice_a)).T
    reciprocal_b = np.asarray(hkl_b) @ np.linalg.inv(np.asarray(lattice_b)).T
    lookup = {}
    for index, vector in enumerate(reciprocal_b):
        lookup.setdefault(tuple(np.rint(vector / tolerance).astype(np.int64)), index)
    pairs = []
    for index, vector in enumerate(reciprocal_a):
        other = lookup.get(tuple(np.rint(vector / tolerance).astype(np.int64)))
        if other is not None and np.linalg.norm(vector - reciprocal_b[other]) <= 2 * tolerance:
            pairs.append((index, other))
    if not pairs:
        raise ValueError("commensurate cells have no matched reciprocal vectors in range")
    indices_a = np.asarray([item[0] for item in pairs], dtype=np.int64)
    indices_b = np.asarray([item[1] for item in pairs], dtype=np.int64)
    return ReflectionMatch(
        hkl=np.asarray(hkl_a, dtype=np.int64)[indices_a].copy(),
        indices_a=indices_a,
        indices_b=indices_b,
    )


def _match_hkl_from_relationship(hkl_a, hkl_b, relationship):
    transformation = relationship.transformation
    direction = relationship.transformation_direction
    if transformation is None or direction not in {"a_to_b", "b_to_a"}:
        return None
    matrix = np.asarray(transformation, dtype=np.float64)
    if not np.allclose(matrix, np.rint(matrix), atol=1e-8):
        return None
    matrix = np.rint(matrix).astype(np.int64)
    lookup_a = {tuple(int(value) for value in row): index for index, row in enumerate(hkl_a)}
    lookup_b = {tuple(int(value) for value in row): index for index, row in enumerate(hkl_b)}
    pairs = []
    if direction == "a_to_b":
        for index_a, row in enumerate(np.asarray(hkl_a, dtype=np.int64)):
            target = tuple(int(value) for value in row @ matrix.T)
            if target in lookup_b:
                pairs.append((index_a, lookup_b[target]))
    else:
        for index_b, row in enumerate(np.asarray(hkl_b, dtype=np.int64)):
            target = tuple(int(value) for value in row @ matrix.T)
            if target in lookup_a:
                pairs.append((lookup_a[target], index_b))
    if not pairs:
        return None
    pairs.sort()
    indices_a = np.asarray([item[0] for item in pairs], dtype=np.int64)
    indices_b = np.asarray([item[1] for item in pairs], dtype=np.int64)
    return ReflectionMatch(
        hkl=np.asarray(hkl_a, dtype=np.int64)[indices_a].copy(),
        indices_a=indices_a,
        indices_b=indices_b,
    )


def _structure_factor_normalization(calculator):
    if calculator.mode == "xray":
        weights = np.asarray(calculator._symm["Z"], dtype=np.float64)
    else:
        weights = neutron_b_coherent(
            calculator._symm["Z"],
            NumpyBackend(),
            overrides=calculator.neutron_scattering_lengths,
        )
    return max(float(np.sum(np.abs(weights) * calculator._symm["occ"])), 1e-30)
