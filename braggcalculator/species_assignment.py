"""Discrete species assignment across asymmetric-unit sites."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .core import BraggCalculator
from .dataset import DiffractionDataset
from .session import (
    CandidateRefinementResult,
    RefinementPolicy,
    RefinementSession,
    refined_structure_from_candidate,
)


@dataclass(frozen=True)
class SpeciesAssignmentConfig:
    """Rules for generating, screening, and refining site assignments."""

    search: Literal["auto", "pairwise", "complete", "bounded", "random"] = "auto"
    fixed_sites: tuple[int | str, ...] = ()
    site_groups: Mapping[str, Sequence[int]] = field(default_factory=dict)
    allowed_species: Mapping[int | str, Sequence[str]] = field(default_factory=dict)
    max_candidates: int = 256
    continuous_top_k: int = 3
    screening_background_degree: int = 1
    ambiguity_tolerance: float = 0.002
    composition_preserving: bool = True
    mixed_occupancy_policy: Literal["fixed", "reject"] = "fixed"
    displacement_policy: Literal["site", "species"] = "site"
    oxidation_states: Mapping[str, float] | None = None
    target_charge: float = 0.0
    charge_tolerance: float = 1e-6
    symprec: float = 1e-3
    angle_tolerance: float = 5.0
    seed: int | None = None

    def __post_init__(self):
        if self.search not in {"auto", "pairwise", "complete", "bounded", "random"}:
            raise ValueError("unknown species-assignment search mode")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.continuous_top_k < 1:
            raise ValueError("continuous_top_k must be positive")
        if self.screening_background_degree < 0:
            raise ValueError("screening_background_degree must be non-negative")
        if self.ambiguity_tolerance < 0 or not np.isfinite(self.ambiguity_tolerance):
            raise ValueError("ambiguity_tolerance must be finite and non-negative")
        if self.mixed_occupancy_policy not in {"fixed", "reject"}:
            raise ValueError("mixed_occupancy_policy must be 'fixed' or 'reject'")
        if self.displacement_policy not in {"site", "species"}:
            raise ValueError("displacement_policy must be 'site' or 'species'")
        if self.symprec <= 0 or self.angle_tolerance <= 0:
            raise ValueError("symmetry tolerances must be positive")
        if self.charge_tolerance < 0 or not np.isfinite(self.charge_tolerance):
            raise ValueError("charge_tolerance must be finite and non-negative")
        if self.seed is not None and (not isinstance(self.seed, int) or self.seed < 0):
            raise ValueError("seed must be a non-negative integer or None")


@dataclass(frozen=True)
class AsymmetricUnitSite:
    """One independent crystallographic site and its symmetry-expanded indices."""

    site_index: int
    representative_index: int
    equivalent_indices: tuple[int, ...]
    label: str
    wyckoff_symbol: str
    multiplicity: int
    original_species: tuple[tuple[str, float], ...]
    mixed_occupancy: bool


@dataclass(frozen=True)
class SiteAssignment:
    """Original and proposed species for one independent site."""

    site_index: int
    representative_index: int
    equivalent_indices: tuple[int, ...]
    label: str
    wyckoff_symbol: str
    multiplicity: int
    original_species: tuple[tuple[str, float], ...]
    proposed_species: str


@dataclass(frozen=True)
class SpeciesAssignmentCandidate:
    """One screened assignment and its optional continuous refinement."""

    assignment_id: str
    sites: tuple[SiteAssignment, ...]
    structure: Structure
    screening_score: float
    continuous_score: float | None
    convergence: Mapping[str, Any] | None
    continuous_result: CandidateRefinementResult | None
    refined_structure: Structure | None
    refined_cif: str | None
    indistinguishable: bool = False


@dataclass(frozen=True)
class SpeciesAssignmentResult:
    """Ranked species assignments with search and ambiguity information."""

    sites: tuple[AsymmetricUnitSite, ...]
    candidates: tuple[SpeciesAssignmentCandidate, ...]
    search_mode: str
    generated_count: int
    evaluated_count: int
    deduplicated_count: int
    truncated: bool
    ambiguity_tolerance: float
    indistinguishable_assignments: tuple[str, ...]
    warnings: tuple[str, ...]
    target_composition: Mapping[str, float]


@dataclass(frozen=True)
class SpeciesAssignmentEnumeration:
    """Generated assignments and bounded-search accounting."""

    assignments: tuple[tuple[str, ...], ...]
    mode: str
    generated_count: int
    deduplicated_count: int
    truncated: bool


def asymmetric_unit_sites(
    structure: Structure,
    *,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
) -> tuple[AsymmetricUnitSite, ...]:
    """Find independent sites and their full-structure symmetry orbits."""
    symmetrized = SpacegroupAnalyzer(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    ).get_symmetrized_structure()
    sites = []
    for site_index, (indices, wyckoff) in enumerate(
        zip(symmetrized.equivalent_indices, symmetrized.wyckoff_symbols)
    ):
        equivalent = tuple(sorted(int(index) for index in indices))
        representative = equivalent[0]
        site = structure[representative]
        species = tuple(
            sorted(
                (element.symbol, float(amount))
                for element, amount in site.species.items()
            )
        )
        sites.append(
            AsymmetricUnitSite(
                site_index=site_index,
                representative_index=representative,
                equivalent_indices=equivalent,
                label=site.label,
                wyckoff_symbol=str(wyckoff),
                multiplicity=len(equivalent),
                original_species=species,
                mixed_occupancy=len(species) != 1 or not np.isclose(species[0][1], 1.0),
            )
        )
    return tuple(sites)


def enumerate_species_assignments(
    structure: Structure,
    config: SpeciesAssignmentConfig | None = None,
) -> tuple[tuple[AsymmetricUnitSite, ...], SpeciesAssignmentEnumeration]:
    """Generate valid composition-aware assignments in deterministic order."""
    selected = SpeciesAssignmentConfig() if config is None else config
    sites = asymmetric_unit_sites(
        structure,
        symprec=selected.symprec,
        angle_tolerance=selected.angle_tolerance,
    )
    fixed = _resolved_site_set(selected.fixed_sites, selected.site_groups, len(sites))
    choices = _species_choices(sites, selected, fixed)
    if selected.mixed_occupancy_policy == "reject" and any(
        site.mixed_occupancy for site in sites
    ):
        raise ValueError("mixed occupancies are present and the configured policy is 'reject'")
    if selected.oxidation_states is not None:
        missing = sorted(
            {
                symbol
                for allowed in choices
                for symbol in allowed
                if symbol not in selected.oxidation_states
            }
        )
        if missing:
            raise ValueError(f"oxidation_states is missing values for {missing}")

    original = tuple(_ordered_species(site) for site in sites)
    estimated_product = int(np.prod([len(item) for item in choices], dtype=object))
    mode = selected.search
    if mode == "auto":
        mode = "complete" if estimated_product <= selected.max_candidates else "bounded"
    if mode == "pairwise":
        source = _pairwise_assignments(original, choices, fixed)
    elif mode == "random":
        source = _random_assignments(original, choices, sites, fixed, selected)
    else:
        source = product(*choices)

    accepted = []
    seen = set()
    generated = 0
    duplicates = 0
    truncated = False
    for assignment in source:
        generated += 1
        key = tuple(assignment)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if not _assignment_is_valid(structure, sites, key, selected):
            continue
        if len(accepted) >= selected.max_candidates:
            truncated = True
            break
        accepted.append(key)
    if mode == "complete" and estimated_product > generated:
        truncated = True
    return sites, SpeciesAssignmentEnumeration(
        assignments=tuple(accepted),
        mode=mode,
        generated_count=generated,
        deduplicated_count=duplicates,
        truncated=truncated,
    )


def apply_species_assignment(
    structure: Structure,
    assignment: Sequence[str],
    config: SpeciesAssignmentConfig | None = None,
) -> Structure:
    """Build a structure from one asymmetric-unit species assignment."""
    selected = SpeciesAssignmentConfig() if config is None else config
    sites = asymmetric_unit_sites(
        structure,
        symprec=selected.symprec,
        angle_tolerance=selected.angle_tolerance,
    )
    normalized = tuple(Element(str(value)).symbol for value in assignment)
    if len(normalized) != len(sites):
        raise ValueError(
            f"assignment has {len(normalized)} entries for {len(sites)} asymmetric-unit sites"
        )
    if not _assignment_is_valid(structure, sites, normalized, selected):
        raise ValueError("assignment violates composition, mixed-occupancy, or charge rules")
    return _structure_with_assignment(
        structure,
        sites,
        normalized,
        displacement_policy=selected.displacement_policy,
    )


def refine_species_assignments(
    dataset: DiffractionDataset,
    structure: Structure,
    *,
    config: SpeciesAssignmentConfig | None = None,
    policy: RefinementPolicy | None = None,
    device: str = "cpu",
) -> SpeciesAssignmentResult:
    """Screen assignments, refine the top candidates, and report ambiguity."""
    selected = SpeciesAssignmentConfig() if config is None else config
    selected_policy = RefinementPolicy.quick() if policy is None else policy
    sites, enumeration = enumerate_species_assignments(structure, selected)
    screened = []
    for assignment_index, assignment in enumerate(enumeration.assignments):
        candidate_structure = _structure_with_assignment(
            structure,
            sites,
            assignment,
            displacement_policy=selected.displacement_policy,
        )
        score = _screening_score(
            dataset,
            candidate_structure,
            background_degree=selected.screening_background_degree,
        )
        screened.append((score, assignment_index, assignment, candidate_structure))
    screened.sort(key=lambda item: (item[0], item[2]))
    if not screened:
        return SpeciesAssignmentResult(
            sites=sites,
            candidates=(),
            search_mode=enumeration.mode,
            generated_count=enumeration.generated_count,
            evaluated_count=0,
            deduplicated_count=enumeration.deduplicated_count,
            truncated=enumeration.truncated,
            ambiguity_tolerance=selected.ambiguity_tolerance,
            indistinguishable_assignments=(),
            warnings=("No assignment satisfied the composition and site constraints.",),
            target_composition=structure.composition.get_el_amt_dict(),
        )

    refinement_count = min(selected.continuous_top_k, len(screened))
    refinement_inputs = screened[:refinement_count]
    names = tuple(f"assignment-{index + 1:04d}" for index in range(refinement_count))
    session_result = RefinementSession(
        dataset,
        [item[3] for item in refinement_inputs],
        names=names,
        device=device,
    ).run(selected_policy)
    refined_by_name = {candidate.name: candidate for candidate in session_result.candidates}

    candidates = []
    for ranked_index, (screen_score, _, assignment, candidate_structure) in enumerate(screened):
        continuous = (
            refined_by_name[names[ranked_index]]
            if ranked_index < refinement_count
            else None
        )
        refined_structure = (
            refined_structure_from_candidate(continuous, dataset)
            if continuous is not None
            else None
        )
        assignment_id = _assignment_id(assignment)
        candidates.append(
            SpeciesAssignmentCandidate(
                assignment_id=assignment_id,
                sites=_site_assignment_records(sites, assignment),
                structure=candidate_structure,
                screening_score=screen_score,
                continuous_score=(
                    _continuous_score(continuous, selected_policy)
                    if continuous is not None
                    else None
                ),
                convergence=continuous.convergence if continuous is not None else None,
                continuous_result=continuous,
                refined_structure=refined_structure,
                refined_cif=(
                    str(CifWriter(refined_structure, symprec=None))
                    if refined_structure is not None
                    else None
                ),
            )
        )
    candidates.sort(
        key=lambda item: (
            item.continuous_score is None,
            item.continuous_score
            if item.continuous_score is not None
            else item.screening_score,
            item.assignment_id,
        )
    )
    refined_scores = [
        item.continuous_score
        for item in candidates
        if item.continuous_score is not None
    ]
    best_score = min(refined_scores)
    ambiguous_ids = tuple(
        item.assignment_id
        for item in candidates
        if item.continuous_score is not None
        and item.continuous_score <= best_score + selected.ambiguity_tolerance
    )
    ambiguous_set = set(ambiguous_ids)
    candidates = [
        SpeciesAssignmentCandidate(
            **{
                **item.__dict__,
                "indistinguishable": item.assignment_id in ambiguous_set,
            }
        )
        for item in candidates
    ]
    warnings = []
    if enumeration.truncated:
        warnings.append(
            f"Species search reached max_candidates={selected.max_candidates}; "
            "the ranked list covers the evaluated subset."
        )
    if len(ambiguous_ids) > 1:
        warnings.append(
            "Several species assignments fall within the configured score tolerance."
        )
    return SpeciesAssignmentResult(
        sites=sites,
        candidates=tuple(candidates),
        search_mode=enumeration.mode,
        generated_count=enumeration.generated_count,
        evaluated_count=len(screened),
        deduplicated_count=enumeration.deduplicated_count,
        truncated=enumeration.truncated,
        ambiguity_tolerance=selected.ambiguity_tolerance,
        indistinguishable_assignments=ambiguous_ids,
        warnings=tuple(warnings),
        target_composition=structure.composition.get_el_amt_dict(),
    )


def _resolved_site_set(values, groups, site_count):
    result = set()
    for value in values:
        if isinstance(value, str):
            if value not in groups:
                raise ValueError(f"unknown site group: {value}")
            result.update(int(index) for index in groups[value])
        else:
            result.add(int(value))
    invalid = sorted(index for index in result if index < 0 or index >= site_count)
    if invalid:
        raise ValueError(f"site indices are outside the asymmetric unit: {invalid}")
    return frozenset(result)


def _species_choices(sites, config, fixed):
    group_indices = {
        name: _resolved_site_set((name,), config.site_groups, len(sites))
        for name in config.site_groups
    }
    explicit = {}
    for key, values in config.allowed_species.items():
        indices = (
            group_indices[key]
            if isinstance(key, str)
            else _resolved_site_set((int(key),), config.site_groups, len(sites))
        )
        normalized = tuple(sorted({Element(str(value)).symbol for value in values}))
        if not normalized:
            raise ValueError(f"allowed_species for {key!r} is empty")
        for index in indices:
            if index in explicit and explicit[index] != normalized:
                raise ValueError(f"conflicting allowed_species rules for site {index}")
            explicit[index] = normalized
    movable_symbols = tuple(
        sorted(
            {
                _ordered_species(site)
                for site in sites
                if site.site_index not in fixed and not site.mixed_occupancy
            }
        )
    )
    choices = []
    for site in sites:
        original = _ordered_species(site)
        if site.site_index in fixed or site.mixed_occupancy:
            if site.site_index in explicit and original not in explicit[site.site_index]:
                raise ValueError(f"fixed site {site.site_index} excludes its original species")
            choices.append((original,))
        else:
            choices.append(explicit.get(site.site_index, movable_symbols))
    return tuple(choices)


def _ordered_species(site):
    if site.mixed_occupancy:
        return max(site.original_species, key=lambda item: (item[1], item[0]))[0]
    return site.original_species[0][0]


def _pairwise_assignments(original, choices, fixed):
    yield original
    for left in range(len(original)):
        for right in range(left + 1, len(original)):
            if left in fixed or right in fixed or original[left] == original[right]:
                continue
            swapped = list(original)
            swapped[left], swapped[right] = swapped[right], swapped[left]
            if swapped[left] in choices[left] and swapped[right] in choices[right]:
                yield tuple(swapped)


def _random_assignments(original, choices, sites, fixed, config):
    yield original
    rng = np.random.default_rng(config.seed)
    movable_by_multiplicity = {}
    for index, site in enumerate(sites):
        if index not in fixed and not site.mixed_occupancy:
            movable_by_multiplicity.setdefault(site.multiplicity, []).append(index)
    attempts = max(100, config.max_candidates * 50)
    for _ in range(attempts):
        assignment = list(original)
        for indices in movable_by_multiplicity.values():
            values = [original[index] for index in indices]
            rng.shuffle(values)
            for index, value in zip(indices, values):
                assignment[index] = value
        if all(value in choices[index] for index, value in enumerate(assignment)):
            yield tuple(assignment)


def _assignment_is_valid(structure, sites, assignment, config):
    for site, symbol in zip(sites, assignment):
        if site.mixed_occupancy and symbol != _ordered_species(site):
            return False
    if config.composition_preserving:
        counts = {}
        for site, symbol in zip(sites, assignment):
            if site.mixed_occupancy:
                for original, amount in site.original_species:
                    counts[original] = counts.get(original, 0.0) + site.multiplicity * amount
            else:
                counts[symbol] = counts.get(symbol, 0.0) + site.multiplicity
        target = structure.composition.get_el_amt_dict()
        symbols = set(counts) | set(target)
        if any(
            not np.isclose(counts.get(symbol, 0.0), target.get(symbol, 0.0), atol=1e-8)
            for symbol in symbols
        ):
            return False
    if config.oxidation_states is not None:
        charge = 0.0
        for site, symbol in zip(sites, assignment):
            if site.mixed_occupancy:
                charge += site.multiplicity * sum(
                    amount * config.oxidation_states[original]
                    for original, amount in site.original_species
                )
            else:
                charge += site.multiplicity * config.oxidation_states[symbol]
        if abs(charge - config.target_charge) > config.charge_tolerance:
            return False
    return True


def _structure_with_assignment(structure, sites, assignment, *, displacement_policy):
    species = [site.species for site in structure]
    for asymmetric_site, symbol in zip(sites, assignment):
        if asymmetric_site.mixed_occupancy:
            continue
        for index in asymmetric_site.equivalent_indices:
            species[index] = symbol
    original_properties = {
        name: list(values) for name, values in structure.site_properties.items()
    }
    properties = {name: list(values) for name, values in original_properties.items()}
    if displacement_policy == "species" and properties:
        source_by_key = {}
        for site in sites:
            if not site.mixed_occupancy:
                source_by_key.setdefault(
                    (_ordered_species(site), site.multiplicity),
                    site,
                )
        for target, symbol in zip(sites, assignment):
            if target.mixed_occupancy:
                continue
            source = source_by_key.get((symbol, target.multiplicity))
            if source is None:
                raise ValueError(
                    f"no compatible source properties for {symbol} at site {target.site_index}"
                )
            for target_index, source_index in zip(
                target.equivalent_indices,
                source.equivalent_indices,
            ):
                for name, values in properties.items():
                    values[target_index] = original_properties[name][source_index]
    return Structure(
        structure.lattice,
        species,
        structure.frac_coords,
        site_properties=properties,
        labels=[site.label for site in structure],
        coords_are_cartesian=False,
    )


def _screening_score(dataset, structure, *, background_degree):
    if dataset.domain != "two_theta":
        raise ValueError("species screening currently uses two-theta coordinates")
    calculator = BraggCalculator(
        mode=dataset.radiation,
        wavelength=dataset.wavelength,
        two_theta_range=(
            float(dataset.coordinate[0]),
            float(dataset.coordinate[-1]),
        ),
        two_theta_step=dataset.step,
        primitive=False,
    ).load(structure)
    coordinate, profile = calculator.pattern(domain="two_theta")
    profile = np.interp(dataset.coordinate, np.asarray(coordinate), np.asarray(profile))
    scaled_coordinate = (
        2.0
        * (dataset.coordinate - dataset.coordinate[0])
        / (dataset.coordinate[-1] - dataset.coordinate[0])
        - 1.0
    )
    columns = [profile] + [
        scaled_coordinate**degree for degree in range(background_degree + 1)
    ]
    design = np.column_stack(columns)
    selected = dataset.mask
    weighted_design = design[selected] / dataset.sigma[selected, None]
    weighted_observed = dataset.intensity[selected] / dataset.sigma[selected]
    coefficients = np.linalg.lstsq(
        weighted_design,
        weighted_observed,
        rcond=None,
    )[0]
    if coefficients[0] < 0:
        coefficients[0] = 0.0
        coefficients[1:] = np.linalg.lstsq(
            weighted_design[:, 1:],
            weighted_observed,
            rcond=None,
        )[0]
    calculated = design @ coefficients
    residual = dataset.intensity - calculated
    denominator = np.sum(
        dataset.weights[selected] * dataset.intensity[selected] ** 2
    )
    if denominator <= 0:
        return float(np.sqrt(np.mean((residual[selected] / dataset.sigma[selected]) ** 2)))
    return float(
        np.sqrt(
            np.sum(dataset.weights[selected] * residual[selected] ** 2)
            / denominator
        )
    )


def _continuous_score(candidate, policy):
    if policy.likelihood == "poisson":
        return float(candidate.physical_parameters["mean_poisson_deviance"])
    return float(candidate.r_wp)


def _assignment_id(assignment):
    return "|".join(f"{index}:{symbol}" for index, symbol in enumerate(assignment))


def _site_assignment_records(sites, assignment):
    return tuple(
        SiteAssignment(
            site_index=site.site_index,
            representative_index=site.representative_index,
            equivalent_indices=site.equivalent_indices,
            label=site.label,
            wyckoff_symbol=site.wyckoff_symbol,
            multiplicity=site.multiplicity,
            original_species=site.original_species,
            proposed_species=symbol,
        )
        for site, symbol in zip(sites, assignment)
    )
