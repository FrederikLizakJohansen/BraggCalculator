"""High-level powder-diffraction refinement workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from pymatgen.io.cif import CifWriter

from .dataset import DiffractionDataset
from .io import to_pmg_structure
from .session import (
    CandidateRefinementResult,
    RefinementPolicy,
    RefinementSession,
    refined_structure_from_candidate,
)


@dataclass(frozen=True)
class RefinedParameter:
    """One reported physical value and its valid range."""

    path: str
    value: float
    lower_bound: float | None
    upper_bound: float | None
    unit: str | None
    released: bool
    constraint: str | None = None


@dataclass(frozen=True)
class RefinementResult:
    """Complete result from one structure refinement workflow."""

    dataset: DiffractionDataset
    starting_structure: Any
    refined_structure: Any
    refined_cif: str
    coordinate: np.ndarray
    observed: np.ndarray
    calculated: np.ndarray
    residual: np.ndarray
    objective_history: np.ndarray
    stage_history: tuple[str, ...]
    status: Literal["converged", "completed"]
    convergence: Mapping[str, Any]
    parameters: tuple[RefinedParameter, ...]
    fit_statistics: Mapping[str, float | None]
    diagnostics: Mapping[str, Any]
    warnings: tuple[str, ...]
    provenance: Mapping[str, Any]
    candidate: CandidateRefinementResult
    species_assignments: Any | None = None

    def write_cif(self, path: str | PathLike[str]) -> Path:
        """Write the refined CIF text and return its path."""
        output = Path(path)
        output.write_text(self.refined_cif, encoding="utf-8")
        return output


def load_refinement_dataset(
    pattern,
    *,
    wavelength: float | None = None,
    radiation: Literal["xray", "neutron"] = "xray",
    domain: Literal["two_theta", "q"] = "two_theta",
    third_column: Literal["sigma", "weight"] = "sigma",
    sigma=None,
    weights=None,
    mask=None,
    metadata: Mapping[str, object] | None = None,
) -> DiffractionDataset:
    """Load a path or numeric pattern into a checked diffraction dataset."""
    if isinstance(pattern, DiffractionDataset):
        if wavelength is not None and not np.isclose(wavelength, pattern.wavelength):
            raise ValueError("wavelength disagrees with the supplied DiffractionDataset")
        if radiation != pattern.radiation:
            raise ValueError("radiation disagrees with the supplied DiffractionDataset")
        return pattern
    if wavelength is None or not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be supplied as a positive finite value")
    if isinstance(pattern, (str, PathLike)):
        return DiffractionDataset.from_xye(
            pattern,
            domain=domain,
            wavelength=wavelength,
            radiation=radiation,
            third_column=third_column,
            metadata=metadata,
        )

    values = np.asarray(pattern, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in {2, 3}:
        raise ValueError("pattern arrays must have two or three columns: coordinate, intensity, value")
    if values.shape[1] == 3 and (sigma is not None or weights is not None):
        raise ValueError("supply uncertainties in either the third column or a separate argument")
    if sigma is not None and weights is not None:
        raise ValueError("supply sigma or weights, once")
    coordinate = values[:, 0]
    intensity = values[:, 1]
    if values.shape[1] == 3:
        uncertainty = values[:, 2]
        sigma_values = (
            uncertainty
            if third_column == "sigma"
            else _sigma_from_weights(uncertainty)
        )
    elif sigma is not None:
        sigma_values = np.asarray(sigma, dtype=np.float64)
    elif weights is not None:
        sigma_values = _sigma_from_weights(weights)
    else:
        sigma_values = np.sqrt(np.maximum(intensity, 0.0) + 1.0)
    mask_values = (
        np.ones(len(coordinate), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    return DiffractionDataset(
        coordinate=coordinate,
        intensity=intensity,
        sigma=sigma_values,
        mask=mask_values,
        domain=domain,
        wavelength=float(wavelength),
        radiation=radiation,
        metadata={} if metadata is None else metadata,
    )


def refine_structure(
    pattern,
    structure,
    *,
    wavelength: float | None = None,
    radiation: Literal["xray", "neutron"] = "xray",
    domain: Literal["two_theta", "q"] = "two_theta",
    third_column: Literal["sigma", "weight"] = "sigma",
    sigma=None,
    weights=None,
    mask=None,
    metadata: Mapping[str, object] | None = None,
    policy: RefinementPolicy | None = None,
    species_assignment=None,
    device: str = "cpu",
) -> RefinementResult:
    """Refine a structure against an observed constant-wavelength PXRD pattern.

    ``structure`` accepts every structure input supported by
    :func:`braggcalculator.io.to_pmg_structure`, including a CIF path, CIF
    text, and a pymatgen ``Structure``. Pattern coordinates may be two-theta
    degrees or Q in inverse angstroms.
    """
    dataset = load_refinement_dataset(
        pattern,
        wavelength=wavelength,
        radiation=radiation,
        domain=domain,
        third_column=third_column,
        sigma=sigma,
        weights=weights,
        mask=mask,
        metadata=metadata,
    )
    structure = to_pmg_structure(structure)
    selected_policy = RefinementPolicy.quick() if policy is None else policy
    if species_assignment is None:
        candidate = RefinementSession(
            dataset,
            [structure],
            names=["candidate"],
            device=device,
        ).run(selected_policy).candidates[0]
        return _result_from_candidate(dataset, candidate)

    from .species_assignment import refine_species_assignments

    assignment_result = refine_species_assignments(
        dataset,
        structure,
        config=species_assignment,
        policy=selected_policy,
        device=device,
    )
    if not assignment_result.candidates:
        raise RuntimeError("species-assignment search produced no valid candidate")
    best = assignment_result.candidates[0]
    result = _result_from_candidate(dataset, best.continuous_result)
    return replace(result, species_assignments=assignment_result)


def _sigma_from_weights(weights) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be positive and finite")
    return 1.0 / np.sqrt(values)


def _result_from_candidate(
    dataset: DiffractionDataset,
    candidate: CandidateRefinementResult,
) -> RefinementResult:
    refined = refined_structure_from_candidate(candidate, dataset)
    refined_cif = str(CifWriter(refined, symprec=None))
    convergence = {} if candidate.convergence is None else dict(candidate.convergence)
    classification = str(convergence.get("classification", "maximum_steps"))
    status: Literal["converged", "completed"] = (
        "converged"
        if classification in {"gradient_converged", "loss_stalled"}
        else "completed"
    )
    parameters = _parameter_table(candidate)
    return RefinementResult(
        dataset=dataset,
        starting_structure=candidate.structure,
        refined_structure=refined,
        refined_cif=refined_cif,
        coordinate=dataset.coordinate.copy(),
        observed=dataset.intensity.copy(),
        calculated=candidate.calculated.copy(),
        residual=candidate.residual.copy(),
        objective_history=candidate.loss_history.copy(),
        stage_history=candidate.stage_history,
        status=status,
        convergence=convergence,
        parameters=parameters,
        fit_statistics={
            "r_wp": candidate.r_wp,
            "chi_squared": candidate.chi_squared,
            "held_out_r_wp": candidate.held_out_r_wp,
            "mean_poisson_deviance": candidate.physical_parameters.get(
                "mean_poisson_deviance"
            ),
        },
        diagnostics={
            "identifiability": candidate.identifiability,
            "informative_regions": candidate.informative_regions,
            "recommendation": candidate.recommendation,
        },
        warnings=candidate.warnings,
        provenance=candidate.provenance,
        candidate=candidate,
    )


def _parameter_table(candidate: CandidateRefinementResult) -> tuple[RefinedParameter, ...]:
    released = set(candidate.provenance["policy"]["released_parameter_groups"])
    records = []
    for path, value in _flatten_numbers(candidate.physical_parameters):
        lower, upper, unit, constraint, group = _parameter_metadata(path)
        records.append(
            RefinedParameter(
                path=path,
                value=value,
                lower_bound=lower,
                upper_bound=upper,
                unit=unit,
                released=group in released,
                constraint=constraint,
            )
        )
    return tuple(records)


def _flatten_numbers(value, prefix=""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_numbers(item, path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _flatten_numbers(item, f"{prefix}[{index}]")
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        yield prefix, float(value)


def _parameter_metadata(path):
    lower = upper = unit = constraint = None
    group = path.split(".", 1)[0].split("[", 1)[0]
    if path == "scale":
        lower, unit, constraint, group = 0.0, "profile scale", "positive", "scale"
    elif path.startswith("background_coefficients"):
        unit, group = "intensity", "background"
    elif path == "zero_shift":
        unit, group = "degree 2-theta", "zero_shift"
    elif path in {"gaussian_u", "gaussian_w"}:
        lower, unit, constraint, group = 0.0, "degree²", "positive", "profile"
    elif path == "gaussian_v":
        unit, group = "degree²", "profile"
    elif path in {"lorentzian_x", "lorentzian_y"}:
        lower, unit, constraint, group = 0.0, "degree", "positive", "profile"
    elif path == "axial_asymmetry":
        lower, constraint, group = 0.0, "positive", "profile"
    elif path == "eta":
        lower, upper, constraint, group = 0.0, 1.0, "unit interval", "profile"
    elif path == "specimen_displacement_mm":
        unit, group = "mm", "specimen_displacement"
    elif path.startswith("cell_parameters."):
        name = path.rsplit(".", 1)[-1]
        if name in {"a", "b", "c"}:
            lower, unit, constraint = 0.0, "Å", "positive"
        else:
            lower, upper, unit, constraint = 0.0, 180.0, "degree", "cell angle"
        group = "lattice"
    elif path.startswith("lattice"):
        unit, group = "Å", "lattice"
    elif path.startswith("coordinate_displacements"):
        unit, group = "fractional coordinate", "coordinates"
    elif ".species." in path and path.startswith("occupancy_groups"):
        lower, upper, constraint, group = 0.0, 1.0, "occupancy simplex", "occupancies"
    elif path.startswith("isotropic_displacement_groups"):
        lower, unit, constraint, group = 0.0, "Å²", "positive", "b_iso"
    elif path.startswith("anisotropic_displacement_groups"):
        unit, constraint, group = "Å²", "positive-semidefinite tensor", "u_aniso"
    elif path.startswith("rigid_body_groups"):
        group = "rigid_bodies"
    return lower, upper, unit, constraint, group
