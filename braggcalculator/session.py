"""End-to-end candidate-guided experimental refinement sessions."""

from __future__ import annotations

import html
from hashlib import sha256
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .backends import TorchBackend
from .core import BraggCalculator
from .dataset import DiffractionDataset
from .experimental_profile import (
    axial_divergence_widths,
    caglioti_fwhm,
    emission_lorentzian_fwhm,
    render_pseudo_voigt,
    render_split_pseudo_voigt,
    specimen_displacement_shift,
    thompson_cox_hastings,
)
from .io import to_pmg_structure
from .optimization import (
    OptimizationStage,
    recommend_parameter_groups,
    staged_optimize,
)
from .parameters import lattice_parameters
from .restraints import StructuralRestraintSet
from .sensitivity import analyze_jacobian


def _wavelength_components(metadata, default_wavelength):
    """Normalize legacy tuples or explicit component dictionaries."""
    raw = metadata.get("wavelength_components", [(default_wavelength, 1.0)])
    components = []
    for item in raw:
        if isinstance(item, dict):
            wavelength = float(item["wavelength_angstrom"])
            weight = float(item["weight"])
            line_width = float(item.get("lorentzian_fwhm_angstrom", 0.0))
        else:
            if len(item) not in {2, 3}:
                raise ValueError("wavelength component tuples must have two or three values")
            wavelength = float(item[0])
            weight = float(item[1])
            line_width = float(item[2]) if len(item) == 3 else 0.0
        if wavelength <= 0 or not np.isfinite(wavelength):
            raise ValueError("component wavelengths must be positive and finite")
        if weight <= 0 or not np.isfinite(weight):
            raise ValueError("component weights must be positive and finite")
        if line_width < 0 or not np.isfinite(line_width):
            raise ValueError("emission line widths must be finite and non-negative")
        components.append(
            {
                "wavelength_angstrom": wavelength,
                "weight": weight,
                "lorentzian_fwhm_angstrom": line_width,
            }
        )
    weights = np.asarray([item["weight"] for item in components], dtype=np.float64)
    weights /= weights.sum()
    return tuple(
        {**item, "normalized_weight": float(weight)} for item, weight in zip(components, weights)
    )


@dataclass(frozen=True)
class RefinementPolicy:
    """Declared release policy and restraint strengths for one refinement."""

    background_degree: int = 4
    refine_lattice: bool = True
    refine_coordinates: bool = False
    coordinate_restraint: float = 10.0
    occupancy_mode: str = "fixed"
    occupancy_restraint: float = 1.0
    refine_b_iso: bool = False
    b_iso_restraint: float = 0.1
    default_b_iso: float = 0.5
    refine_u_aniso: bool = False
    u_aniso_restraint: float = 0.1
    default_u_iso: float = 0.006
    structural_restraints: dict[str, Any] | None = None
    structural_restraint_weight: float = 1.0
    rigid_bodies: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None
    rigid_body_restraint: float = 0.1
    rigid_translation_scale: float = 0.1
    rigid_rotation_scale_degrees: float = 5.0
    holdout_stride: int = 10
    restarts: int = 1
    diagnostic_points: int = 48
    profile_model: str = "tch"
    axial_asymmetry: bool = True
    goniometer_radius_mm: float | None = None
    specimen_displacement_mm: float = 0.0
    refine_specimen_displacement: bool = False
    likelihood: str = "gaussian"
    validation_rollback_tolerance: float | None = None
    adaptive_release: bool = False
    minimum_relative_sensitivity: float = 0.02
    minimum_residual_support: float = 0.1
    maximum_release_correlation: float = 0.98
    restart_seed: int = 1729
    stages: tuple[OptimizationStage, ...] | None = None

    def __post_init__(self):
        if self.background_degree < 0:
            raise ValueError("background_degree must be non-negative")
        if self.holdout_stride < 2:
            raise ValueError("holdout_stride must be at least 2")
        if self.restarts < 1:
            raise ValueError("restarts must be positive")
        if self.diagnostic_points < 0:
            raise ValueError("diagnostic_points must be non-negative")
        if self.coordinate_restraint < 0 or not np.isfinite(self.coordinate_restraint):
            raise ValueError("coordinate_restraint must be finite and non-negative")
        if self.occupancy_mode not in {"fixed", "composition", "vacancy"}:
            raise ValueError("occupancy_mode must be 'fixed', 'composition', or 'vacancy'")
        if self.occupancy_restraint < 0 or not np.isfinite(self.occupancy_restraint):
            raise ValueError("occupancy_restraint must be finite and non-negative")
        if self.b_iso_restraint < 0 or not np.isfinite(self.b_iso_restraint):
            raise ValueError("b_iso_restraint must be finite and non-negative")
        if not np.isfinite(self.default_b_iso) or self.default_b_iso <= 0:
            raise ValueError("default_b_iso must be positive and finite")
        if self.refine_b_iso and self.refine_u_aniso:
            raise ValueError("isotropic and anisotropic displacement cannot be refined together")
        if self.u_aniso_restraint < 0 or not np.isfinite(self.u_aniso_restraint):
            raise ValueError("u_aniso_restraint must be finite and non-negative")
        if not np.isfinite(self.default_u_iso) or self.default_u_iso <= 0:
            raise ValueError("default_u_iso must be positive and finite")
        if self.structural_restraint_weight < 0 or not np.isfinite(
            self.structural_restraint_weight
        ):
            raise ValueError("structural_restraint_weight must be finite and non-negative")
        if self.structural_restraints is not None and not isinstance(
            self.structural_restraints, dict
        ):
            raise TypeError("structural_restraints must be a dictionary or None")
        if self.rigid_bodies is not None and not isinstance(self.rigid_bodies, (list, tuple)):
            raise TypeError("rigid_bodies must be a list/tuple of declarations or None")
        if self.rigid_bodies and self.refine_coordinates:
            raise ValueError("free coordinates and rigid bodies cannot be refined together")
        if self.rigid_body_restraint < 0 or not np.isfinite(self.rigid_body_restraint):
            raise ValueError("rigid_body_restraint must be finite and non-negative")
        if self.rigid_translation_scale <= 0 or not np.isfinite(self.rigid_translation_scale):
            raise ValueError("rigid_translation_scale must be positive and finite")
        if self.rigid_rotation_scale_degrees <= 0 or not np.isfinite(
            self.rigid_rotation_scale_degrees
        ):
            raise ValueError("rigid_rotation_scale_degrees must be positive and finite")
        if self.profile_model not in {"legacy", "tch"}:
            raise ValueError("profile_model must be 'legacy' or 'tch'")
        if self.goniometer_radius_mm is not None and (
            not np.isfinite(self.goniometer_radius_mm) or self.goniometer_radius_mm <= 0
        ):
            raise ValueError("goniometer_radius_mm must be positive and finite")
        if not np.isfinite(self.specimen_displacement_mm):
            raise ValueError("specimen_displacement_mm must be finite")
        if self.likelihood not in {"gaussian", "poisson"}:
            raise ValueError("likelihood must be 'gaussian' or 'poisson'")
        if self.validation_rollback_tolerance is not None and (
            self.validation_rollback_tolerance < 0
            or not np.isfinite(self.validation_rollback_tolerance)
        ):
            raise ValueError("validation_rollback_tolerance must be non-negative or None")
        if not 0 <= self.minimum_relative_sensitivity <= 1:
            raise ValueError("minimum_relative_sensitivity must be in [0, 1]")
        if self.minimum_residual_support < 0:
            raise ValueError("minimum_residual_support must be non-negative")
        if not 0 <= self.maximum_release_correlation <= 1:
            raise ValueError("maximum_release_correlation must be in [0, 1]")
        if not isinstance(self.restart_seed, int) or self.restart_seed < 0:
            raise ValueError("restart_seed must be a non-negative integer")
        if self.stages and self.stages[-1].width_multiplier != 1.0:
            raise ValueError("the final optimization stage must use the physical width multiplier 1.0")

    @classmethod
    def quick(
        cls,
        *,
        refine_coordinates: bool = False,
        occupancy_mode: str = "fixed",
        refine_b_iso: bool = False,
        refine_u_aniso: bool = False,
        rigid_bodies=None,
    ) -> "RefinementPolicy":
        active_joint = (
            "scale",
            "background",
            "zero_shift",
            "profile",
            "lattice",
            "specimen_displacement",
        )
        if refine_coordinates:
            active_joint += ("coordinates",)
        if occupancy_mode != "fixed":
            active_joint += ("occupancies",)
        if refine_b_iso:
            active_joint += ("b_iso",)
        if refine_u_aniso:
            active_joint += ("u_aniso",)
        if rigid_bodies:
            active_joint += ("rigid_bodies",)
        return cls(
            refine_coordinates=refine_coordinates,
            occupancy_mode=occupancy_mode,
            refine_b_iso=refine_b_iso,
            refine_u_aniso=refine_u_aniso,
            rigid_bodies=rigid_bodies,
            stages=(
                OptimizationStage("scale/background", ("scale", "background"), 40, 0.04),
                OptimizationStage(
                    "calibration/profile",
                    ("zero_shift", "profile", "lattice", "specimen_displacement"),
                    60,
                    0.025,
                ),
                OptimizationStage("joint", active_joint, 100, 0.01),
            ),
        )

    @classmethod
    def cautious(
        cls,
        *,
        refine_coordinates: bool = False,
        occupancy_mode: str = "fixed",
        refine_b_iso: bool = False,
        refine_u_aniso: bool = False,
        rigid_bodies=None,
    ) -> "RefinementPolicy":
        active_joint = (
            "scale",
            "background",
            "zero_shift",
            "profile",
            "lattice",
            "specimen_displacement",
        )
        stages = [
            OptimizationStage("scale/background", ("scale", "background"), 120, 0.03),
            OptimizationStage(
                "calibration/profile",
                ("zero_shift", "profile", "lattice", "specimen_displacement"),
                180,
                0.015,
            ),
        ]
        if refine_coordinates:
            stages.append(OptimizationStage("coordinates", ("coordinates",), 150, 0.006))
            active_joint += ("coordinates",)
        if occupancy_mode != "fixed":
            stages.append(OptimizationStage("occupancies", ("occupancies",), 140, 0.008))
            active_joint += ("occupancies",)
        if refine_b_iso:
            stages.append(OptimizationStage("isotropic displacement", ("b_iso",), 140, 0.008))
            active_joint += ("b_iso",)
        if refine_u_aniso:
            stages.append(OptimizationStage("anisotropic displacement", ("u_aniso",), 180, 0.006))
            active_joint += ("u_aniso",)
        if rigid_bodies:
            stages.append(OptimizationStage("rigid bodies", ("rigid_bodies",), 180, 0.008))
            active_joint += ("rigid_bodies",)
        stages.append(OptimizationStage("joint", active_joint, 250, 0.005))
        return cls(
            refine_coordinates=refine_coordinates,
            occupancy_mode=occupancy_mode,
            refine_b_iso=refine_b_iso,
            refine_u_aniso=refine_u_aniso,
            rigid_bodies=rigid_bodies,
            stages=tuple(stages),
        )

    @classmethod
    def robust(
        cls,
        *,
        refine_coordinates: bool = False,
        occupancy_mode: str = "fixed",
        refine_b_iso: bool = False,
        refine_u_aniso: bool = False,
        rigid_bodies=None,
        likelihood: str = "gaussian",
        restarts: int = 3,
    ) -> "RefinementPolicy":
        """Guarded coarse-to-fine recipe ending in an L-BFGS polish."""
        active = ["scale", "background", "zero_shift", "profile", "lattice"]
        optional = []
        if refine_coordinates:
            optional.append("coordinates")
        if occupancy_mode != "fixed":
            optional.append("occupancies")
        if refine_b_iso:
            optional.append("b_iso")
        if refine_u_aniso:
            optional.append("u_aniso")
        if rigid_bodies:
            optional.append("rigid_bodies")
        return cls(
            refine_coordinates=refine_coordinates,
            occupancy_mode=occupancy_mode,
            refine_b_iso=refine_b_iso,
            refine_u_aniso=refine_u_aniso,
            rigid_bodies=rigid_bodies,
            likelihood=likelihood,
            restarts=restarts,
            adaptive_release=bool(optional),
            validation_rollback_tolerance=0.02,
            stages=(
                OptimizationStage(
                    "wide profile", tuple(active), 100, 0.02, width_multiplier=2.5
                ),
                OptimizationStage(
                    "intermediate profile",
                    tuple(active + optional),
                    140,
                    0.01,
                    width_multiplier=1.5,
                ),
                OptimizationStage(
                    "physical profile",
                    tuple(active + optional),
                    120,
                    0.006,
                    width_multiplier=1.0,
                ),
                OptimizationStage(
                    "L-BFGS polish",
                    tuple(active + optional),
                    60,
                    0.5,
                    optimizer="lbfgs",
                    width_multiplier=1.0,
                ),
            ),
        )


@dataclass(frozen=True)
class CandidateRefinementResult:
    name: str
    structure: Any
    calculated: np.ndarray
    residual: np.ndarray
    physical_parameters: dict[str, Any]
    r_wp: float
    chi_squared: float
    held_out_r_wp: float | None
    loss_history: np.ndarray
    stage_history: tuple[str, ...]
    informative_regions: tuple[dict[str, float], ...]
    identifiability: dict[str, Any]
    recommendation: str
    warnings: tuple[str, ...]
    provenance: dict[str, Any]
    convergence: dict[str, Any] | None = None


@dataclass(frozen=True)
class SessionResult:
    dataset: DiffractionDataset
    candidates: tuple[CandidateRefinementResult, ...]
    ranking: tuple[str, ...]
    pairwise_discrimination: dict[str, float]
    conclusion: str


class RefinementSession:
    """Refine and compare plausible structures against one powder dataset."""

    def __init__(self, dataset: DiffractionDataset, models, *, names=None, device="cpu"):
        self.input_dataset = dataset
        self.dataset = dataset.convert_domain("two_theta")
        self.structures = tuple(to_pmg_structure(model) for model in models)
        if not self.structures:
            raise ValueError("at least one candidate model is required")
        self.names = (
            tuple(names)
            if names is not None
            else tuple(f"model_{index + 1}" for index in range(len(self.structures)))
        )
        if len(self.names) != len(self.structures) or len(set(self.names)) != len(self.names):
            raise ValueError("candidate names must be unique and match the model count")
        self.device = device

    def _calculator(self, structure):
        return BraggCalculator(
            mode=self.dataset.radiation,
            wavelength=self.dataset.wavelength,
            two_theta_range=(float(self.dataset.coordinate[0]), float(self.dataset.coordinate[-1])),
            two_theta_step=self.dataset.step,
            backend=TorchBackend(device=self.device),
            primitive=False,
        ).load(structure)

    def refine_candidate(
        self,
        index: int,
        policy: RefinementPolicy | None = None,
        *,
        restart_index: int = 0,
        checkpoint: dict[str, Any] | None = None,
    ):
        import torch

        policy = RefinementPolicy.quick() if policy is None else policy
        if policy.likelihood == "poisson":
            if np.any(self.dataset.intensity[self.dataset.mask] < 0):
                raise ValueError("Poisson refinement requires non-negative observed counts")
            if self.dataset.observation_covariance is not None:
                raise ValueError("Poisson refinement does not accept a Gaussian covariance matrix")
        calculator = self._calculator(self.structures[index])
        components = _wavelength_components(self.dataset.metadata, self.dataset.wavelength)
        component_weights = np.asarray([item["normalized_weight"] for item in components])
        coordinate_model = calculator.symmetry_coordinate_parameterization()
        lattice_model = calculator.symmetry_lattice_parameterization()
        occupancy_model = (
            calculator.symmetry_occupancy_parameterization(mode=policy.occupancy_mode)
            if policy.occupancy_mode != "fixed"
            else None
        )
        b_iso_model = (
            calculator.symmetry_b_iso_parameterization(default_if_zero=policy.default_b_iso)
            if policy.refine_b_iso
            else None
        )
        u_aniso_model = (
            calculator.symmetry_u_aniso_parameterization(default_u_iso=policy.default_u_iso)
            if policy.refine_u_aniso
            else None
        )
        restraint_set = StructuralRestraintSet.from_dict(calculator, policy.structural_restraints)
        rigid_body_model = (
            calculator.rigid_body_parameterization(
                policy.rigid_bodies,
                translation_scale=policy.rigid_translation_scale,
                rotation_scale_degrees=policy.rigid_rotation_scale_degrees,
            )
            if policy.rigid_bodies
            else None
        )
        base_parameters = calculator.tensor_parameters()
        grid = torch.as_tensor(self.dataset.coordinate, dtype=torch.float64, device=self.device)
        observed = torch.as_tensor(self.dataset.intensity, dtype=torch.float64, device=self.device)
        sigma = torch.as_tensor(self.dataset.sigma, dtype=torch.float64, device=self.device)
        selected = self.dataset.mask.copy()
        held_out = np.zeros(len(selected), dtype=bool)
        if policy.holdout_stride > 1:
            held_out[np.flatnonzero(selected)[:: policy.holdout_stride]] = True
        training = selected & ~held_out
        training_tensor = torch.as_tensor(training, dtype=torch.bool, device=self.device)
        held_out_tensor = torch.as_tensor(held_out, dtype=torch.bool, device=self.device)
        training_indices = np.flatnonzero(training)
        covariance_cholesky = None
        held_out_cholesky = None
        if self.dataset.observation_covariance is not None:
            training_covariance = self.dataset.observation_covariance[
                np.ix_(training_indices, training_indices)
            ]
            covariance_cholesky = torch.linalg.cholesky(
                torch.as_tensor(training_covariance, dtype=torch.float64, device=self.device)
            )
            held_out_indices = np.flatnonzero(held_out)
            if len(held_out_indices):
                held_out_covariance = self.dataset.observation_covariance[
                    np.ix_(held_out_indices, held_out_indices)
                ]
                held_out_cholesky = torch.linalg.cholesky(
                    torch.as_tensor(
                        held_out_covariance, dtype=torch.float64, device=self.device
                    )
                )

        def whiten_training(residual):
            selected_residual = residual[training_tensor]
            if covariance_cholesky is None:
                return selected_residual / sigma[training_tensor]
            return torch.linalg.solve_triangular(
                covariance_cholesky,
                selected_residual[:, None],
                upper=False,
            )[:, 0]

        def whiten_validation(residual):
            selected_residual = residual[held_out_tensor]
            if held_out_cholesky is None:
                return selected_residual / sigma[held_out_tensor]
            return torch.linalg.solve_triangular(
                held_out_cholesky,
                selected_residual[:, None],
                upper=False,
            )[:, 0]

        with torch.no_grad():
            component_areas = [
                float(torch.sum(item[1]).cpu())
                for item in calculator.line_components(
                    [component["wavelength_angstrom"] for component in components],
                    domain="two_theta",
                )
            ]
            background0 = max(float(np.percentile(self.dataset.intensity[selected], 10)), 1e-6)
            signal_area = max(
                float(
                    np.trapezoid(
                        np.maximum(self.dataset.intensity - background0, 0.0),
                        self.dataset.coordinate,
                    )
                ),
                1e-9,
            )
            calculated_area = max(float(component_weights @ component_areas), 1e-9)
            scale0 = signal_area / calculated_area

        scale = torch.zeros((), dtype=torch.float64, device=self.device, requires_grad=True)
        background = torch.zeros(
            policy.background_degree + 1,
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        with torch.no_grad():
            background[0] = background0
        zero_shift = torch.zeros((), dtype=torch.float64, device=self.device, requires_grad=True)
        profile_size = 4 if policy.profile_model == "legacy" else 6
        profile = torch.zeros(
            profile_size, dtype=torch.float64, device=self.device, requires_grad=True
        )
        lattice = lattice_model.initial_values(calculator.backend, requires_grad=True)
        displacement = torch.zeros((), dtype=torch.float64, device=self.device, requires_grad=True)
        coordinates = coordinate_model.initial_values(calculator.backend, requires_grad=True)
        occupancies = (
            occupancy_model.initial_values(calculator.backend, requires_grad=True)
            if occupancy_model is not None
            else None
        )
        b_iso = (
            b_iso_model.initial_values(calculator.backend, requires_grad=True)
            if b_iso_model is not None
            else None
        )
        u_aniso = (
            u_aniso_model.initial_values(calculator.backend, requires_grad=True)
            if u_aniso_model is not None
            else None
        )
        rigid_bodies = (
            rigid_body_model.initial_values(calculator.backend, requires_grad=True)
            if rigid_body_model is not None
            else None
        )
        restart_seed = policy.restart_seed + restart_index
        if restart_index:
            generator = np.random.default_rng(restart_seed)
            with torch.no_grad():
                scale.copy_(
                    torch.tensor(
                        generator.normal(0.0, 0.05), dtype=torch.float64, device=self.device
                    )
                )
                zero_shift.copy_(
                    torch.tensor(
                        generator.normal(0.0, 0.1), dtype=torch.float64, device=self.device
                    )
                )
                profile.copy_(
                    torch.as_tensor(
                        generator.normal(0.0, 0.05, profile_size),
                        dtype=torch.float64,
                        device=self.device,
                    )
                )
                lattice.copy_(
                    torch.as_tensor(
                        generator.normal(0.0, 0.05, lattice_model.independent_count),
                        dtype=torch.float64,
                        device=self.device,
                    )
                )
                if coordinate_model.independent_count:
                    coordinates.copy_(
                        torch.as_tensor(
                            generator.normal(0.0, 1e-3, coordinate_model.independent_count),
                            dtype=torch.float64,
                            device=self.device,
                        )
                    )
                if occupancy_model is not None and occupancy_model.independent_count:
                    occupancies.copy_(
                        torch.as_tensor(
                            occupancy_model.initial_raw_values
                            + generator.normal(0.0, 0.02, occupancy_model.independent_count),
                            dtype=torch.float64,
                            device=self.device,
                        )
                    )
                if b_iso_model is not None:
                    b_iso.copy_(
                        torch.as_tensor(
                            b_iso_model.initial_raw_values
                            + generator.normal(0.0, 0.02, b_iso_model.independent_count),
                            dtype=torch.float64,
                            device=self.device,
                        )
                    )
                if u_aniso_model is not None:
                    u_aniso.copy_(
                        torch.as_tensor(
                            generator.normal(0.0, 0.02, u_aniso_model.independent_count),
                            dtype=torch.float64,
                            device=self.device,
                        )
                    )
                if rigid_body_model is not None:
                    rigid_bodies.copy_(
                        torch.as_tensor(
                            generator.normal(0.0, 0.02, rigid_body_model.independent_count),
                            dtype=torch.float64,
                            device=self.device,
                        )
                    )
        groups = {
            "scale": scale,
            "background": background,
            "zero_shift": zero_shift,
            "profile": profile,
        }
        if policy.refine_lattice:
            groups["lattice"] = lattice
        if policy.refine_specimen_displacement:
            groups["specimen_displacement"] = displacement
        if policy.refine_coordinates and coordinate_model.independent_count:
            groups["coordinates"] = coordinates
        if occupancy_model is not None and occupancy_model.independent_count:
            groups["occupancies"] = occupancies
        if b_iso_model is not None:
            groups["b_iso"] = b_iso
        if u_aniso_model is not None:
            groups["u_aniso"] = u_aniso
        if rigid_body_model is not None:
            groups["rigid_bodies"] = rigid_bodies
        all_groups = dict(groups)
        if checkpoint is not None:
            if restart_index != 0:
                raise ValueError("checkpoint continuation is only valid for restart index zero")
            raw_groups = checkpoint.get("raw_groups")
            if not isinstance(raw_groups, dict):
                raise ValueError("checkpoint must contain a raw_groups mapping")
            expected = set(all_groups)
            supplied = set(raw_groups)
            if supplied != expected:
                missing = sorted(expected - supplied)
                unknown = sorted(supplied - expected)
                raise ValueError(
                    f"checkpoint parameter groups do not match policy; missing={missing}, "
                    f"unknown={unknown}"
                )
            with torch.no_grad():
                for name, tensor in all_groups.items():
                    restored = torch.as_tensor(
                        raw_groups[name], dtype=torch.float64, device=self.device
                    )
                    if restored.shape != tensor.shape:
                        raise ValueError(
                            f"checkpoint group {name!r} has shape {tuple(restored.shape)}, "
                            f"expected {tuple(tensor.shape)}"
                        )
                    tensor.copy_(restored)

        normalized_x = torch.linspace(-1.0, 1.0, len(grid), dtype=torch.float64, device=self.device)
        background_basis = torch.stack(
            [normalized_x**degree for degree in range(policy.background_degree + 1)], dim=1
        )
        instrument_metadata = self.dataset.metadata.get("instrument", {})
        goniometer_radius = policy.goniometer_radius_mm
        if goniometer_radius is None:
            goniometer_radius = instrument_metadata.get("goniometer_radius_mm")
        if (
            policy.refine_specimen_displacement or abs(policy.specimen_displacement_mm) > 0.0
        ) and goniometer_radius is None:
            raise ValueError("a goniometer radius is required for specimen-displacement correction")

        continuation_width_multiplier = 1.0

        def structural_parameters():
            structural = dict(base_parameters)
            structural["lattice"] = lattice_model.expand(lattice, calculator.backend)
            if policy.refine_coordinates and coordinate_model.independent_count:
                structural["frac_coords"] = coordinate_model.expand(coordinates, calculator.backend)
            if rigid_body_model is not None:
                structural["frac_coords"] = rigid_body_model.expand(
                    rigid_bodies,
                    calculator.backend,
                    lattice=structural["lattice"],
                )
            if occupancy_model is not None and occupancy_model.independent_count:
                structural["occupancies"] = occupancy_model.expand(occupancies, calculator.backend)
            if b_iso_model is not None:
                structural.pop("u_cart", None)
                structural["b_iso"] = b_iso_model.expand(b_iso, calculator.backend)
            if u_aniso_model is not None:
                structural["u_cart"] = u_aniso_model.expand(u_aniso, calculator.backend)
            return structural

        def calculate():
            peaks = torch.zeros_like(grid)
            structural = structural_parameters()
            component_patterns = calculator.line_components(
                [component["wavelength_angstrom"] for component in components],
                domain="two_theta",
                parameters=structural,
            )
            for (peak_centers, peak_areas), component in zip(component_patterns, components):
                component_weight = component["normalized_weight"]
                peak_centers = peak_centers + self.dataset.step * zero_shift
                physical_displacement = (
                    policy.specimen_displacement_mm + 0.05 * displacement
                    if policy.refine_specimen_displacement
                    else policy.specimen_displacement_mm
                )
                if goniometer_radius is not None and (
                    policy.refine_specimen_displacement
                    or abs(policy.specimen_displacement_mm) > 0.0
                ):
                    peak_centers = peak_centers + specimen_displacement_shift(
                        torch.deg2rad(peak_centers),
                        physical_displacement,
                        float(goniometer_radius),
                        calculator.backend,
                    )
                peak_areas = peak_areas * scale0 * torch.exp(scale) * component_weight
                radians = torch.deg2rad(peak_centers)
                if policy.profile_model == "legacy":
                    u = 0.0025 * torch.exp(profile[0])
                    v = 1e-6 * torch.exp(profile[1])
                    w = 0.0064 * torch.exp(profile[2])
                    widths = (
                        caglioti_fwhm(radians, u, v, w, calculator.backend)
                        * continuation_width_multiplier
                    )
                    eta = torch.sigmoid(profile[3])
                    peaks = peaks + render_pseudo_voigt(
                        grid,
                        peak_centers,
                        peak_areas,
                        widths,
                        eta,
                        calculator.backend,
                    )
                else:
                    u = 0.0025 * torch.exp(profile[0])
                    v = 0.001 * torch.sinh(profile[1])
                    w = 0.0036 * torch.exp(profile[2])
                    x = 0.01 * torch.exp(profile[3])
                    y = 0.01 * torch.exp(profile[4])
                    widths, eta = thompson_cox_hastings(
                        radians,
                        u,
                        v,
                        w,
                        x,
                        y,
                        calculator.backend,
                        extra_lorentzian=emission_lorentzian_fwhm(
                            radians,
                            component["wavelength_angstrom"],
                            component["lorentzian_fwhm_angstrom"],
                            calculator.backend,
                        ),
                    )
                    asymmetry = 0.05 * torch.exp(profile[5]) if policy.axial_asymmetry else 0.0
                    low_widths, high_widths = axial_divergence_widths(
                        widths, radians, asymmetry, calculator.backend
                    )
                    low_widths = low_widths * continuation_width_multiplier
                    high_widths = high_widths * continuation_width_multiplier
                    peaks = peaks + render_split_pseudo_voigt(
                        grid,
                        peak_centers,
                        peak_areas,
                        low_widths,
                        high_widths,
                        eta,
                        calculator.backend,
                    )
            calculated = peaks + background_basis @ background
            return (
                _positive_expected_counts(calculated)
                if policy.likelihood == "poisson"
                else calculated
            )

        adaptive_decisions = ()
        optional_structural_groups = tuple(
            name
            for name in ("coordinates", "occupancies", "b_iso", "u_aniso", "rigid_bodies")
            if name in groups
        )
        if policy.adaptive_release and optional_structural_groups:
            sensitivity, support, correlation = _group_release_evidence(
                calculate,
                {name: groups[name] for name in optional_structural_groups},
                self.dataset.intensity,
                self.dataset.sigma,
                training,
                max_points=min(max(policy.diagnostic_points, 8), 32),
            )
            adaptive_decisions = recommend_parameter_groups(
                sensitivity,
                support,
                correlation,
                minimum_relative_sensitivity=policy.minimum_relative_sensitivity,
                minimum_residual_support=policy.minimum_residual_support,
                maximum_correlation=policy.maximum_release_correlation,
            )
            rejected = {decision.group for decision in adaptive_decisions if not decision.accepted}
            groups = {name: value for name, value in groups.items() if name not in rejected}

        def objective():
            calculated = calculate()
            if policy.likelihood == "poisson":
                loss = torch.mean(
                    _poisson_deviance_torch(observed[training_tensor], calculated[training_tensor])
                )
            else:
                standardized = whiten_training(calculated - observed)
                loss = torch.mean(standardized**2)
            negative_background = torch.relu(-(background_basis @ background))
            loss = loss + 0.01 * torch.mean(negative_background**2)
            if policy.refine_coordinates and coordinate_model.independent_count:
                loss = loss + policy.coordinate_restraint * torch.mean(coordinates**2)
            if occupancy_model is not None and occupancy_model.independent_count:
                occupancy_initial = torch.as_tensor(
                    occupancy_model.initial_raw_values,
                    dtype=torch.float64,
                    device=self.device,
                )
                loss = loss + policy.occupancy_restraint * torch.mean(
                    (occupancies - occupancy_initial) ** 2
                )
            if b_iso_model is not None:
                b_iso_initial = torch.as_tensor(
                    b_iso_model.initial_raw_values,
                    dtype=torch.float64,
                    device=self.device,
                )
                loss = loss + policy.b_iso_restraint * torch.mean((b_iso - b_iso_initial) ** 2)
            if u_aniso_model is not None:
                loss = loss + policy.u_aniso_restraint * torch.mean(u_aniso**2)
            if rigid_body_model is not None:
                loss = loss + policy.rigid_body_restraint * torch.mean(rigid_bodies**2)
            if restraint_set.count:
                structural = structural_parameters()
                restraint_loss, _ = restraint_set.loss(
                    structural["lattice"],
                    structural["frac_coords"],
                    structural["occupancies"],
                    calculator.backend,
                )
                loss = loss + policy.structural_restraint_weight * restraint_loss
            return loss

        def validation_objective():
            calculated = calculate()
            if policy.likelihood == "poisson":
                return torch.mean(
                    _poisson_deviance_torch(observed[held_out_tensor], calculated[held_out_tensor])
                )
            standardized = whiten_validation(calculated - observed)
            return torch.mean(standardized**2)

        stages = (
            policy.stages
            or RefinementPolicy.cautious(
                refine_coordinates=policy.refine_coordinates,
                occupancy_mode=policy.occupancy_mode,
                refine_b_iso=policy.refine_b_iso,
                refine_u_aniso=policy.refine_u_aniso,
                rigid_bodies=policy.rigid_bodies,
            ).stages
        )
        stages = tuple(
            OptimizationStage(
                stage.name,
                tuple(name for name in stage.active if name in groups),
                stage.steps,
                stage.learning_rate,
                optimizer=stage.optimizer,
                width_multiplier=stage.width_multiplier,
            )
            for stage in stages
            if any(name in groups for name in stage.active)
        )
        released_names = {name for stage in stages for name in stage.active if name in groups}
        released_groups = {
            name: tensor for name, tensor in groups.items() if name in released_names
        }

        uncertainty_scales = {}
        uncertainty_step_descriptions = {}
        for group_name, tensor in released_groups.items():
            if group_name == "scale":
                uncertainty_scales[group_name] = 0.01
                uncertainty_step_descriptions[group_name] = "0.01 log scale (about 1%)"
            elif group_name == "background":
                uncertainty_scales[group_name] = float(np.median(self.dataset.sigma))
                uncertainty_step_descriptions[group_name] = (
                    "one median marginal observation sigma in intensity units"
                )
            elif group_name == "zero_shift":
                uncertainty_scales[group_name] = 1.0
                uncertainty_step_descriptions[group_name] = (
                    f"one profile bin ({self.dataset.step:.6g} degrees 2-theta)"
                )
            elif group_name == "profile":
                uncertainty_scales[group_name] = 0.1
                uncertainty_step_descriptions[group_name] = "0.1 profile raw/log coefficient"
            elif group_name == "lattice":
                uncertainty_scales[group_name] = 0.1
                uncertainty_step_descriptions[group_name] = "0.001 Cartesian log strain"
            elif group_name == "specimen_displacement":
                uncertainty_scales[group_name] = 0.2
                uncertainty_step_descriptions[group_name] = "0.01 mm specimen displacement"
            elif group_name == "coordinates":
                uncertainty_scales[group_name] = 0.01
                uncertainty_step_descriptions[group_name] = "0.01 fractional-coordinate mode"
            elif group_name == "occupancies":
                uncertainty_scales[group_name] = 0.1
                uncertainty_step_descriptions[group_name] = "0.1 occupancy log-ratio"
            elif group_name == "b_iso":
                local_derivative = torch.sigmoid(tensor).detach().cpu().numpy()
                uncertainty_scales[group_name] = 0.1 / np.maximum(local_derivative, 1e-8)
                uncertainty_step_descriptions[group_name] = "local 0.1 square-angstrom Biso change"
            elif group_name == "u_aniso":
                uncertainty_scales[group_name] = 0.1
                uncertainty_step_descriptions[group_name] = "0.025 Cartesian log-U mode"
            elif group_name == "rigid_bodies":
                descriptions = []
                for _ in rigid_body_model.bodies:
                    descriptions.extend(
                        [f"{rigid_body_model.translation_scale:.6g} angstrom translation"] * 3
                    )
                    descriptions.extend(
                        [f"{np.degrees(rigid_body_model.rotation_scale):.6g} degree rotation"] * 3
                    )
                uncertainty_scales[group_name] = 1.0
                uncertainty_step_descriptions[group_name] = descriptions

        structural_restraint_released = any(
            name in released_groups
            for name in ("lattice", "coordinates", "occupancies", "rigid_bodies")
        )
        prior_is_active = any(
            (
                name == "coordinates"
                and policy.coordinate_restraint > 0
                or name == "occupancies"
                and policy.occupancy_restraint > 0
                or name == "b_iso"
                and policy.b_iso_restraint > 0
                or name == "u_aniso"
                and policy.u_aniso_restraint > 0
                or name == "rigid_bodies"
                and policy.rigid_body_restraint > 0
            )
            for name in released_groups
        ) or (
            structural_restraint_released
            and restraint_set.count
            and policy.structural_restraint_weight > 0
        )

        def uncertainty_prior_residuals():
            values = []
            observation_count = max(len(training_indices), 1)

            def append_quadratic(name, tensor, target, weight):
                if name not in released_groups or weight <= 0:
                    return
                factor = np.sqrt(observation_count * weight / tensor.numel())
                values.append(factor * (tensor - target).reshape(-1))

            append_quadratic("coordinates", coordinates, 0.0, policy.coordinate_restraint)
            if occupancy_model is not None:
                occupancy_initial = torch.as_tensor(
                    occupancy_model.initial_raw_values,
                    dtype=torch.float64,
                    device=self.device,
                )
                append_quadratic(
                    "occupancies", occupancies, occupancy_initial, policy.occupancy_restraint
                )
            if b_iso_model is not None:
                b_iso_initial = torch.as_tensor(
                    b_iso_model.initial_raw_values,
                    dtype=torch.float64,
                    device=self.device,
                )
                append_quadratic("b_iso", b_iso, b_iso_initial, policy.b_iso_restraint)
            append_quadratic("u_aniso", u_aniso, 0.0, policy.u_aniso_restraint)
            append_quadratic("rigid_bodies", rigid_bodies, 0.0, policy.rigid_body_restraint)
            if (
                structural_restraint_released
                and restraint_set.count
                and policy.structural_restraint_weight > 0
            ):
                structural = structural_parameters()
                restraint_residuals = restraint_set.residuals(
                    structural["lattice"],
                    structural["frac_coords"],
                    structural["occupancies"],
                    calculator.backend,
                )
                factor = np.sqrt(
                    observation_count * policy.structural_restraint_weight / restraint_set.count
                )
                values.append(factor * torch.stack(tuple(restraint_residuals.values())))
            return torch.cat(values)

        def prepare_stage(stage):
            nonlocal continuation_width_multiplier
            continuation_width_multiplier = stage.width_multiplier

        trace = staged_optimize(
            objective,
            groups,
            stages,
            validation_objective=(
                validation_objective
                if np.any(held_out) and policy.validation_rollback_tolerance is not None
                else None
            ),
            before_stage=prepare_stage,
            validation_tolerance=policy.validation_rollback_tolerance or 0.0,
        )
        calculated = calculate().detach().cpu().numpy()
        residual = self.dataset.intensity - calculated
        weights = self.dataset.weights
        denominator = np.sum(weights[selected] * self.dataset.intensity[selected] ** 2)
        r_wp = float(np.sqrt(np.sum(weights[selected] * residual[selected] ** 2) / denominator))
        if self.dataset.observation_covariance is None:
            whitened_residual = residual[training] / self.dataset.sigma[training]
        else:
            whitened_residual = np.linalg.solve(
                np.linalg.cholesky(
                    self.dataset.observation_covariance[np.ix_(training_indices, training_indices)]
                ),
                residual[training],
            )
        chi_squared = float(np.mean(whitened_residual**2))
        poisson_deviance = (
            float(
                np.mean(
                    _poisson_deviance_numpy(
                        self.dataset.intensity[training], calculated[training]
                    )
                )
            )
            if policy.likelihood == "poisson"
            else None
        )
        held_out_r_wp = None
        if np.any(held_out):
            held_denominator = np.sum(weights[held_out] * self.dataset.intensity[held_out] ** 2)
            held_out_r_wp = float(
                np.sqrt(np.sum(weights[held_out] * residual[held_out] ** 2) / held_denominator)
            )
        refined_lattice = lattice_model.expand(lattice, calculator.backend).detach().cpu().numpy()
        if policy.profile_model == "legacy":
            profile_physical = {
                "profile_model": "legacy pseudo-Voigt",
                "caglioti_u": float((0.0025 * torch.exp(profile[0])).detach().cpu()),
                "caglioti_v": float((1e-6 * torch.exp(profile[1])).detach().cpu()),
                "caglioti_w": float((0.0064 * torch.exp(profile[2])).detach().cpu()),
                "eta": float(torch.sigmoid(profile[3]).detach().cpu()),
            }
        else:
            profile_physical = {
                "profile_model": "TCH split pseudo-Voigt",
                "gaussian_u": float((0.0025 * torch.exp(profile[0])).detach().cpu()),
                "gaussian_v": float((0.001 * torch.sinh(profile[1])).detach().cpu()),
                "gaussian_w": float((0.0036 * torch.exp(profile[2])).detach().cpu()),
                "lorentzian_x": float((0.01 * torch.exp(profile[3])).detach().cpu()),
                "lorentzian_y": float((0.01 * torch.exp(profile[4])).detach().cpu()),
                "axial_asymmetry": float(
                    (0.05 * torch.exp(profile[5])).detach().cpu() if policy.axial_asymmetry else 0.0
                ),
            }
        final_structural = structural_parameters()
        restraint_loss, restraint_terms = restraint_set.loss(
            final_structural["lattice"],
            final_structural["frac_coords"],
            final_structural["occupancies"],
            calculator.backend,
        )
        restraint_contributions = {
            name: float(value.detach().cpu()) for name, value in restraint_terms.items()
        }
        physical = {
            "scale": float(scale0 * torch.exp(scale).detach().cpu()),
            "background_coefficients": background.detach().cpu().numpy().tolist(),
            "zero_shift": float((self.dataset.step * zero_shift).detach().cpu()),
            **profile_physical,
            "specimen_displacement_mm": float(
                policy.specimen_displacement_mm + (0.05 * displacement).detach().cpu()
                if policy.refine_specimen_displacement
                else policy.specimen_displacement_mm
            ),
            "goniometer_radius_mm": goniometer_radius,
            "lattice": refined_lattice.tolist(),
            "cell_parameters": lattice_parameters(refined_lattice),
            "lattice_parameterization": {
                "crystal_system": lattice_model.crystal_system,
                "mode_labels": list(lattice_model.labels),
                "log_strain_coordinates": lattice.detach().cpu().numpy().tolist(),
            },
            "coordinate_displacements": coordinates.detach().cpu().numpy().tolist(),
            "occupancy_mode": policy.occupancy_mode,
            "occupancy_groups": (
                list(occupancy_model.physical_groups(occupancies.detach().cpu().numpy()))
                if occupancy_model is not None
                else []
            ),
            "isotropic_displacement_groups": (
                list(b_iso_model.physical_groups(b_iso.detach().cpu().numpy()))
                if b_iso_model is not None
                else []
            ),
            "anisotropic_displacement_groups": (
                list(u_aniso_model.physical_groups(u_aniso.detach().cpu().numpy()))
                if u_aniso_model is not None
                else []
            ),
            "rigid_body_groups": (
                list(rigid_body_model.physical_groups(rigid_bodies.detach().cpu().numpy()))
                if rigid_body_model is not None
                else []
            ),
            "structural_restraint_mean_chi_squared": float(restraint_loss.detach().cpu()),
            "structural_restraint_contributions": restraint_contributions,
            "fit_objective": policy.likelihood,
            "mean_poisson_deviance": poisson_deviance,
            "adaptive_release": [
                {
                    "group": item.group,
                    "accepted": item.accepted,
                    "sensitivity": item.sensitivity,
                    "residual_support": item.residual_support,
                    "maximum_correlation": item.maximum_correlation,
                    "reason": item.reason,
                }
                for item in adaptive_decisions
            ],
        }
        warnings = []
        declared_limitations = self.dataset.metadata.get("model_limitations", ())
        if isinstance(declared_limitations, str):
            declared_limitations = (declared_limitations,)
        warnings.extend(str(item) for item in declared_limitations)
        if r_wp > 0.15:
            warnings.append(
                "Large profile residual: the instrument/background model is incomplete."
            )
        if policy.refine_coordinates:
            warnings.append("Coordinate uncertainties are not yet calibrated for experimental use.")
        if policy.occupancy_mode != "fixed":
            warnings.append("Occupancy uncertainties are not yet calibrated for experimental use.")
        if policy.refine_b_iso:
            warnings.append("Biso uncertainties are not yet calibrated for experimental use.")
        if policy.refine_u_aniso:
            warnings.append(
                "Anisotropic displacement uncertainties are not yet calibrated for "
                "experimental use."
            )
        if restraint_set.count:
            warnings.append(
                "Structural restraints contribute prior information; inspect their separate "
                "penalties before interpreting the fit."
            )
        if rigid_body_model is not None:
            warnings.append(
                "Declared rigid-body motion may break the starting space-group symmetry; "
                "validate the resulting model explicitly."
            )
        if r_wp > 0.15 or declared_limitations:
            recommendation = (
                "Improve the wavelength, instrument-profile, background, or phase model before "
                "interpreting structural parameters."
            )
        elif not (
            policy.refine_coordinates
            or policy.occupancy_mode != "fixed"
            or policy.refine_b_iso
            or policy.refine_u_aniso
            or rigid_body_model is not None
        ):
            recommendation = "Inspect parameter sensitivity and correlations before releasing structural coordinates."
        else:
            recommendation = (
                "Validate the refined structural parameters across restarts and held-out regions."
            )
        regions = _informative_regions(
            self.input_dataset.coordinate, residual / self.dataset.sigma, count=5
        )
        identifiability = _local_identifiability(
            calculate,
            released_groups,
            self.dataset.sigma,
            training,
            max_points=policy.diagnostic_points,
            observation_covariance=self.dataset.observation_covariance,
            group_scales=uncertainty_scales,
            group_step_descriptions=uncertainty_step_descriptions,
            prior_residuals=uncertainty_prior_residuals if prior_is_active else None,
            group_labels={
                "lattice": lattice_model.labels,
                "occupancies": (occupancy_model.labels if occupancy_model is not None else ()),
                "b_iso": b_iso_model.labels if b_iso_model is not None else (),
                "u_aniso": u_aniso_model.labels if u_aniso_model is not None else (),
                "rigid_bodies": (rigid_body_model.labels if rigid_body_model is not None else ()),
                "profile": (
                    ("U", "V", "W", "eta")
                    if policy.profile_model == "legacy"
                    else ("U", "V", "W", "X", "Y", "axial_asymmetry")
                ),
            },
        )
        if identifiability and not identifiability["data_covariance_is_identifiable"]:
            warnings.append(
                "The diffraction data Jacobian is locally rank deficient; inspect the "
                "reported null parameter combinations."
            )
        if (
            identifiability
            and not identifiability["data_covariance_is_identifiable"]
            and identifiability["posterior_covariance_is_identifiable"]
        ):
            warnings.append(
                "Restraints or priors make the posterior finite, but they do not make "
                "the corresponding directions identifiable from diffraction data."
            )
        if identifiability and identifiability["maximum_absolute_correlation"] > 0.98:
            warnings.append("At least one released parameter pair is extremely correlated.")
        occupancy_b_correlation = _maximum_cross_group_correlation(
            identifiability, "occupancies.", "b_iso."
        )
        if occupancy_b_correlation > 0.85:
            warnings.append(
                "Occupancy and Biso directions are strongly correlated; do not interpret "
                "their joint refinement independently."
            )
        return CandidateRefinementResult(
            name=self.names[index],
            structure=self.structures[index],
            calculated=calculated,
            residual=residual,
            physical_parameters=physical,
            r_wp=r_wp,
            chi_squared=chi_squared,
            held_out_r_wp=held_out_r_wp,
            loss_history=trace.loss,
            stage_history=trace.stage,
            informative_regions=regions,
            identifiability=identifiability,
            recommendation=recommendation,
            warnings=tuple(warnings),
            provenance={
                "dataset_sha256": self.dataset.source_sha256,
                "coordinate_system": {
                    "input_domain": self.input_dataset.domain,
                    "refinement_domain": self.dataset.domain,
                    "wavelength_angstrom": self.dataset.wavelength,
                },
                "observation_uncertainty": {
                    "model": (
                        "full covariance"
                        if self.dataset.observation_covariance is not None
                        else "independent marginal sigma"
                    ),
                    "covariance_sha256": self.dataset.observation_covariance_sha256,
                },
                "wavelength_components": [dict(item) for item in components],
                "policy": {
                    "background_degree": policy.background_degree,
                    "refine_lattice": policy.refine_lattice,
                    "refine_coordinates": policy.refine_coordinates,
                    "coordinate_restraint": policy.coordinate_restraint,
                    "occupancy_mode": policy.occupancy_mode,
                    "occupancy_restraint": policy.occupancy_restraint,
                    "refine_b_iso": policy.refine_b_iso,
                    "b_iso_restraint": policy.b_iso_restraint,
                    "default_b_iso": policy.default_b_iso,
                    "refine_u_aniso": policy.refine_u_aniso,
                    "u_aniso_restraint": policy.u_aniso_restraint,
                    "default_u_iso": policy.default_u_iso,
                    "structural_restraint_weight": policy.structural_restraint_weight,
                    "structural_restraints": restraint_set.specification(),
                    "rigid_bodies": list(policy.rigid_bodies or ()),
                    "rigid_body_restraint": policy.rigid_body_restraint,
                    "rigid_translation_scale": policy.rigid_translation_scale,
                    "rigid_rotation_scale_degrees": policy.rigid_rotation_scale_degrees,
                    "holdout_stride": policy.holdout_stride,
                    "diagnostic_points": policy.diagnostic_points,
                    "profile_model": policy.profile_model,
                    "axial_asymmetry": policy.axial_asymmetry,
                    "goniometer_radius_mm": goniometer_radius,
                    "specimen_displacement_mm": policy.specimen_displacement_mm,
                    "refine_specimen_displacement": policy.refine_specimen_displacement,
                    "likelihood": policy.likelihood,
                    "validation_rollback_tolerance": policy.validation_rollback_tolerance,
                    "adaptive_release": policy.adaptive_release,
                    "restart_seed": policy.restart_seed,
                    "released_parameter_groups": sorted(released_names),
                },
                "instrument": instrument_metadata,
                "declared_model_limitations": list(declared_limitations),
                "restart_index": restart_index,
                "restart_seed": restart_seed,
                "resumed_from_checkpoint": checkpoint is not None,
                "resume_checkpoint_sha256": (
                    sha256(
                        json.dumps(
                            checkpoint, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest()
                    if checkpoint is not None
                    else None
                ),
                "checkpoint": {
                    "format": "braggcalculator.raw-parameter-state/v1",
                    "raw_groups": {
                        name: tensor.detach().cpu().numpy().tolist()
                        for name, tensor in all_groups.items()
                    },
                },
            },
            convergence={
                "classification": trace.convergence_classification,
                "final_gradient_norm": trace.final_gradient_norm,
                "relative_loss_change": trace.relative_loss_change,
                "stages": [
                    {
                        "name": item.name,
                        "optimizer": item.optimizer,
                        "accepted": item.accepted,
                        "reason": item.reason,
                        "training_before": item.training_before,
                        "training_after": item.training_after,
                        "validation_before": item.validation_before,
                        "validation_after": item.validation_after,
                        "gradient_norm": item.gradient_norm,
                        "width_multiplier": item.width_multiplier,
                    }
                    for item in trace.stage_outcomes
                ],
            },
        )

    def run(
        self,
        policy: RefinementPolicy | None = None,
        *,
        checkpoints: dict[str, dict[str, Any]] | None = None,
    ) -> SessionResult:
        policy = RefinementPolicy.quick() if policy is None else policy
        checkpoints = {} if checkpoints is None else dict(checkpoints)
        unknown = set(checkpoints) - set(self.names)
        if unknown:
            raise ValueError(f"checkpoint supplied for unknown candidates: {sorted(unknown)}")
        candidates = []
        for index in range(len(self.structures)):
            attempts = tuple(
                self.refine_candidate(
                    index,
                    policy=policy,
                    restart_index=restart,
                    checkpoint=(checkpoints.get(self.names[index]) if restart == 0 else None),
                )
                for restart in range(policy.restarts)
            )
            best = min(attempts, key=lambda item: _candidate_objective_score(item, policy))
            restart_rwp = [item.r_wp for item in attempts]
            restart_scores = [_candidate_objective_score(item, policy) for item in attempts]
            extra_warnings = list(best.warnings)
            if len(attempts) > 1 and np.ptp(restart_rwp) > 0.02:
                extra_warnings.append("Refinement is sensitive to starting values across restarts.")
            provenance = dict(best.provenance)
            provenance["restart_rwp"] = restart_rwp
            provenance["restart_objective"] = {
                "name": "mean_poisson_deviance" if policy.likelihood == "poisson" else "r_wp",
                "values": restart_scores,
            }
            provenance["restart_attempts"] = [
                {
                    "restart_index": attempt.provenance["restart_index"],
                    "seed": attempt.provenance["restart_seed"],
                    "r_wp": attempt.r_wp,
                    "classification": attempt.convergence["classification"],
                }
                for attempt in attempts
            ]
            candidates.append(replace(best, warnings=tuple(extra_warnings), provenance=provenance))
        candidates = tuple(candidates)
        ranking = tuple(
            result.name
            for result in sorted(
                candidates, key=lambda item: _candidate_objective_score(item, policy)
            )
        )
        pairwise = {}
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                difference = candidates[left].calculated - candidates[right].calculated
                score = float(
                    _weighted_squared_norm(
                        difference,
                        self.dataset.mask,
                        self.dataset.sigma,
                        self.dataset.observation_covariance,
                    )
                )
                pairwise[f"{candidates[left].name} vs {candidates[right].name}"] = score
        if len(candidates) == 1:
            conclusion = (
                f"Refined {candidates[0].name}; candidate discrimination was not requested."
            )
        elif pairwise and max(pairwise.values()) < 9.0:
            conclusion = "The supplied experiment does not discriminate the refined candidates."
        else:
            metric = "mean Poisson deviance" if policy.likelihood == "poisson" else "Rwp"
            conclusion = (
                f"{ranking[0]} has the lowest {metric}; inspect robustness and residual evidence."
            )
        return SessionResult(self.input_dataset, candidates, ranking, pairwise, conclusion)

    def write_html(self, result: SessionResult, path) -> Path:
        output = Path(path)
        output.write_text(_render_html(result), encoding="utf-8")
        return output


def refined_structure_from_candidate(
    candidate: CandidateRefinementResult,
    dataset: DiffractionDataset,
):
    """Rebuild the final pymatgen structure from a refinement checkpoint."""
    from pymatgen.core import Structure

    dataset = dataset.convert_domain("two_theta")
    policy = candidate.provenance["policy"]
    checkpoint = candidate.provenance["checkpoint"]["raw_groups"]
    calculator = BraggCalculator(
        mode=dataset.radiation,
        wavelength=dataset.wavelength,
        two_theta_range=(float(dataset.coordinate[0]), float(dataset.coordinate[-1])),
        primitive=False,
    ).load(candidate.structure)
    backend = calculator.backend
    parameters = calculator.tensor_parameters()
    raw = {name: np.asarray(value, dtype=np.float64) for name, value in checkpoint.items()}
    lattice_model = calculator.symmetry_lattice_parameterization()
    if "lattice" in raw:
        parameters["lattice"] = lattice_model.expand(raw["lattice"], backend)
    coordinate_model = calculator.symmetry_coordinate_parameterization()
    if "coordinates" in raw:
        parameters["frac_coords"] = coordinate_model.expand(raw["coordinates"], backend)
    if "rigid_bodies" in raw:
        rigid = calculator.rigid_body_parameterization(
            policy["rigid_bodies"],
            translation_scale=policy["rigid_translation_scale"],
            rotation_scale_degrees=policy["rigid_rotation_scale_degrees"],
        )
        parameters["frac_coords"] = rigid.expand(
            raw["rigid_bodies"], backend, lattice=parameters["lattice"]
        )
    if "occupancies" in raw:
        occupancy = calculator.symmetry_occupancy_parameterization(
            mode=policy["occupancy_mode"]
        )
        parameters["occupancies"] = occupancy.expand(raw["occupancies"], backend)
    if "b_iso" in raw:
        model = calculator.symmetry_b_iso_parameterization(
            default_if_zero=policy["default_b_iso"]
        )
        parameters["b_iso"] = model.expand(raw["b_iso"], backend)
        parameters.pop("u_cart", None)
    if "u_aniso" in raw:
        model = calculator.symmetry_u_aniso_parameterization(
            default_u_iso=policy["default_u_iso"]
        )
        parameters["u_cart"] = model.expand(raw["u_aniso"], backend)
    arrays = {name: np.asarray(value) for name, value in parameters.items()}
    site_indices = calculator._symm["site_indices"]
    symbols = calculator._symm["symbols"]
    species, coordinates, b_values, u_values = [], [], [], []
    for site_index in range(len(calculator._symm["structure"])):
        contributions = np.flatnonzero(site_indices == site_index)
        species.append(
            {
                symbols[index]: float(arrays["occupancies"][index])
                for index in contributions
                if arrays["occupancies"][index] > 1e-10
            }
        )
        representative = int(contributions[0])
        coordinates.append(arrays["frac_coords"][representative] % 1.0)
        if "b_iso" in arrays:
            b_values.append(float(arrays["b_iso"][representative]))
        if "u_cart" in arrays:
            u_values.append(arrays["u_cart"][representative].tolist())
    properties = {}
    if b_values:
        properties["B_iso"] = b_values
    if u_values:
        properties["U_cart"] = u_values
    return Structure(
        arrays["lattice"],
        species,
        coordinates,
        site_properties=properties,
        coords_are_cartesian=False,
    )


def _poisson_deviance_torch(observed, calculated):
    """Pointwise Poisson deviance for positive expected counts."""
    import torch

    expected = torch.clamp(calculated, min=1e-12)
    logarithmic = torch.where(
        observed > 0,
        observed * torch.log(torch.clamp(observed, min=1e-12) / expected),
        torch.zeros_like(observed),
    )
    return 2.0 * (expected - observed + logarithmic)


def _candidate_objective_score(candidate, policy):
    if policy.likelihood == "poisson":
        return float(candidate.physical_parameters["mean_poisson_deviance"])
    return candidate.r_wp


def _poisson_deviance_numpy(observed, calculated):
    observed = np.asarray(observed, dtype=np.float64)
    calculated = np.asarray(calculated, dtype=np.float64)
    expected = np.maximum(calculated, 1e-12)
    logarithmic = np.zeros_like(observed)
    positive = observed > 0
    logarithmic[positive] = observed[positive] * np.log(
        observed[positive] / expected[positive]
    )
    return 2.0 * (expected - observed + logarithmic)


def _positive_expected_counts(calculated):
    """Smoothly map an unconstrained profile to strictly positive count means."""
    import torch

    return torch.nn.functional.softplus(calculated, beta=20.0) + 1e-12


def _group_release_evidence(
    calculate,
    groups,
    observed,
    sigma,
    selected,
    *,
    max_points,
):
    """Compute compact whitened evidence for adaptive group release."""
    import torch

    available = np.flatnonzero(selected)
    chosen = available[
        np.linspace(0, len(available) - 1, min(max_points, len(available))).astype(int)
    ]
    profile = calculate()
    residual = (
        np.asarray(observed, dtype=np.float64)[chosen]
        - profile[chosen].detach().cpu().numpy()
    ) / np.asarray(sigma, dtype=np.float64)[chosen]
    group_jacobians = {}
    for name, tensor in groups.items():
        rows = []
        for point in chosen:
            gradient = torch.autograd.grad(
                profile[int(point)], tensor, retain_graph=True, allow_unused=True
            )[0]
            if gradient is None:
                gradient = torch.zeros_like(tensor)
            rows.append(gradient.reshape(-1).detach().cpu().numpy())
        group_jacobians[name] = np.asarray(rows) / np.asarray(sigma)[chosen, None]

    sensitivity = {}
    support = {}
    for name, jacobian in group_jacobians.items():
        sensitivity[name] = float(np.linalg.norm(jacobian))
        projection = jacobian.T @ residual
        support[name] = float(np.linalg.norm(projection) / max(sensitivity[name], 1e-30))
    correlation = {}
    names = tuple(groups)
    for left_index, left in enumerate(names):
        left_q = _orthonormal_columns(group_jacobians[left])
        for right in names[left_index + 1 :]:
            right_q = _orthonormal_columns(group_jacobians[right])
            value = (
                0.0
                if left_q.shape[1] == 0 or right_q.shape[1] == 0
                else float(np.linalg.svd(left_q.T @ right_q, compute_uv=False)[0])
            )
            correlation[(left, right)] = value
    return sensitivity, support, correlation


def _orthonormal_columns(matrix):
    if not np.any(matrix):
        return np.empty((matrix.shape[0], 0))
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > np.finfo(float).eps * max(matrix.shape) * singular[0]
    return left[:, keep]


def _local_identifiability(
    calculate,
    groups,
    sigma,
    training,
    *,
    max_points,
    group_labels=None,
    group_scales=None,
    group_step_descriptions=None,
    observation_covariance=None,
    prior_residuals=None,
):
    """Estimate local data and posterior Gauss--Newton information."""
    if max_points == 0:
        return {}
    import torch

    tensors = tuple(groups.values())
    labels = []
    scales = []
    step_descriptions = []
    for name, tensor in groups.items():
        declared = None if group_labels is None else group_labels.get(name)
        if declared is not None:
            if len(declared) != tensor.numel():
                raise ValueError(f"group_labels for {name} do not match its parameter count")
            labels.extend(f"{name}.{item}" for item in declared)
        else:
            labels.extend(
                [name]
                if tensor.numel() == 1
                else [f"{name}[{index}]" for index in range(tensor.numel())]
            )
        declared_scale = None if group_scales is None else group_scales.get(name)
        if declared_scale is None:
            scales.extend([1.0] * tensor.numel())
        else:
            values = np.asarray(declared_scale, dtype=np.float64)
            if values.ndim == 0:
                values = np.full(tensor.numel(), float(values))
            if values.shape != (tensor.numel(),):
                raise ValueError(f"group_scales for {name} do not match its parameter count")
            scales.extend(values.tolist())
        declared_description = (
            None if group_step_descriptions is None else group_step_descriptions.get(name)
        )
        if declared_description is None:
            step_descriptions.extend(["one raw parameter unit"] * tensor.numel())
        elif isinstance(declared_description, str):
            step_descriptions.extend([declared_description] * tensor.numel())
        else:
            if len(declared_description) != tensor.numel():
                raise ValueError(
                    f"group_step_descriptions for {name} do not match its parameter count"
                )
            step_descriptions.extend(str(item) for item in declared_description)
    available = np.flatnonzero(training)
    selected = available[
        np.linspace(0, len(available) - 1, min(max_points, len(available))).astype(int)
    ]
    sampling_factor = len(available) / len(selected)
    profile = calculate()
    rows = []
    for point in selected:
        gradients = torch.autograd.grad(
            profile[int(point)], tensors, retain_graph=True, allow_unused=True
        )
        flattened = []
        for tensor, gradient in zip(tensors, gradients):
            value = torch.zeros_like(tensor) if gradient is None else gradient
            flattened.append(value.reshape(-1))
        rows.append(torch.cat(flattened).detach().cpu().numpy())
    prior_rows = []
    if prior_residuals is not None:
        residual_vector = prior_residuals()
        for residual in residual_vector.reshape(-1):
            if not residual.requires_grad:
                continue
            gradients = torch.autograd.grad(residual, tensors, retain_graph=True, allow_unused=True)
            flattened = []
            for tensor, gradient in zip(tensors, gradients):
                value = torch.zeros_like(tensor) if gradient is None else gradient
                flattened.append(value.reshape(-1))
            prior_rows.append(torch.cat(flattened).detach().cpu().numpy())
    uncertainty = (
        {"covariance": observation_covariance[np.ix_(selected, selected)] / sampling_factor}
        if observation_covariance is not None
        else {"weights": sampling_factor / np.asarray(sigma)[selected] ** 2}
    )
    diagnostics = analyze_jacobian(
        np.asarray(rows),
        parameter_names=labels,
        parameter_scales=np.asarray(scales),
        prior_jacobian=np.asarray(prior_rows) if prior_rows else None,
        **uncertainty,
    )
    correlation = diagnostics.correlation
    off_diagonal = correlation.copy()
    np.fill_diagonal(off_diagonal, np.nan)
    maximum_correlation = float(np.nanmax(np.abs(off_diagonal))) if len(labels) > 1 else 0.0
    return {
        "parameter_names": list(labels),
        "characteristic_raw_steps": list(scales),
        "characteristic_step_descriptions": step_descriptions,
        "jacobian_sampled_points": len(selected),
        "jacobian_population_points": len(available),
        "jacobian_sampling_factor": sampling_factor,
        "sensitivity": diagnostics.sensitivity.tolist(),
        "rank": diagnostics.rank,
        "data_rank": diagnostics.rank,
        "prior_rank": diagnostics.prior_rank,
        "posterior_rank": diagnostics.posterior_rank,
        "parameter_count": len(labels),
        "condition_number": diagnostics.condition_number,
        "data_condition_number": diagnostics.condition_number,
        "posterior_condition_number": diagnostics.posterior_condition_number,
        "covariance_is_identifiable": diagnostics.covariance_is_identifiable,
        "data_covariance_is_identifiable": diagnostics.covariance_is_identifiable,
        "posterior_covariance_is_identifiable": (diagnostics.posterior_covariance_is_identifiable),
        "standard_errors_in_characteristic_steps": [
            float(value) if np.isfinite(value) else None
            for value in diagnostics.standard_errors_scaled
        ],
        "standard_errors_in_raw_parameters": [
            float(value) if np.isfinite(value) else None
            for value in diagnostics.standard_errors_physical
        ],
        "maximum_absolute_correlation": maximum_correlation,
        "correlation": correlation.tolist(),
        "posterior_correlation": diagnostics.posterior_correlation.tolist(),
        "null_directions": [
            {
                "coefficients": {
                    name: float(coefficient)
                    for name, coefficient in zip(labels, vector)
                    if abs(coefficient) > 1e-8
                }
            }
            for vector in diagnostics.null_space_vectors
        ],
        "warning": (
            "Subsampled local Gaussian approximation conditional on the supplied noise, "
            "forward and prior models; inspect bootstrap coverage before uncertainty use."
        ),
    }


def _weighted_squared_norm(values, selected, sigma, observation_covariance=None):
    indices = np.flatnonzero(selected)
    vector = np.asarray(values, dtype=np.float64)[indices]
    if observation_covariance is None:
        return float(np.sum((vector / np.asarray(sigma)[indices]) ** 2))
    covariance = observation_covariance[np.ix_(indices, indices)]
    return float(vector @ np.linalg.solve(covariance, vector))


def _informative_regions(coordinate, standardized_residual, *, count):
    order = np.argsort(np.abs(standardized_residual))[::-1]
    chosen = []
    minimum_separation = 5 * float(np.median(np.diff(coordinate)))
    for index in order:
        position = float(coordinate[index])
        if all(abs(position - item["coordinate"]) >= minimum_separation for item in chosen):
            chosen.append(
                {
                    "coordinate": position,
                    "standardized_residual": float(standardized_residual[index]),
                }
            )
        if len(chosen) == count:
            break
    return tuple(chosen)


def _maximum_cross_group_correlation(diagnostics, left_prefix, right_prefix):
    if not diagnostics:
        return 0.0
    names = diagnostics["parameter_names"]
    left = [index for index, name in enumerate(names) if name.startswith(left_prefix)]
    right = [index for index, name in enumerate(names) if name.startswith(right_prefix)]
    if not left or not right:
        return 0.0
    correlation = np.asarray(diagnostics["correlation"], dtype=np.float64)
    return float(np.max(np.abs(correlation[np.ix_(left, right)])))


def _svg_profile(dataset, candidate, width=900, height=320):
    selected = np.linspace(
        0, len(dataset.coordinate) - 1, min(1200, len(dataset.coordinate))
    ).astype(int)
    x = dataset.coordinate[selected]
    curves = [
        dataset.intensity[selected],
        candidate.calculated[selected],
        candidate.residual[selected],
    ]
    x_scaled = (x - x.min()) / max(float(np.ptp(x)), 1e-12) * (width - 60) + 45
    y_min = min(float(np.min(curve)) for curve in curves)
    y_max = max(float(np.max(curve)) for curve in curves)

    def points(values):
        y = height - 30 - (values - y_min) / max(y_max - y_min, 1e-12) * (height - 55)
        return " ".join(f"{left:.1f},{top:.1f}" for left, top in zip(x_scaled, y))

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="profile fit">'
        f'<polyline fill="none" stroke="#222" stroke-width="1" points="{points(curves[0])}"/>'
        f'<polyline fill="none" stroke="#0072b2" stroke-width="1" points="{points(curves[1])}"/>'
        f'<polyline fill="none" stroke="#d55e00" stroke-width="1" points="{points(curves[2])}"/>'
        "</svg>"
    )


def _render_html(result: SessionResult) -> str:
    sections = []
    for candidate in result.candidates:
        rows = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td><code>{html.escape(json.dumps(value))}</code></td></tr>"
            for key, value in candidate.physical_parameters.items()
        )
        warnings = "".join(f"<li>{html.escape(item)}</li>" for item in candidate.warnings)
        regions = "".join(
            f"<li>{item['coordinate']:.4f}: standardized residual {item['standardized_residual']:+.3f}</li>"
            for item in candidate.informative_regions
        )
        identifiability = html.escape(json.dumps(candidate.identifiability))
        convergence = html.escape(json.dumps(candidate.convergence or {}))
        sections.append(
            f"<section><h2>{html.escape(candidate.name)}</h2>"
            f"<p>Rwp={candidate.r_wp:.5f}; chi²={candidate.chi_squared:.3f}; "
            f"held-out Rwp={candidate.held_out_r_wp if candidate.held_out_r_wp is not None else 'n/a'}</p>"
            f"{_svg_profile(result.dataset, candidate)}<p>Black: observed; blue: calculated; orange: residual.</p>"
            f"<p><strong>Recommended next action:</strong> {html.escape(candidate.recommendation)}</p>"
            f"<ul>{warnings}</ul><h3>Largest unexplained regions</h3><ol>{regions}</ol>"
            f"<h3>Optimization and validation</h3><pre>{convergence}</pre>"
            f"<h3>Local identifiability</h3><pre>{identifiability}</pre>"
            f"<table>{rows}</table></section>"
        )
    pairwise = "".join(
        f"<li>{html.escape(name)}: expected Δχ²={value:.3f}</li>"
        for name, value in result.pairwise_discrimination.items()
    )
    source = html.escape(result.dataset.source or "in-memory dataset")
    digest = html.escape(result.dataset.source_sha256 or "not available")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>BraggCalculator diagnostic report</title>
<style>body{{font-family:system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse}}th,td{{text-align:left;border-bottom:1px solid #ddd;padding:.35rem}}svg{{width:100%;border:1px solid #ddd}}</style></head>
<body><h1>Diffraction diagnostic report</h1><p><strong>Conclusion:</strong> {html.escape(result.conclusion)}</p>
<p>Dataset: {source}<br>SHA-256: <code>{digest}</code></p><h2>Candidate ranking</h2><ol>{"".join(f"<li>{html.escape(name)}</li>" for name in result.ranking)}</ol>
<h2>Pairwise discrimination</h2><ul>{pairwise or "<li>Not applicable</li>"}</ul>{"".join(sections)}</body></html>"""
