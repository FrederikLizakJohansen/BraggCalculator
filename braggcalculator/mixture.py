"""Joint profile-area fraction refinement for physical phase mixtures."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
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
    thompson_cox_hastings,
)
from .io import to_pmg_structure
from .optimization import OptimizationStage, staged_adam
from .parameters import SimplexPhaseFractionParameterization
from .session import (
    _informative_regions,
    _local_identifiability,
    _wavelength_components,
    _weighted_squared_norm,
)


@dataclass(frozen=True)
class PhaseMixturePolicy:
    """Release policy for fixed-structure multi-phase profile refinement."""

    background_degree: int = 2
    initial_fractions: tuple[float, ...] | None = None
    refine_profile: bool = True
    refine_zero_shift: bool = False
    profile_model: str = "tch"
    axial_asymmetry: bool = True
    holdout_stride: int = 10
    diagnostic_points: int = 48
    stages: tuple[OptimizationStage, ...] | None = None

    def __post_init__(self):
        if self.background_degree < 0:
            raise ValueError("background_degree must be non-negative")
        if self.profile_model not in {"legacy", "tch"}:
            raise ValueError("profile_model must be 'legacy' or 'tch'")
        if self.holdout_stride < 2:
            raise ValueError("holdout_stride must be at least two")
        if self.diagnostic_points < 0:
            raise ValueError("diagnostic_points must be non-negative")


@dataclass(frozen=True)
class PhaseMixtureResult:
    phase_names: tuple[str, ...]
    phase_fractions: dict[str, float]
    calculated: np.ndarray
    residual: np.ndarray
    component_profiles: dict[str, np.ndarray]
    r_wp: float
    chi_squared: float
    held_out_r_wp: float | None
    loss_history: np.ndarray
    stage_history: tuple[str, ...]
    identifiability: dict[str, Any]
    phase_detectability: dict[str, float]
    informative_regions: tuple[dict[str, float], ...]
    warnings: tuple[str, ...]
    provenance: dict[str, Any]


class PhaseMixtureSession:
    """Refine several fixed structures as phases in one observed pattern."""

    def __init__(self, dataset: DiffractionDataset, phases, *, names=None, device="cpu"):
        self.input_dataset = dataset
        self.dataset = dataset.convert_domain("two_theta")
        self.structures = tuple(to_pmg_structure(phase) for phase in phases)
        if len(self.structures) < 2:
            raise ValueError("a phase mixture requires at least two structures")
        self.names = (
            tuple(str(name) for name in names)
            if names is not None
            else tuple(f"phase_{index + 1}" for index in range(len(self.structures)))
        )
        if len(self.names) != len(self.structures) or len(set(self.names)) != len(self.names):
            raise ValueError("phase names must be unique and match the phase count")
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

    def run(self, policy: PhaseMixturePolicy | None = None):
        import torch

        policy = PhaseMixturePolicy() if policy is None else policy
        calculators = tuple(self._calculator(structure) for structure in self.structures)
        components = _wavelength_components(self.dataset.metadata, self.dataset.wavelength)
        wavelengths = [item["wavelength_angstrom"] for item in components]
        phase_lines = tuple(
            calculator.line_components(wavelengths, domain="two_theta")
            for calculator in calculators
        )
        fraction_model = SimplexPhaseFractionParameterization.create(
            self.names, policy.initial_fractions
        )
        grid = torch.as_tensor(self.dataset.coordinate, dtype=torch.float64, device=self.device)
        observed = torch.as_tensor(self.dataset.intensity, dtype=torch.float64, device=self.device)
        sigma = torch.as_tensor(self.dataset.sigma, dtype=torch.float64, device=self.device)
        selected = self.dataset.mask.copy()
        held_out = np.zeros(len(selected), dtype=bool)
        held_out[np.flatnonzero(selected)[:: policy.holdout_stride]] = True
        training = selected & ~held_out
        training_tensor = torch.as_tensor(training, dtype=torch.bool, device=self.device)
        training_indices = np.flatnonzero(training)
        covariance_cholesky = None
        if self.dataset.observation_covariance is not None:
            training_covariance = self.dataset.observation_covariance[
                np.ix_(training_indices, training_indices)
            ]
            covariance_cholesky = torch.linalg.cholesky(
                torch.as_tensor(training_covariance, dtype=torch.float64, device=self.device)
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
        scale = torch.zeros((), dtype=torch.float64, device=self.device, requires_grad=True)
        background = torch.zeros(
            policy.background_degree + 1,
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        with torch.no_grad():
            background[0] = background0
        phase_values = fraction_model.initial_values(calculators[0].backend, requires_grad=True)
        zero_shift = torch.zeros((), dtype=torch.float64, device=self.device, requires_grad=True)
        profile_size = 4 if policy.profile_model == "legacy" else 6
        profile = torch.zeros(
            profile_size, dtype=torch.float64, device=self.device, requires_grad=True
        )
        normalized_x = torch.linspace(-1.0, 1.0, len(grid), dtype=torch.float64, device=self.device)
        background_basis = torch.stack(
            [normalized_x**degree for degree in range(policy.background_degree + 1)], dim=1
        )

        def render_phase(lines):
            pattern = torch.zeros_like(grid)
            for (centers, areas), component in zip(lines, components):
                centers = centers + self.dataset.step * zero_shift
                radians = torch.deg2rad(centers)
                if policy.profile_model == "legacy":
                    widths = caglioti_fwhm(
                        radians,
                        0.0025 * torch.exp(profile[0]),
                        1e-6 * torch.exp(profile[1]),
                        0.0064 * torch.exp(profile[2]),
                        calculators[0].backend,
                    )
                    rendered = render_pseudo_voigt(
                        grid,
                        centers,
                        areas,
                        widths,
                        torch.sigmoid(profile[3]),
                        calculators[0].backend,
                    )
                else:
                    widths, eta = thompson_cox_hastings(
                        radians,
                        0.0025 * torch.exp(profile[0]),
                        0.001 * torch.sinh(profile[1]),
                        0.0036 * torch.exp(profile[2]),
                        0.01 * torch.exp(profile[3]),
                        0.01 * torch.exp(profile[4]),
                        calculators[0].backend,
                        extra_lorentzian=emission_lorentzian_fwhm(
                            radians,
                            component["wavelength_angstrom"],
                            component["lorentzian_fwhm_angstrom"],
                            calculators[0].backend,
                        ),
                    )
                    asymmetry = 0.05 * torch.exp(profile[5]) if policy.axial_asymmetry else 0.0
                    low, high = axial_divergence_widths(
                        widths, radians, asymmetry, calculators[0].backend
                    )
                    rendered = render_split_pseudo_voigt(
                        grid, centers, areas, low, high, eta, calculators[0].backend
                    )
                pattern = pattern + component["normalized_weight"] * rendered
            area = torch.sum(pattern) * self.dataset.step
            return pattern / torch.clamp(area, min=1e-30)

        def calculate(*, components_out=False):
            fractions = fraction_model.expand(phase_values, calculators[0].backend)
            unit_profiles = [render_phase(lines) for lines in phase_lines]
            signal_scale = signal_area * torch.exp(scale)
            contributions = [
                signal_scale * fraction * unit for fraction, unit in zip(fractions, unit_profiles)
            ]
            calculated = torch.stack(contributions).sum(dim=0) + background_basis @ background
            return (calculated, contributions) if components_out else calculated

        def objective():
            calculated = calculate()
            standardized = whiten_training(calculated - observed)
            negative_background = torch.relu(-(background_basis @ background))
            return torch.mean(standardized**2) + 0.01 * torch.mean(negative_background**2)

        groups = {
            "scale": scale,
            "background": background,
            "phase_fractions": phase_values,
        }
        if policy.refine_profile:
            groups["profile"] = profile
        if policy.refine_zero_shift:
            groups["zero_shift"] = zero_shift
        active_joint = tuple(groups)
        stages = policy.stages or (
            OptimizationStage("scale/background", ("scale", "background"), 80, 0.03),
            OptimizationStage("phase fractions", ("phase_fractions",), 180, 0.025),
            OptimizationStage("joint", active_joint, 220, 0.008),
        )
        stages = tuple(
            OptimizationStage(
                stage.name,
                tuple(name for name in stage.active if name in groups),
                stage.steps,
                stage.learning_rate,
            )
            for stage in stages
            if any(name in groups for name in stage.active)
        )
        released_names = {name for stage in stages for name in stage.active}
        trace = staged_adam(objective, groups, stages)
        calculated_tensor, contribution_tensors = calculate(components_out=True)
        calculated = calculated_tensor.detach().cpu().numpy()
        contributions = {
            name: value.detach().cpu().numpy()
            for name, value in zip(self.names, contribution_tensors)
        }
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
        held_denominator = np.sum(weights[held_out] * self.dataset.intensity[held_out] ** 2)
        held_out_r_wp = (
            float(np.sqrt(np.sum(weights[held_out] * residual[held_out] ** 2) / held_denominator))
            if np.any(held_out)
            else None
        )
        released = {name: value for name, value in groups.items() if name in released_names}
        uncertainty_scales = {
            "scale": 0.01,
            "background": float(np.median(self.dataset.sigma)),
            "phase_fractions": 0.1,
            "profile": 0.1,
            "zero_shift": 1.0,
        }
        uncertainty_descriptions = {
            "scale": "0.01 log scale (about 1%)",
            "background": "one median marginal observation sigma in intensity units",
            "phase_fractions": "0.1 phase-fraction log ratio",
            "profile": "0.1 profile raw/log coefficient",
            "zero_shift": f"one profile bin ({self.dataset.step:.6g} degrees 2-theta)",
        }
        identifiability = _local_identifiability(
            calculate,
            released,
            self.dataset.sigma,
            training,
            max_points=policy.diagnostic_points,
            observation_covariance=self.dataset.observation_covariance,
            group_scales=uncertainty_scales,
            group_step_descriptions=uncertainty_descriptions,
            group_labels={"phase_fractions": fraction_model.labels},
        )
        fractions = fraction_model.physical(phase_values.detach().cpu().numpy())
        detectability = {
            name: float(
                np.sqrt(
                    _weighted_squared_norm(
                        contribution,
                        self.dataset.mask,
                        self.dataset.sigma,
                        self.dataset.observation_covariance,
                    )
                )
            )
            for name, contribution in contributions.items()
        }
        warnings = [
            "Phase fractions are integrated profile-area fractions over the fitted range, "
            "not quantitative mass fractions."
        ]
        for name, score in detectability.items():
            if score < 3.0:
                warnings.append(
                    f"Phase {name} is below the approximate 3-sigma profile detectability "
                    "threshold; its fraction is not supported."
                )
        if identifiability and not identifiability["data_covariance_is_identifiable"]:
            warnings.append(
                "The released mixture data Jacobian is locally rank deficient; inspect "
                "the null parameter combinations."
            )
        if identifiability and identifiability["maximum_absolute_correlation"] > 0.98:
            warnings.append("At least one mixture parameter pair is extremely correlated.")
        return PhaseMixtureResult(
            phase_names=self.names,
            phase_fractions=fractions,
            calculated=calculated,
            residual=residual,
            component_profiles=contributions,
            r_wp=r_wp,
            chi_squared=chi_squared,
            held_out_r_wp=held_out_r_wp,
            loss_history=trace.loss,
            stage_history=trace.stage,
            identifiability=identifiability,
            phase_detectability=detectability,
            informative_regions=_informative_regions(
                self.input_dataset.coordinate, residual / self.dataset.sigma, count=5
            ),
            warnings=tuple(warnings),
            provenance={
                "fraction_definition": "integrated profile area over fitted range",
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
                "phase_names": list(self.names),
                "initial_fractions": fraction_model.initial_fractions.tolist(),
                "wavelength_components": [dict(item) for item in components],
                "policy": {
                    "background_degree": policy.background_degree,
                    "refine_profile": policy.refine_profile,
                    "refine_zero_shift": policy.refine_zero_shift,
                    "profile_model": policy.profile_model,
                    "axial_asymmetry": policy.axial_asymmetry,
                    "diagnostic_points": policy.diagnostic_points,
                    "released_parameter_groups": sorted(released_names),
                },
            },
        )

    def write_html(self, result: PhaseMixtureResult, path):
        output = Path(path)
        fraction_rows = "".join(
            f"<tr><th>{html.escape(name)}</th><td>{fraction:.6f}</td>"
            f"<td>{result.phase_detectability[name]:.3f}</td></tr>"
            for name, fraction in result.phase_fractions.items()
        )
        warnings = "".join(f"<li>{html.escape(item)}</li>" for item in result.warnings)
        provenance = html.escape(json.dumps(result.provenance, indent=2))
        output.write_text(
            f"""<!doctype html><html><head><meta charset="utf-8"><title>Phase mixture report</title>
<style>body{{font-family:system-ui;max-width:950px;margin:2rem auto}}table{{border-collapse:collapse}}
th,td{{padding:.4rem;border-bottom:1px solid #ddd}}pre{{background:#f4f4f4;padding:1rem}}</style></head>
<body><h1>Phase mixture diagnostic</h1><p>Rwp={result.r_wp:.6f}; chi2={result.chi_squared:.6f}</p>
<table><tr><th>Phase</th><th>Profile-area fraction</th><th>Detectability</th></tr>{fraction_rows}</table>
<h2>Warnings</h2><ul>{warnings}</ul><h2>Provenance</h2><pre>{provenance}</pre></body></html>""",
            encoding="utf-8",
        )
        return output
