"""End-to-end candidate-guided experimental refinement sessions."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .backends import TorchBackend
from .core import BraggCalculator
from .dataset import DiffractionDataset
from .experimental_profile import caglioti_fwhm, render_pseudo_voigt
from .io import to_pmg_structure
from .optimization import OptimizationStage, staged_adam
from .sensitivity import analyze_jacobian


@dataclass(frozen=True)
class RefinementPolicy:
    """Declared release policy and restraint strengths for one refinement."""

    background_degree: int = 4
    refine_lattice: bool = True
    refine_coordinates: bool = False
    coordinate_restraint: float = 10.0
    holdout_stride: int = 10
    restarts: int = 1
    diagnostic_points: int = 48
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

    @classmethod
    def quick(cls, *, refine_coordinates: bool = False) -> "RefinementPolicy":
        active_joint = ("scale", "background", "zero_shift", "profile", "lattice")
        if refine_coordinates:
            active_joint += ("coordinates",)
        return cls(
            refine_coordinates=refine_coordinates,
            stages=(
                OptimizationStage("scale/background", ("scale", "background"), 40, 0.04),
                OptimizationStage("calibration/profile", ("zero_shift", "profile", "lattice"), 60, 0.025),
                OptimizationStage("joint", active_joint, 100, 0.01),
            ),
        )

    @classmethod
    def cautious(cls, *, refine_coordinates: bool = False) -> "RefinementPolicy":
        active_joint = ("scale", "background", "zero_shift", "profile", "lattice")
        stages = [
            OptimizationStage("scale/background", ("scale", "background"), 120, 0.03),
            OptimizationStage("calibration/profile", ("zero_shift", "profile", "lattice"), 180, 0.015),
        ]
        if refine_coordinates:
            stages.append(OptimizationStage("coordinates", ("coordinates",), 150, 0.006))
            active_joint += ("coordinates",)
        stages.append(OptimizationStage("joint", active_joint, 250, 0.005))
        return cls(refine_coordinates=refine_coordinates, stages=tuple(stages))


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
        if dataset.domain != "two_theta":
            raise NotImplementedError("experimental sessions currently require two_theta data")
        self.dataset = dataset
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
    ):
        import torch

        policy = RefinementPolicy.quick() if policy is None else policy
        calculator = self._calculator(self.structures[index])
        raw_components = self.dataset.metadata.get(
            "wavelength_components", [(self.dataset.wavelength, 1.0)]
        )
        component_weights = np.asarray([float(item[1]) for item in raw_components])
        if np.any(component_weights <= 0) or not np.all(np.isfinite(component_weights)):
            raise ValueError("wavelength component weights must be positive and finite")
        component_weights = component_weights / component_weights.sum()
        component_calculators = []
        for wavelength, _ in raw_components:
            wavelength = float(wavelength)
            if np.isclose(wavelength, self.dataset.wavelength):
                component_calculators.append(calculator)
            else:
                component_calculators.append(
                    BraggCalculator(
                        mode=self.dataset.radiation,
                        wavelength=wavelength,
                        two_theta_range=(
                            float(self.dataset.coordinate[0]),
                            float(self.dataset.coordinate[-1]),
                        ),
                        two_theta_step=self.dataset.step,
                        backend=TorchBackend(device=self.device),
                        primitive=False,
                    ).load(self.structures[index])
                )
        coordinate_model = calculator.symmetry_coordinate_parameterization()
        component_bases = [item.tensor_parameters() for item in component_calculators]
        base = component_bases[0]
        grid = torch.as_tensor(self.dataset.coordinate, dtype=torch.float64, device=self.device)
        observed = torch.as_tensor(self.dataset.intensity, dtype=torch.float64, device=self.device)
        sigma = torch.as_tensor(self.dataset.sigma, dtype=torch.float64, device=self.device)
        selected = self.dataset.mask.copy()
        held_out = np.zeros(len(selected), dtype=bool)
        if policy.holdout_stride > 1:
            held_out[np.flatnonzero(selected)[:: policy.holdout_stride]] = True
        training = selected & ~held_out
        training_tensor = torch.as_tensor(training, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            component_areas = [
                float(torch.sum(item.iq(domain="two_theta")[1]).cpu())
                for item in component_calculators
            ]
            background0 = max(float(np.percentile(self.dataset.intensity[selected], 10)), 1e-6)
            signal_area = max(
                float(np.trapezoid(np.maximum(self.dataset.intensity - background0, 0.0), self.dataset.coordinate)),
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
        profile = torch.zeros(4, dtype=torch.float64, device=self.device, requires_grad=True)
        lattice = torch.zeros((), dtype=torch.float64, device=self.device, requires_grad=True)
        coordinates = coordinate_model.initial_values(calculator.backend, requires_grad=True)
        if restart_index:
            generator = np.random.default_rng(1729 + restart_index)
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
                        generator.normal(0.0, 0.05, 4),
                        dtype=torch.float64,
                        device=self.device,
                    )
                )
                lattice.copy_(
                    torch.tensor(
                        generator.normal(0.0, 0.05), dtype=torch.float64, device=self.device
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
        groups = {
            "scale": scale,
            "background": background,
            "zero_shift": zero_shift,
            "profile": profile,
        }
        if policy.refine_lattice:
            groups["lattice"] = lattice
        if policy.refine_coordinates and coordinate_model.independent_count:
            groups["coordinates"] = coordinates

        normalized_x = torch.linspace(
            -1.0, 1.0, len(grid), dtype=torch.float64, device=self.device
        )
        background_basis = torch.stack(
            [normalized_x**degree for degree in range(policy.background_degree + 1)], dim=1
        )

        def calculate():
            peaks = torch.zeros_like(grid)
            for component_calculator, component_base, component_weight in zip(
                component_calculators, component_bases, component_weights
            ):
                structural = dict(component_base)
                structural["lattice"] = component_base["lattice"] * torch.exp(0.01 * lattice)
                if policy.refine_coordinates and coordinate_model.independent_count:
                    structural["frac_coords"] = coordinate_model.expand(
                        coordinates, calculator.backend
                    )
                peak_centers, peak_areas = component_calculator.iq(
                    domain="two_theta", parameters=structural
                )
                peak_centers = peak_centers + self.dataset.step * zero_shift
                peak_areas = peak_areas * scale0 * torch.exp(scale) * component_weight
                radians = torch.deg2rad(peak_centers)
                u = 0.0025 * torch.exp(profile[0])
                v = 1e-6 * torch.exp(profile[1])
                w = 0.0064 * torch.exp(profile[2])
                widths = caglioti_fwhm(radians, u, v, w, calculator.backend)
                eta = torch.sigmoid(profile[3])
                peaks = peaks + render_pseudo_voigt(
                    grid,
                    peak_centers,
                    peak_areas,
                    widths,
                    eta,
                    component_calculator.backend,
                )
            return peaks + background_basis @ background

        def objective():
            calculated = calculate()
            standardized = (calculated[training_tensor] - observed[training_tensor]) / sigma[
                training_tensor
            ]
            loss = torch.mean(standardized**2)
            negative_background = torch.relu(-(background_basis @ background))
            loss = loss + 0.01 * torch.mean(negative_background**2)
            if policy.refine_coordinates and coordinate_model.independent_count:
                loss = loss + policy.coordinate_restraint * torch.mean(coordinates**2)
            return loss

        stages = policy.stages or RefinementPolicy.cautious(
            refine_coordinates=policy.refine_coordinates
        ).stages
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
        trace = staged_adam(objective, groups, stages)
        calculated = calculate().detach().cpu().numpy()
        residual = self.dataset.intensity - calculated
        weights = self.dataset.weights
        denominator = np.sum(weights[selected] * self.dataset.intensity[selected] ** 2)
        r_wp = float(np.sqrt(np.sum(weights[selected] * residual[selected] ** 2) / denominator))
        chi_squared = float(np.mean((residual[training] / self.dataset.sigma[training]) ** 2))
        held_out_r_wp = None
        if np.any(held_out):
            held_denominator = np.sum(weights[held_out] * self.dataset.intensity[held_out] ** 2)
            held_out_r_wp = float(
                np.sqrt(np.sum(weights[held_out] * residual[held_out] ** 2) / held_denominator)
            )
        physical = {
            "scale": float(scale0 * torch.exp(scale).detach().cpu()),
            "background_coefficients": background.detach().cpu().numpy().tolist(),
            "zero_shift": float((self.dataset.step * zero_shift).detach().cpu()),
            "caglioti_u": float((0.0025 * torch.exp(profile[0])).detach().cpu()),
            "caglioti_v": float((1e-6 * torch.exp(profile[1])).detach().cpu()),
            "caglioti_w": float((0.0064 * torch.exp(profile[2])).detach().cpu()),
            "eta": float(torch.sigmoid(profile[3]).detach().cpu()),
            "lattice_scale": float(torch.exp(0.01 * lattice).detach().cpu()),
            "lattice": (base["lattice"] * torch.exp(0.01 * lattice)).detach().cpu().numpy().tolist(),
            "coordinate_displacements": coordinates.detach().cpu().numpy().tolist(),
        }
        warnings = []
        if r_wp > 0.15:
            warnings.append("Large profile residual: the instrument/background model is incomplete.")
        if policy.refine_coordinates:
            warnings.append("Coordinate uncertainties are not yet calibrated for experimental use.")
        if r_wp > 0.15:
            recommendation = (
                "Improve the wavelength, instrument-profile, background, or phase model before "
                "interpreting structural parameters."
            )
        elif not policy.refine_coordinates:
            recommendation = (
                "Inspect parameter sensitivity and correlations before releasing structural coordinates."
            )
        else:
            recommendation = "Validate the refined coordinates across restarts and held-out regions."
        regions = _informative_regions(
            self.dataset.coordinate, residual / self.dataset.sigma, count=5
        )
        identifiability = _local_identifiability(
            calculate,
            groups,
            self.dataset.sigma,
            training,
            max_points=policy.diagnostic_points,
        )
        if identifiability and not identifiability["covariance_is_identifiable"]:
            warnings.append("The released parameter set is locally rank deficient.")
        if identifiability and identifiability["maximum_absolute_correlation"] > 0.98:
            warnings.append("At least one released parameter pair is extremely correlated.")
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
                "wavelength_components": [list(item) for item in raw_components],
                "policy": {
                    "background_degree": policy.background_degree,
                    "refine_lattice": policy.refine_lattice,
                    "refine_coordinates": policy.refine_coordinates,
                    "coordinate_restraint": policy.coordinate_restraint,
                    "holdout_stride": policy.holdout_stride,
                    "diagnostic_points": policy.diagnostic_points,
                },
                "restart_index": restart_index,
            },
        )

    def run(self, policy: RefinementPolicy | None = None) -> SessionResult:
        policy = RefinementPolicy.quick() if policy is None else policy
        candidates = []
        for index in range(len(self.structures)):
            attempts = tuple(
                self.refine_candidate(index, policy=policy, restart_index=restart)
                for restart in range(policy.restarts)
            )
            best = min(attempts, key=lambda item: item.r_wp)
            restart_rwp = [item.r_wp for item in attempts]
            extra_warnings = list(best.warnings)
            if len(attempts) > 1 and np.ptp(restart_rwp) > 0.02:
                extra_warnings.append("Refinement is sensitive to starting values across restarts.")
            provenance = dict(best.provenance)
            provenance["restart_rwp"] = restart_rwp
            candidates.append(
                replace(best, warnings=tuple(extra_warnings), provenance=provenance)
            )
        candidates = tuple(candidates)
        ranking = tuple(result.name for result in sorted(candidates, key=lambda item: item.r_wp))
        pairwise = {}
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                difference = candidates[left].calculated - candidates[right].calculated
                score = float(np.sum((difference[self.dataset.mask] / self.dataset.sigma[self.dataset.mask]) ** 2))
                pairwise[f"{candidates[left].name} vs {candidates[right].name}"] = score
        if len(candidates) == 1:
            conclusion = f"Refined {candidates[0].name}; candidate discrimination was not requested."
        elif pairwise and max(pairwise.values()) < 9.0:
            conclusion = "The supplied experiment does not discriminate the refined candidates."
        else:
            conclusion = f"{ranking[0]} has the lowest Rwp; inspect robustness and residual evidence."
        return SessionResult(self.dataset, candidates, ranking, pairwise, conclusion)

    def write_html(self, result: SessionResult, path) -> Path:
        output = Path(path)
        output.write_text(_render_html(result), encoding="utf-8")
        return output


def _local_identifiability(calculate, groups, sigma, training, *, max_points):
    """Estimate a subsampled local Gauss-Newton matrix for released raw parameters."""
    if max_points == 0:
        return {}
    import torch

    tensors = tuple(groups.values())
    labels = []
    for name, tensor in groups.items():
        labels.extend(
            [name]
            if tensor.numel() == 1
            else [f"{name}[{index}]" for index in range(tensor.numel())]
        )
    selected = np.flatnonzero(training)
    selected = selected[
        np.linspace(0, len(selected) - 1, min(max_points, len(selected))).astype(int)
    ]
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
    diagnostics = analyze_jacobian(
        np.asarray(rows),
        weights=1.0 / np.asarray(sigma)[selected] ** 2,
        parameter_names=labels,
    )
    correlation = diagnostics.correlation
    off_diagonal = correlation.copy()
    np.fill_diagonal(off_diagonal, np.nan)
    maximum_correlation = (
        float(np.nanmax(np.abs(off_diagonal))) if len(labels) > 1 else 0.0
    )
    return {
        "parameter_names": list(labels),
        "sensitivity": diagnostics.sensitivity.tolist(),
        "rank": diagnostics.rank,
        "parameter_count": len(labels),
        "condition_number": diagnostics.condition_number,
        "covariance_is_identifiable": diagnostics.covariance_is_identifiable,
        "maximum_absolute_correlation": maximum_correlation,
        "correlation": correlation.tolist(),
        "warning": "Subsampled local raw-parameter approximation; not calibrated uncertainty.",
    }


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


def _svg_profile(dataset, candidate, width=900, height=320):
    selected = np.linspace(0, len(dataset.coordinate) - 1, min(1200, len(dataset.coordinate))).astype(int)
    x = dataset.coordinate[selected]
    curves = [dataset.intensity[selected], candidate.calculated[selected], candidate.residual[selected]]
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
        sections.append(
            f"<section><h2>{html.escape(candidate.name)}</h2>"
            f"<p>Rwp={candidate.r_wp:.5f}; chi²={candidate.chi_squared:.3f}; "
            f"held-out Rwp={candidate.held_out_r_wp if candidate.held_out_r_wp is not None else 'n/a'}</p>"
            f"{_svg_profile(result.dataset, candidate)}<p>Black: observed; blue: calculated; orange: residual.</p>"
            f"<p><strong>Recommended next action:</strong> {html.escape(candidate.recommendation)}</p>"
            f"<ul>{warnings}</ul><h3>Largest unexplained regions</h3><ol>{regions}</ol>"
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
<p>Dataset: {source}<br>SHA-256: <code>{digest}</code></p><h2>Candidate ranking</h2><ol>{''.join(f'<li>{html.escape(name)}</li>' for name in result.ranking)}</ol>
<h2>Pairwise discrimination</h2><ul>{pairwise or '<li>Not applicable</li>'}</ul>{''.join(sections)}</body></html>"""
