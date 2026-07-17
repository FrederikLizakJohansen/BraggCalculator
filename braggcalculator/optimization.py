"""Guarded optimization utilities for differentiable refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class OptimizationStage:
    """One optimizer stage and the parameter groups released within it."""

    name: str
    active: tuple[str, ...]
    steps: int
    learning_rate: float
    optimizer: str = "adam"
    width_multiplier: float = 1.0

    def __post_init__(self):
        if not self.name or not self.active:
            raise ValueError("stage name and active groups must be non-empty")
        if self.steps <= 0:
            raise ValueError("stage steps must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("stage learning_rate must be positive and finite")
        if self.optimizer not in {"adam", "lbfgs"}:
            raise ValueError("stage optimizer must be 'adam' or 'lbfgs'")
        if not np.isfinite(self.width_multiplier) or self.width_multiplier <= 0:
            raise ValueError("stage width_multiplier must be positive and finite")


@dataclass(frozen=True)
class StageOutcome:
    """Acceptance and convergence evidence for one optimization stage."""

    name: str
    optimizer: str
    accepted: bool
    reason: str
    training_before: float
    training_after: float
    validation_before: float | None
    validation_after: float | None
    gradient_norm: float
    width_multiplier: float


@dataclass(frozen=True)
class StagedOptimizationResult:
    """Trace, acceptance evidence, and final values from staged optimization."""

    loss: np.ndarray
    stage: tuple[str, ...]
    step_in_stage: np.ndarray
    final_values: dict[str, np.ndarray]
    stage_outcomes: tuple[StageOutcome, ...] = ()
    convergence_classification: str = "maximum_steps"
    final_gradient_norm: float = np.nan
    relative_loss_change: float = np.nan


@dataclass(frozen=True)
class GaussNewtonResult:
    """Trace and trust-region evidence from damped Gauss--Newton."""

    values: np.ndarray
    loss: np.ndarray
    damping: np.ndarray
    trust_radius: np.ndarray
    accepted: np.ndarray
    gain_ratio: np.ndarray
    gradient_norm: float
    convergence_classification: str


@dataclass(frozen=True)
class ReleaseDecision:
    """Evidence for accepting or rejecting one parameter group."""

    group: str
    accepted: bool
    sensitivity: float
    residual_support: float
    maximum_correlation: float
    reason: str


def staged_optimize(
    objective,
    parameter_groups,
    stages,
    *,
    validation_objective=None,
    before_stage=None,
    validation_tolerance: float = 0.0,
    gradient_tolerance: float = 1e-6,
    relative_loss_tolerance: float = 1e-9,
) -> StagedOptimizationResult:
    """Run declared Adam/L-BFGS stages with optional validation rollback.

    ``before_stage`` can update forward-model state, such as a continuation
    peak-width multiplier. If a validation objective is supplied, every stage
    is accepted only when validation does not worsen beyond the relative
    ``validation_tolerance``. Rejected stages are restored exactly.
    """
    torch = _require_torch("staged_optimize")
    groups = dict(parameter_groups)
    declared_stages = tuple(stages)
    _validate_groups_and_stages(torch, groups, declared_stages)
    if validation_tolerance < 0 or not np.isfinite(validation_tolerance):
        raise ValueError("validation_tolerance must be non-negative and finite")

    losses: list[float] = []
    stage_names: list[str] = []
    stage_steps: list[int] = []
    outcomes: list[StageOutcome] = []

    for stage in declared_stages:
        if before_stage is not None:
            before_stage(stage)
        snapshots = {name: value.detach().clone() for name, value in groups.items()}
        training_before = _objective_value(torch, objective, stage.name)
        validation_before = (
            _objective_value(torch, validation_objective, stage.name)
            if validation_objective is not None
            else None
        )
        active = [groups[name] for name in stage.active]

        if stage.optimizer == "adam":
            optimizer = torch.optim.Adam(active, lr=stage.learning_rate)
            for step in range(stage.steps):
                optimizer.zero_grad()
                loss = _checked_objective(torch, objective, stage.name)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                stage_names.append(stage.name)
                stage_steps.append(step)
        else:
            optimizer = torch.optim.LBFGS(
                active,
                lr=stage.learning_rate,
                max_iter=stage.steps,
                tolerance_grad=gradient_tolerance,
                tolerance_change=relative_loss_tolerance,
                line_search_fn="strong_wolfe",
            )
            evaluation = 0

            def closure():
                nonlocal evaluation
                optimizer.zero_grad()
                loss = _checked_objective(torch, objective, stage.name)
                loss.backward()
                losses.append(float(loss.detach().cpu()))
                stage_names.append(stage.name)
                stage_steps.append(evaluation)
                evaluation += 1
                return loss

            optimizer.step(closure)

        training_after = _objective_value(torch, objective, stage.name)
        validation_after = (
            _objective_value(torch, validation_objective, stage.name)
            if validation_objective is not None
            else None
        )
        accepted = True
        reason = "training stage completed"
        if validation_before is not None and validation_after is not None:
            limit = validation_before + validation_tolerance * max(abs(validation_before), 1e-12)
            if validation_after > limit:
                accepted = False
                reason = "validation loss worsened; parameters restored"
                with torch.no_grad():
                    for name, value in groups.items():
                        value.copy_(snapshots[name])
            else:
                reason = "validation gate accepted stage"

        gradient_norm = _gradient_norm(torch, objective, active, stage.name)
        outcomes.append(
            StageOutcome(
                name=stage.name,
                optimizer=stage.optimizer,
                accepted=accepted,
                reason=reason,
                training_before=training_before,
                training_after=training_after,
                validation_before=validation_before,
                validation_after=validation_after,
                gradient_norm=gradient_norm,
                width_multiplier=stage.width_multiplier,
            )
        )

    released_names = tuple(
        dict.fromkeys(name for stage in declared_stages for name in stage.active)
    )
    final_gradient = _gradient_norm(
        torch,
        objective,
        [groups[name] for name in released_names],
        declared_stages[-1].name,
    )
    if len(losses) >= 2:
        relative_change = abs(losses[-1] - losses[-2]) / max(abs(losses[-2]), 1e-12)
    else:
        relative_change = np.inf
    if final_gradient <= gradient_tolerance:
        classification = "gradient_converged"
    elif relative_change <= relative_loss_tolerance:
        classification = "loss_stalled"
    elif any(not outcome.accepted for outcome in outcomes):
        classification = "completed_with_rollback"
    else:
        classification = "maximum_steps"

    return StagedOptimizationResult(
        loss=np.asarray(losses),
        stage=tuple(stage_names),
        step_in_stage=np.asarray(stage_steps, dtype=np.int64),
        final_values={name: value.detach().cpu().numpy().copy() for name, value in groups.items()},
        stage_outcomes=tuple(outcomes),
        convergence_classification=classification,
        final_gradient_norm=final_gradient,
        relative_loss_change=float(relative_change),
    )


def staged_adam(objective, parameter_groups, stages) -> StagedOptimizationResult:
    """Backward-compatible Adam-only staged optimization."""
    declared_stages = tuple(stages)
    if any(stage.optimizer != "adam" for stage in declared_stages):
        raise ValueError("staged_adam accepts only stages with optimizer='adam'")
    return staged_optimize(objective, parameter_groups, declared_stages)


def damped_gauss_newton(
    residual_function,
    initial_values,
    *,
    max_steps: int = 40,
    damping: float = 1e-3,
    trust_radius: float = 1.0,
    gradient_tolerance: float = 1e-7,
    step_tolerance: float = 1e-9,
) -> GaussNewtonResult:
    """Minimize a residual vector with damped Gauss--Newton trust steps."""
    torch = _require_torch("damped_gauss_newton")
    if max_steps <= 0 or damping <= 0 or trust_radius <= 0:
        raise ValueError("max_steps, damping, and trust_radius must be positive")
    values = torch.as_tensor(initial_values, dtype=torch.float64).detach().clone()
    if values.ndim != 1:
        raise ValueError("initial_values must be a one-dimensional vector")

    loss_history: list[float] = []
    damping_history: list[float] = []
    radius_history: list[float] = []
    accepted_history: list[bool] = []
    ratio_history: list[float] = []
    classification = "maximum_steps"
    current_damping = float(damping)
    current_radius = float(trust_radius)
    gradient_norm = np.inf

    for _ in range(max_steps):
        point = values.detach().requires_grad_(True)
        residual = residual_function(point)
        if residual.ndim != 1 or not torch.all(torch.isfinite(residual)):
            raise ValueError("residual_function must return a finite one-dimensional tensor")
        jacobian = torch.autograd.functional.jacobian(residual_function, point)
        loss = 0.5 * torch.dot(residual, residual)
        gradient = jacobian.T @ residual
        hessian = jacobian.T @ jacobian
        gradient_norm = float(torch.linalg.vector_norm(gradient).detach().cpu())
        loss_history.append(float(loss.detach().cpu()))
        damping_history.append(current_damping)
        radius_history.append(current_radius)
        if gradient_norm <= gradient_tolerance:
            classification = "gradient_converged"
            break

        identity = torch.eye(values.numel(), dtype=values.dtype, device=values.device)
        step = torch.linalg.solve(hessian + current_damping * identity, -gradient)
        step_norm = float(torch.linalg.vector_norm(step).detach().cpu())
        if step_norm > current_radius:
            step = step * (current_radius / step_norm)
            step_norm = current_radius
        if step_norm <= step_tolerance:
            classification = "step_converged"
            break

        candidate = (values + step.detach()).detach()
        candidate_residual = residual_function(candidate)
        candidate_loss = 0.5 * torch.dot(candidate_residual, candidate_residual)
        predicted = -(torch.dot(gradient, step) + 0.5 * torch.dot(step, hessian @ step))
        actual = loss.detach() - candidate_loss.detach()
        ratio = float((actual / torch.clamp(predicted.detach(), min=1e-18)).cpu())
        accept = bool(torch.isfinite(candidate_loss) and actual > 0 and ratio > 0)
        accepted_history.append(accept)
        ratio_history.append(ratio)
        if accept:
            values = candidate
        if ratio > 0.75:
            current_damping = max(current_damping * 0.5, 1e-12)
            current_radius = max(current_radius, 2.0 * step_norm)
        elif ratio < 0.25:
            current_damping = min(current_damping * 4.0, 1e12)
            current_radius = max(current_radius * 0.5, step_tolerance)

    final_point = values.detach().requires_grad_(True)
    final_residual = residual_function(final_point)
    final_jacobian = torch.autograd.functional.jacobian(residual_function, final_point)
    gradient_norm = float(
        torch.linalg.vector_norm(final_jacobian.T @ final_residual).detach().cpu()
    )
    if classification == "maximum_steps" and gradient_norm <= gradient_tolerance:
        classification = "gradient_converged"
    return GaussNewtonResult(
        values=values.detach().cpu().numpy().copy(),
        loss=np.asarray(loss_history),
        damping=np.asarray(damping_history),
        trust_radius=np.asarray(radius_history),
        accepted=np.asarray(accepted_history, dtype=bool),
        gain_ratio=np.asarray(ratio_history),
        gradient_norm=gradient_norm,
        convergence_classification=classification,
    )


def recommend_parameter_groups(
    sensitivity: Mapping[str, float],
    residual_support: Mapping[str, float],
    correlation: Mapping[tuple[str, str], float] | None = None,
    *,
    minimum_relative_sensitivity: float = 0.02,
    minimum_residual_support: float = 0.1,
    maximum_correlation: float = 0.98,
) -> tuple[ReleaseDecision, ...]:
    """Rank release candidates using information, residual, and correlation evidence."""
    if not sensitivity:
        raise ValueError("sensitivity must be non-empty")
    if not 0 <= minimum_relative_sensitivity <= 1:
        raise ValueError("minimum_relative_sensitivity must be in [0, 1]")
    if minimum_residual_support < 0 or not 0 <= maximum_correlation <= 1:
        raise ValueError("release thresholds are invalid")
    correlation = correlation or {}
    scale = max((abs(float(value)) for value in sensitivity.values()), default=0.0)
    decisions: list[ReleaseDecision] = []
    accepted: list[str] = []
    for group, raw_sensitivity in sensitivity.items():
        group_sensitivity = abs(float(raw_sensitivity))
        support = abs(float(residual_support.get(group, 0.0)))
        maximum = max(
            (
                abs(float(correlation.get((group, other), correlation.get((other, group), 0.0))))
                for other in accepted
            ),
            default=0.0,
        )
        relative = group_sensitivity / max(scale, 1e-30)
        if relative < minimum_relative_sensitivity:
            allow, reason = False, "insufficient pattern sensitivity"
        elif support < minimum_residual_support:
            allow, reason = False, "current residual does not support release"
        elif maximum > maximum_correlation:
            allow, reason = False, "too strongly correlated with an accepted group"
        else:
            allow, reason = True, "sensitivity and residual support justify release"
            accepted.append(group)
        decisions.append(
            ReleaseDecision(group, allow, group_sensitivity, support, maximum, reason)
        )
    return tuple(decisions)


def _require_torch(caller: str):
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise ImportError(f"{caller} requires the 'torch' extra") from error
    return torch


def _validate_groups_and_stages(torch, groups, stages):
    if not groups or not stages:
        raise ValueError("parameter_groups and stages must be non-empty")
    for name, value in groups.items():
        if not isinstance(value, torch.Tensor) or not value.is_leaf or not value.requires_grad:
            raise TypeError(f"parameter group {name!r} must be a grad-enabled Torch leaf tensor")
    for stage in stages:
        unknown = set(stage.active) - set(groups)
        if unknown:
            raise ValueError(f"stage {stage.name!r} names unknown groups: {sorted(unknown)}")


def _checked_objective(torch, objective: Callable, stage_name: str):
    loss = objective()
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError(f"objective returned invalid loss in stage {stage_name!r}")
    return loss


def _objective_value(torch, objective: Callable, stage_name: str) -> float:
    return float(_checked_objective(torch, objective, stage_name).detach().cpu())


def _gradient_norm(torch, objective, parameters, stage_name: str) -> float:
    loss = _checked_objective(torch, objective, stage_name)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    squares = [torch.sum(gradient.square()) for gradient in gradients if gradient is not None]
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).detach().cpu())
