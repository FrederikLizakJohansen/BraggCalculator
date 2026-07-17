"""Small declared-stage optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptimizationStage:
    """One optimizer stage and the parameter groups released within it."""

    name: str
    active: tuple[str, ...]
    steps: int
    learning_rate: float

    def __post_init__(self):
        if not self.name or not self.active:
            raise ValueError("stage name and active groups must be non-empty")
        if self.steps <= 0:
            raise ValueError("stage steps must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("stage learning_rate must be positive and finite")


@dataclass(frozen=True)
class StagedOptimizationResult:
    """Loss trace and final raw values from a declared staged optimization."""

    loss: np.ndarray
    stage: tuple[str, ...]
    step_in_stage: np.ndarray
    final_values: dict[str, np.ndarray]


def staged_adam(objective, parameter_groups, stages) -> StagedOptimizationResult:
    """Run Adam stages over selected Torch leaf tensors.

    ``objective`` is a zero-argument callable returning a real scalar tensor.
    Parameter groups not named by a stage are not passed to that stage's
    optimizer and therefore remain unchanged.
    """
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise ImportError("staged_adam requires the 'torch' extra") from error
    groups = dict(parameter_groups)
    declared_stages = tuple(stages)
    if not groups or not declared_stages:
        raise ValueError("parameter_groups and stages must be non-empty")
    for name, value in groups.items():
        if not isinstance(value, torch.Tensor) or not value.is_leaf or not value.requires_grad:
            raise TypeError(f"parameter group {name!r} must be a grad-enabled Torch leaf tensor")

    losses = []
    stage_names = []
    stage_steps = []
    for stage in declared_stages:
        unknown = set(stage.active) - set(groups)
        if unknown:
            raise ValueError(f"stage {stage.name!r} names unknown groups: {sorted(unknown)}")
        optimizer = torch.optim.Adam(
            [groups[name] for name in stage.active], lr=stage.learning_rate
        )
        for step in range(stage.steps):
            optimizer.zero_grad()
            loss = objective()
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError(f"objective returned invalid loss in stage {stage.name!r}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            stage_names.append(stage.name)
            stage_steps.append(step)

    return StagedOptimizationResult(
        loss=np.asarray(losses),
        stage=tuple(stage_names),
        step_in_stage=np.asarray(stage_steps, dtype=np.int64),
        final_values={name: value.detach().cpu().numpy().copy() for name, value in groups.items()},
    )
