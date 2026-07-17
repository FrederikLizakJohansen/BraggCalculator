import numpy as np
import torch

from braggcalculator import (
    OptimizationStage,
    damped_gauss_newton,
    recommend_parameter_groups,
    staged_optimize,
)


def test_staged_lbfgs_converges_and_records_evidence():
    value = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)
    result = staged_optimize(
        lambda: (value - 2.5).square(),
        {"value": value},
        [OptimizationStage("polish", ("value",), 20, 1.0, optimizer="lbfgs")],
    )

    np.testing.assert_allclose(result.final_values["value"], 2.5, atol=1e-7)
    assert result.stage_outcomes[0].optimizer == "lbfgs"
    assert result.stage_outcomes[0].accepted
    assert result.convergence_classification == "gradient_converged"


def test_validation_gate_rolls_back_a_harmful_stage():
    value = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
    result = staged_optimize(
        lambda: (value - 1.0).square(),
        {"value": value},
        [OptimizationStage("harmful", ("value",), 40, 0.15)],
        validation_objective=lambda: (value + 1.0).square(),
    )

    np.testing.assert_allclose(result.final_values["value"], -1.0)
    assert not result.stage_outcomes[0].accepted
    assert "restored" in result.stage_outcomes[0].reason
    assert result.convergence_classification == "completed_with_rollback"


def test_continuation_hook_observes_declared_widths():
    value = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    widths = []
    stages = [
        OptimizationStage("wide", ("value",), 2, 0.1, width_multiplier=2.5),
        OptimizationStage("physical", ("value",), 2, 0.1, width_multiplier=1.0),
    ]
    staged_optimize(
        lambda: (value - 1.0).square(),
        {"value": value},
        stages,
        before_stage=lambda stage: widths.append(stage.width_multiplier),
    )
    assert widths == [2.5, 1.0]


def test_damped_gauss_newton_solves_nonlinear_residual():
    target = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)

    def residual(values):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=values.dtype)
        return values[0] * x.pow(values[1]) - target

    result = damped_gauss_newton(residual, [0.35, 1.2], trust_radius=0.5)

    np.testing.assert_allclose(result.values, [1.0, 2.0], atol=1e-6)
    assert result.loss[-1] < result.loss[0] * 1e-8
    assert result.accepted.any()
    assert result.convergence_classification == "gradient_converged"


def test_adaptive_release_requires_signal_support_and_independence():
    decisions = recommend_parameter_groups(
        {"lattice": 10.0, "occupancy": 4.0, "adp": 0.05, "duplicate": 8.0},
        {"lattice": 3.0, "occupancy": 0.01, "adp": 2.0, "duplicate": 2.0},
        {("duplicate", "lattice"): 0.995},
    )
    by_name = {decision.group: decision for decision in decisions}

    assert by_name["lattice"].accepted
    assert not by_name["occupancy"].accepted
    assert "residual" in by_name["occupancy"].reason
    assert not by_name["adp"].accepted
    assert "sensitivity" in by_name["adp"].reason
    assert not by_name["duplicate"].accepted
    assert "correlated" in by_name["duplicate"].reason
