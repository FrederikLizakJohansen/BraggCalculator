import numpy as np
import pytest

from braggcalculator import (
    BraggCalculator,
    OptimizationStage,
    ProfileNuisanceParameterization,
    staged_adam,
)
from braggcalculator.backends import TorchBackend

torch = pytest.importorskip("torch")


def test_nuisance_parameterization_is_positive_and_differentiable(triclinic_structure):
    calculator = BraggCalculator(
        backend=TorchBackend(), q_range=(0.5, 4.0), q_step=0.05
    ).load(triclinic_structure)
    model = ProfileNuisanceParameterization.from_calculator(
        calculator, domain="q", initial_background=2.0
    )
    raw = model.initial_values(calculator.backend, requires_grad=True)
    physical = model.physical(raw, calculator.backend)
    assert float(physical["scale"].detach()) > 0
    assert float(physical["fwhm"].detach()) > 0
    assert float(physical["background"].detach()) > 0
    _, profile = calculator.pattern(domain="q", experiment_parameters=physical)
    loss = torch.sum(profile * torch.linspace(0.5, 1.5, len(profile)))
    loss.backward()
    for name, value in raw.items():
        assert value.grad is not None, name
        assert torch.isfinite(value.grad), name


def test_profile_controls_apply_scale_background_and_shift(triclinic_structure):
    calculator = BraggCalculator(q_range=(0.5, 4.0), q_step=0.02).load(triclinic_structure)
    grid, baseline = calculator.pattern(domain="q")
    _, changed = calculator.pattern(
        domain="q",
        experiment_parameters={
            "scale": 2.0,
            "background": 3.0,
            "zero_shift": 0.0,
        },
    )
    np.testing.assert_allclose(changed, 2.0 * baseline + 3.0)
    _, shifted = calculator.pattern(
        domain="q", experiment_parameters={"zero_shift": calculator.q_step}
    )
    assert grid[np.argmax(shifted)] == pytest.approx(
        grid[np.argmax(baseline)] + calculator.q_step
    )


def test_staged_adam_optimizes_only_declared_groups():
    x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    stages = (
        OptimizationStage("x only", ("x",), 150, 0.05),
        OptimizationStage("y only", ("y",), 180, 0.05),
    )

    def objective():
        return (x - 2.0) ** 2 + (y + 3.0) ** 2

    result = staged_adam(objective, {"x": x, "y": y}, stages)
    assert float(x.detach()) == pytest.approx(2.0, abs=2e-3)
    assert float(y.detach()) == pytest.approx(-3.0, abs=2e-3)
    assert result.stage[0] == "x only"
    assert result.stage[-1] == "y only"
    assert len(result.loss) == 330


def test_staged_adam_rejects_unknown_group():
    value = torch.tensor(0.0, requires_grad=True)
    stage = OptimizationStage("bad", ("missing",), 1, 0.1)
    with pytest.raises(ValueError, match="unknown groups"):
        staged_adam(lambda: value.square(), {"value": value}, [stage])
