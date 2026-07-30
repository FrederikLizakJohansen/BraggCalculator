"""Physically constrained experimental nuisance parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ProfileNuisanceParameterization:
    """Dimensionless raw variables mapped to physical profile parameters."""

    initial_scale: float = 1.0
    initial_zero_shift: float = 0.0
    zero_shift_scale: float = 0.01
    initial_fwhm: float = 0.02
    initial_background: float = 1.0

    def __post_init__(self):
        positive = {
            "initial_scale": self.initial_scale,
            "zero_shift_scale": self.zero_shift_scale,
            "initial_fwhm": self.initial_fwhm,
            "initial_background": self.initial_background,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(self.initial_zero_shift):
            raise ValueError("initial_zero_shift must be finite")

    @classmethod
    def from_calculator(
        cls,
        calculator,
        *,
        domain: Literal["two_theta", "q"] = "q",
        initial_scale: float = 1.0,
        initial_zero_shift: float = 0.0,
        zero_shift_scale: float | None = None,
        initial_background: float = 1.0,
    ) -> "ProfileNuisanceParameterization":
        if domain == "q":
            initial_fwhm = calculator.profile_q.fwhm_q
            shift_scale = calculator.q_step if zero_shift_scale is None else zero_shift_scale
        elif domain == "two_theta":
            initial_fwhm = calculator.profile.fwhm_deg
            shift_scale = (
                calculator.two_theta_step if zero_shift_scale is None else zero_shift_scale
            )
        else:
            raise ValueError("domain must be 'two_theta' or 'q'")
        return cls(
            initial_scale=initial_scale,
            initial_zero_shift=initial_zero_shift,
            zero_shift_scale=shift_scale,
            initial_fwhm=initial_fwhm,
            initial_background=initial_background,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return ("scale", "background", "zero_shift", "fwhm")

    def initial_values(self, backend, *, requires_grad: bool = False):
        """Return separate scalar leaves suitable for staged optimization."""
        result = {}
        for name in self.names:
            value = backend.zeros((), dtype=backend.dtype)
            if getattr(backend, "is_torch", False):
                value = value.clone().detach().requires_grad_(requires_grad)
            elif requires_grad:
                raise TypeError("requires_grad is available only with TorchBackend")
            result[name] = value
        return result

    def physical(self, values, backend):
        """Map raw values to positive scale/width/background and a free shift."""
        missing = set(self.names) - set(values)
        if missing:
            raise ValueError(f"missing nuisance parameters: {sorted(missing)}")
        return {
            "scale": self.initial_scale * backend.exp(values["scale"]),
            "background": self.initial_background * backend.exp(values["background"]),
            "zero_shift": self.initial_zero_shift
            + self.zero_shift_scale * values["zero_shift"],
            "fwhm": self.initial_fwhm * backend.exp(values["fwhm"]),
        }
