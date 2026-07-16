"""Structured diffraction result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReflectionTable:
    """Per-reciprocal-point quantities for one configured output domain.

    ``hkl`` is an integer NumPy array because reflection indices are discrete.
    Numerical columns use the calculator's configured NumPy or Torch backend.
    Angles are in degrees, reciprocal quantities in inverse angstroms, and
    d-spacings in angstroms.
    """

    hkl: np.ndarray
    d_spacing: Any
    q: Any
    two_theta: Any
    f_squared: Any
    intensity: Any

    def __len__(self) -> int:
        return len(self.hkl)
