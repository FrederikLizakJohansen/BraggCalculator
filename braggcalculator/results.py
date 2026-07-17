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
    structure_factor: Any
    f_squared: Any
    intensity: Any

    def __len__(self) -> int:
        return len(self.hkl)


@dataclass(frozen=True)
class ReflectionMatch:
    """Exact Miller-index correspondence between two reflection collections."""

    hkl: np.ndarray
    indices_a: np.ndarray
    indices_b: np.ndarray

    def __len__(self) -> int:
        return len(self.hkl)


@dataclass(frozen=True)
class OriginAlignment:
    """Relative-origin correction applied to the second structure-factor set."""

    shift: np.ndarray
    similarity: float
    structure_factor_b: np.ndarray


@dataclass(frozen=True)
class MismatchDiskResult:
    """Bounded amplitude-phase comparison for matched complex reflections."""

    match: ReflectionMatch
    alignment: OriginAlignment
    structure_factor_a: np.ndarray
    amplitude_a: np.ndarray
    amplitude_b: np.ndarray
    phase_difference: np.ndarray
    phase_defined: np.ndarray
    x: np.ndarray
    y: np.ndarray
    radius: np.ndarray
    weights: np.ndarray
    d_sf: float
    d_amplitude: float
    d_phase: float
    identity_error: float
    epsilon: float
    phase_threshold: float
