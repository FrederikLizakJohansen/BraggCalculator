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


@dataclass(frozen=True)
class ProfileDiscriminationResult:
    """Expected separation of two profiles expressed as measured-bin values."""

    coordinate: np.ndarray
    expected_a: np.ndarray
    expected_b: np.ndarray
    difference: np.ndarray
    variance: np.ndarray | None
    covariance: np.ndarray | None
    whitened_difference: np.ndarray
    pointwise_discrimination: np.ndarray | None
    total_discrimination: float
    bin_widths: np.ndarray | None = None


@dataclass(frozen=True)
class StructuralRelationship:
    """Crystallographic relationship that controls valid pair diagnostics."""

    regime: str
    classification: str
    transformation: np.ndarray | None
    transformation_direction: str | None
    volume_ratio: float
    complex_comparison_allowed: bool
    reason: str


@dataclass(frozen=True)
class PairDistributionComparison:
    """Scattering-weighted periodic radial pair-distribution comparison."""

    radius: np.ndarray
    distribution_a: np.ndarray
    distribution_b: np.ndarray
    similarity: float
    radiation: str
    broadening: float
    r_max: float


@dataclass(frozen=True)
class SuperstructureResult:
    """Reflections in a supercell that do not map to parent-cell indices."""

    parent: str
    supercell: str
    transformation: np.ndarray
    hkl: np.ndarray
    q: np.ndarray
    intensity: np.ndarray
    intensity_fraction: float


@dataclass(frozen=True)
class PeakGroupDiagnostic:
    """One resolution-defined powder group and its counterfactual site effects."""

    q_center: float
    q_min: float
    q_max: float
    integrated_intensity: float
    effective_reflections: float
    hkl: np.ndarray
    reflection_intensity: np.ndarray
    site_effects: dict[str, float]


@dataclass(frozen=True)
class CounterfactualAttribution:
    """Profile effect of substituting a declared site or motif group."""

    name: str
    site_indices: tuple[int, ...]
    effect_norm: float
    alignment_fraction: float
    largest_effect_coordinate: float
    profile_change: np.ndarray


@dataclass(frozen=True)
class ExperimentRecommendation:
    """Expected model discrimination for one declared measurement setup."""

    name: str
    radiation: str
    wavelength: float
    q_range: tuple[float, float]
    fwhm_q: float
    total_discrimination: float
    most_informative_q: float
    assumptions: dict[str, Any]


@dataclass(frozen=True)
class StructuralDiagnosticsResult:
    """High-level, relationship-aware structural diagnostic report."""

    relationship: StructuralRelationship
    similarities: dict[str, float | None]
    dominant_information_loss: str
    explanation: str
    mismatch: MismatchDiskResult | None
    profile_discrimination: ProfileDiscriminationResult
    pair_distribution: PairDistributionComparison
    superstructure: SuperstructureResult | None
    peak_groups_a: tuple[PeakGroupDiagnostic, ...]
    peak_groups_b: tuple[PeakGroupDiagnostic, ...]
    counterfactuals: tuple[CounterfactualAttribution, ...]


@dataclass(frozen=True)
class JacobianDiagnostics:
    """Local parameter information calculated from a scaled profile Jacobian."""

    parameter_names: tuple[str, ...]
    parameter_scales: np.ndarray
    jacobian: np.ndarray
    scaled_jacobian: np.ndarray
    normal_matrix: np.ndarray
    sensitivity: np.ndarray
    residual_support: np.ndarray | None
    column_cosine: np.ndarray
    generalized_covariance_scaled: np.ndarray
    generalized_covariance_physical: np.ndarray
    correlation: np.ndarray
    local_information: np.ndarray | None
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    rank: int
    condition_number: float
    covariance_is_identifiable: bool
    prior_precision_scaled: np.ndarray
    posterior_normal_matrix: np.ndarray
    prior_rank: int
    posterior_rank: int
    posterior_condition_number: float
    posterior_covariance_is_identifiable: bool
    posterior_covariance_scaled: np.ndarray
    posterior_covariance_physical: np.ndarray
    posterior_correlation: np.ndarray
    null_space_vectors: np.ndarray
    standard_errors_scaled: np.ndarray
    standard_errors_physical: np.ndarray
