"""Reproducible metric primitives for the diffraction-diagnostics publication."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from .diagnostics import mismatch_disk


PUBLICATION_SCHEMA = "braggcalculator.diagnostics-publication/v1"
WEIGHTING_SCHEMES = (
    "uniform",
    "mean_intensity",
    "sqrt_mean_intensity",
    "shell_balanced_intensity",
)


def cyclic_difference_multiset(points: Sequence[int], modulus: int) -> dict[int, int]:
    """Count all directed periodic differences for a finite cyclic point set."""
    selected = tuple(int(value) % modulus for value in points)
    if modulus < 2 or not selected or len(set(selected)) != len(selected):
        raise ValueError("points must be a non-empty unique subset of a cyclic group")
    return dict(sorted(Counter((left - right) % modulus for left in selected for right in selected).items()))


def cyclic_sets_dihedrally_equivalent(
    first: Sequence[int], second: Sequence[int], modulus: int
) -> bool:
    """Test equivalence by cyclic translation or inversion plus translation."""
    reference = {int(value) % modulus for value in first}
    candidate = {int(value) % modulus for value in second}
    if len(reference) != len(first) or len(candidate) != len(second):
        raise ValueError("cyclic sets must contain unique points")
    return any(
        {(sign * value + shift) % modulus for value in reference} == candidate
        for sign in (1, -1)
        for shift in range(modulus)
    )


def cosine_similarity(left, right) -> float:
    """Return the normalized dot product of two finite profile vectors."""
    first, second = _profile_pair(left, right)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def pearson_similarity(left, right) -> float:
    """Return the centered linear correlation, with constant equality handled."""
    first, second = _profile_pair(left, right)
    centered_a = first - first.mean()
    centered_b = second - second.mean()
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    if denominator:
        return float(np.dot(centered_a, centered_b) / denominator)
    return float(np.array_equal(first, second))


def jensen_shannon_similarity(left, right) -> float:
    """Compare non-negative normalized profile mass on a bounded [0, 1] scale."""
    first, second = _profile_pair(left, right, nonnegative=True)
    sum_a, sum_b = float(first.sum()), float(second.sum())
    if sum_a <= 0 or sum_b <= 0:
        return float(sum_a == sum_b)
    probability_a = first / sum_a
    probability_b = second / sum_b
    midpoint = 0.5 * (probability_a + probability_b)
    divergence = 0.5 * _kl_divergence(probability_a, midpoint) + 0.5 * _kl_divergence(
        probability_b, midpoint
    )
    distance = np.sqrt(max(divergence, 0.0) / np.log(2.0))
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def gaussian_cross_correlation_similarity(
    left,
    right,
    *,
    coordinate_step: float,
    tolerance: float,
    maximum_shift: float | None = None,
) -> float:
    """Return a normalized Gaussian-weighted cross-correlation similarity.

    This is a transparent member of the generalized weighted cross-correlation
    family used for powder-pattern comparison. It is intentionally named by
    its implemented kernel rather than claimed as a byte-for-byte reproduction
    of any proprietary or external program.
    """
    first, second = _profile_pair(left, right, nonnegative=True)
    if coordinate_step <= 0 or tolerance <= 0:
        raise ValueError("coordinate_step and tolerance must be positive")
    if maximum_shift is None:
        maximum_shift = 4.0 * tolerance
    if maximum_shift <= 0:
        raise ValueError("maximum_shift must be positive")
    maximum_lag = min(int(np.ceil(maximum_shift / coordinate_step)), len(first) - 1)
    lags = np.arange(-maximum_lag, maximum_lag + 1)
    kernel = np.exp(-0.5 * (lags * coordinate_step / tolerance) ** 2)
    correlation_ab = _selected_correlation(first, second, lags)
    correlation_aa = _selected_correlation(first, first, lags)
    correlation_bb = _selected_correlation(second, second, lags)
    numerator = float(kernel @ correlation_ab)
    denominator = float(np.sqrt((kernel @ correlation_aa) * (kernel @ correlation_bb)))
    return float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator else 0.0


def profile_metric_suite(
    left,
    right,
    *,
    coordinate_step: float,
    cross_correlation_tolerance: float,
) -> dict[str, float]:
    """Evaluate the frozen publication baseline metrics on one shared grid."""
    return {
        "cosine": cosine_similarity(left, right),
        "pearson": pearson_similarity(left, right),
        "jensen_shannon": jensen_shannon_similarity(left, right),
        "gaussian_cross_correlation": gaussian_cross_correlation_similarity(
            left,
            right,
            coordinate_step=coordinate_step,
            tolerance=cross_correlation_tolerance,
        ),
    }


def mismatch_weights(
    q,
    amplitude_a,
    amplitude_b,
    *,
    scheme: str,
    shell_width: float = 0.5,
) -> np.ndarray:
    """Construct one normalized reflection-weight declaration."""
    reciprocal = np.asarray(q, dtype=np.float64)
    first = np.asarray(amplitude_a, dtype=np.float64)
    second = np.asarray(amplitude_b, dtype=np.float64)
    if reciprocal.ndim != 1 or first.shape != reciprocal.shape or second.shape != first.shape:
        raise ValueError("q and both amplitude arrays must be matching one-dimensional vectors")
    if len(reciprocal) == 0 or not np.all(np.isfinite(reciprocal)):
        raise ValueError("q must contain finite reflections")
    if np.any(first < 0) or np.any(second < 0) or not np.all(np.isfinite(first + second)):
        raise ValueError("amplitudes must be finite and non-negative")
    if scheme not in WEIGHTING_SCHEMES:
        raise ValueError(f"unknown weighting scheme: {scheme}")
    mean_intensity = 0.5 * (first**2 + second**2)
    if scheme == "uniform":
        weights = np.ones_like(reciprocal)
    elif scheme == "mean_intensity":
        weights = mean_intensity
    elif scheme == "sqrt_mean_intensity":
        weights = np.sqrt(mean_intensity)
    else:
        if shell_width <= 0 or not np.isfinite(shell_width):
            raise ValueError("shell_width must be positive and finite")
        shell = np.floor((reciprocal - reciprocal.min()) / shell_width).astype(int)
        weights = np.zeros_like(reciprocal)
        unique = np.unique(shell)
        for index in unique:
            selected = shell == index
            total = float(mean_intensity[selected].sum())
            if total > 0:
                weights[selected] = mean_intensity[selected] / total / len(unique)
            else:
                weights[selected] = 1.0 / selected.sum() / len(unique)
    total = float(weights.sum())
    if total <= 0:
        weights = np.ones_like(reciprocal)
        total = float(len(weights))
    return weights / total


def compare_weighting_schemes(
    hkl,
    q,
    structure_factor_a,
    structure_factor_b,
    *,
    schemes: Sequence[str] = WEIGHTING_SCHEMES,
    shell_width: float = 0.5,
    optimize_origin: bool = True,
) -> dict[str, dict[str, float | list[float]]]:
    """Evaluate the mismatch decomposition under declared reflection weights."""
    factors_a = np.asarray(structure_factor_a, dtype=np.complex128)
    factors_b = np.asarray(structure_factor_b, dtype=np.complex128)
    results = {}
    for scheme in schemes:
        weights = mismatch_weights(
            q,
            np.abs(factors_a),
            np.abs(factors_b),
            scheme=scheme,
            shell_width=shell_width,
        )
        result = mismatch_disk(
            hkl,
            factors_a,
            factors_b,
            weights=weights,
            optimize_origin=optimize_origin,
        )
        results[scheme] = {
            "d_sf": result.d_sf,
            "d_amplitude": result.d_amplitude,
            "d_phase": result.d_phase,
            "origin_shift": result.alignment.shift.tolist(),
            "identity_error": result.identity_error,
            "effective_reflections": float(1.0 / np.sum(result.weights**2)),
        }
    return results


def verify_input_manifest(root, manifest_path) -> dict[str, str]:
    """Verify every publication input against its frozen SHA-256 digest."""
    base = Path(root).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != PUBLICATION_SCHEMA:
        raise ValueError("unsupported publication manifest schema")
    verified = {}
    for record in manifest["files"]:
        path = (base / record["path"]).resolve()
        try:
            path.relative_to(base)
        except ValueError as error:
            raise ValueError("publication manifest path escapes its root") from error
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"publication input checksum changed: {record['path']}")
        verified[record["path"]] = digest
    return verified


def publication_gate_summary(gates: Mapping[str, bool | None]) -> str:
    """Reduce gates without allowing an unsigned external review to disappear."""
    if any(value is False for value in gates.values()):
        return "failed"
    if any(value is None for value in gates.values()):
        return "pending_external_review"
    return "passed"


def _profile_pair(left, right, *, nonnegative=False):
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.ndim != 1 or second.shape != first.shape or len(first) < 2:
        raise ValueError("profiles must be matching one-dimensional vectors")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("profiles must be finite")
    if nonnegative and (np.any(first < 0) or np.any(second < 0)):
        raise ValueError("this metric requires non-negative profiles")
    return first, second


def _kl_divergence(probability, reference):
    selected = probability > 0
    return float(np.sum(probability[selected] * np.log(probability[selected] / reference[selected])))


def _selected_correlation(left, right, lags):
    complete = np.correlate(left, right, mode="full")
    center = len(left) - 1
    return complete[center + lags]
