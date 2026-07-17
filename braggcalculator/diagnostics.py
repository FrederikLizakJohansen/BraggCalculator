"""Complex structure-factor diagnostics for lattice-compatible models."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from .results import (
    MismatchDiskResult,
    OriginAlignment,
    ProfileDiscriminationResult,
    ReflectionMatch,
)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _bin_widths(coordinate) -> np.ndarray:
    centers = np.asarray(coordinate, dtype=np.float64)
    if centers.ndim != 1 or len(centers) < 2 or not np.all(np.isfinite(centers)):
        raise ValueError("coordinate must contain at least two finite bin centers")
    spacing = np.diff(centers)
    if np.any(spacing <= 0):
        raise ValueError("coordinate must be strictly increasing")
    edges = np.empty(len(centers) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * spacing[0]
    edges[-1] = centers[-1] + 0.5 * spacing[-1]
    return np.diff(edges)


def profile_discrimination(
    coordinate,
    expected_a,
    expected_b,
    *,
    variance=None,
    covariance=None,
    bin_widths=None,
) -> ProfileDiscriminationResult:
    """Compare two vectors of expected measured-bin values under an error model.

    Exactly one of ``variance`` (independent bins) and ``covariance`` must be
    supplied. For a full covariance matrix, only whitened contributions are
    returned because assigning correlated information to original bins is not
    unique.
    """
    centers = np.asarray(coordinate, dtype=np.float64)
    model_a = np.asarray(expected_a, dtype=np.float64)
    model_b = np.asarray(expected_b, dtype=np.float64)
    if centers.ndim != 1 or model_a.shape != centers.shape or model_b.shape != centers.shape:
        raise ValueError("coordinate and expected profiles must be equal-length vectors")
    if not all(np.all(np.isfinite(values)) for values in (centers, model_a, model_b)):
        raise ValueError("coordinate and expected profiles must be finite")
    if (variance is None) == (covariance is None):
        raise ValueError("provide exactly one of variance or covariance")

    widths = None
    if bin_widths is not None:
        widths = np.asarray(bin_widths, dtype=np.float64)
        if widths.shape != centers.shape or np.any(widths <= 0) or not np.all(np.isfinite(widths)):
            raise ValueError("bin_widths must be a positive finite vector")

    difference = model_a - model_b
    variance_array = None
    covariance_array = None
    pointwise = None
    if variance is not None:
        variance_array = np.asarray(variance, dtype=np.float64)
        if (
            variance_array.shape != centers.shape
            or np.any(variance_array <= 0)
            or not np.all(np.isfinite(variance_array))
        ):
            raise ValueError("variance must be a positive finite vector")
        whitened = difference / np.sqrt(variance_array)
        pointwise = whitened**2
    else:
        covariance_array = np.asarray(covariance, dtype=np.float64)
        expected_shape = (len(centers), len(centers))
        if covariance_array.shape != expected_shape or not np.all(np.isfinite(covariance_array)):
            raise ValueError(f"covariance must be a finite matrix with shape {expected_shape}")
        if not np.allclose(covariance_array, covariance_array.T, rtol=1e-12, atol=1e-14):
            raise ValueError("covariance must be symmetric")
        try:
            cholesky = np.linalg.cholesky(covariance_array)
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance must be positive definite") from error
        whitened = np.linalg.solve(cholesky, difference)

    return ProfileDiscriminationResult(
        coordinate=centers,
        expected_a=model_a,
        expected_b=model_b,
        difference=difference,
        variance=variance_array,
        covariance=covariance_array,
        whitened_difference=whitened,
        pointwise_discrimination=pointwise,
        total_discrimination=float(whitened @ whitened),
        bin_widths=widths,
    )


def compare_profile_counts(
    calculator_a,
    calculator_b,
    *,
    domain: str = "q",
    parameters_a=None,
    parameters_b=None,
    count_scale: float = 1.0,
    background_density=0.0,
    minimum_variance: float = 1.0,
) -> ProfileDiscriminationResult:
    """Compare calculator profiles using an explicit synthetic count model.

    Calculated profile densities are converted to expected bin counts using the
    bin widths and ``count_scale``. ``background_density`` is expected counts
    per coordinate unit. The symmetric Poisson approximation uses the mean of
    the two expected counts as its variance.
    """
    if not np.isfinite(count_scale) or count_scale <= 0:
        raise ValueError("count_scale must be positive and finite")
    if not np.isfinite(minimum_variance) or minimum_variance <= 0:
        raise ValueError("minimum_variance must be positive and finite")
    grid_a, density_a = calculator_a.pattern(domain=domain, parameters=parameters_a)
    grid_b, density_b = calculator_b.pattern(domain=domain, parameters=parameters_b)
    coordinate = _as_numpy(grid_a).astype(np.float64, copy=False)
    if not np.allclose(coordinate, _as_numpy(grid_b), rtol=0.0, atol=1e-12):
        raise ValueError("calculators must produce the same profile grid")
    widths = _bin_widths(coordinate)
    background = np.asarray(background_density, dtype=np.float64)
    if background.ndim == 0:
        background = np.full_like(coordinate, float(background))
    if background.shape != coordinate.shape or np.any(background < 0) or not np.all(
        np.isfinite(background)
    ):
        raise ValueError("background_density must be non-negative and scalar or grid-shaped")

    expected_a = np.maximum(_as_numpy(density_a), 0.0) * widths * count_scale
    expected_b = np.maximum(_as_numpy(density_b), 0.0) * widths * count_scale
    background_counts = background * widths
    expected_a = expected_a + background_counts
    expected_b = expected_b + background_counts
    variance = np.maximum(0.5 * (expected_a + expected_b), minimum_variance)
    return profile_discrimination(
        coordinate,
        expected_a,
        expected_b,
        variance=variance,
        bin_widths=widths,
    )


def match_reflections(hkl_a, hkl_b) -> ReflectionMatch:
    """Match unique, exactly equal Miller indices while preserving A's order."""
    a = np.asarray(hkl_a, dtype=np.int64)
    b = np.asarray(hkl_b, dtype=np.int64)
    if a.ndim != 2 or a.shape[1:] != (3,) or b.ndim != 2 or b.shape[1:] != (3,):
        raise ValueError("hkl arrays must both have shape (n, 3)")

    keys_a = [tuple(map(int, row)) for row in a]
    keys_b = [tuple(map(int, row)) for row in b]
    if len(set(keys_a)) != len(keys_a) or len(set(keys_b)) != len(keys_b):
        raise ValueError("hkl arrays must not contain duplicate Miller indices")

    lookup_b = {key: index for index, key in enumerate(keys_b)}
    pairs = [(index, lookup_b[key]) for index, key in enumerate(keys_a) if key in lookup_b]
    if not pairs:
        raise ValueError("the reflection collections have no Miller indices in common")

    indices_a = np.fromiter((pair[0] for pair in pairs), dtype=np.int64)
    indices_b = np.fromiter((pair[1] for pair in pairs), dtype=np.int64)
    return ReflectionMatch(hkl=a[indices_a].copy(), indices_a=indices_a, indices_b=indices_b)


def apply_origin_shift(structure_factors, hkl, shift) -> np.ndarray:
    """Apply ``exp(2 pi i h.shift)`` to a complex structure-factor array."""
    factors = np.asarray(structure_factors, dtype=np.complex128)
    indices = np.asarray(hkl, dtype=np.float64)
    translation = np.asarray(shift, dtype=np.float64)
    if factors.shape != (len(indices),):
        raise ValueError("structure_factors must have one value per hkl row")
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("shift must be a finite three-vector")
    return factors * np.exp(2j * np.pi * (indices @ translation))


def _normalized_weights(weights, count: int) -> np.ndarray:
    if weights is None:
        return np.full(count, 1.0 / count)
    result = np.asarray(weights, dtype=np.float64)
    if result.shape != (count,) or not np.all(np.isfinite(result)):
        raise ValueError("weights must be a finite vector with one value per reflection")
    if np.any(result < 0) or not float(result.sum()) > 0:
        raise ValueError("weights must be non-negative with a positive sum")
    return result / result.sum()


def _alignment_costs(candidates, hkl, factor_a, factor_b, weights, denominator):
    costs = np.empty(len(candidates), dtype=np.float64)
    for start in range(0, len(candidates), 128):
        stop = min(start + 128, len(candidates))
        phase = np.exp(2j * np.pi * (hkl @ candidates[start:stop].T))
        difference = factor_b[:, None] * phase - factor_a[:, None]
        costs[start:stop] = np.sum(
            weights[:, None] * np.abs(difference) ** 2 / denominator[:, None] ** 2,
            axis=0,
        )
    return costs


def align_relative_origin(
    hkl,
    structure_factor_a,
    structure_factor_b,
    *,
    weights=None,
    epsilon: float | None = None,
    grid_size: int = 8,
    refinement_steps: int = 10,
) -> OriginAlignment:
    """Find a periodic relative-origin correction by grid and local refinement."""
    indices = np.asarray(hkl, dtype=np.float64)
    factor_a = np.asarray(structure_factor_a, dtype=np.complex128)
    factor_b = np.asarray(structure_factor_b, dtype=np.complex128)
    if factor_a.shape != (len(indices),) or factor_b.shape != factor_a.shape:
        raise ValueError("both structure-factor arrays must have one value per hkl row")
    if grid_size < 2 or refinement_steps < 0:
        raise ValueError("grid_size must be at least 2 and refinement_steps non-negative")
    normalized_weights = _normalized_weights(weights, len(indices))
    scale = max(float(np.abs(factor_a).max()), float(np.abs(factor_b).max()), 1.0)
    regularizer = np.finfo(np.float64).eps * scale if epsilon is None else float(epsilon)
    if not np.isfinite(regularizer) or regularizer < 0:
        raise ValueError("epsilon must be finite and non-negative")
    denominator = np.abs(factor_a) + np.abs(factor_b) + regularizer

    axis = np.arange(grid_size, dtype=np.float64) / grid_size
    candidates = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    costs = _alignment_costs(
        candidates, indices, factor_a, factor_b, normalized_weights, denominator
    )
    shift = candidates[int(np.argmin(costs))]
    best_cost = float(np.min(costs))

    step = 1.0 / grid_size
    offsets = np.asarray(list(product((-1.0, 0.0, 1.0), repeat=3)))
    for _ in range(refinement_steps):
        candidates = (shift[None, :] + step * offsets) % 1.0
        costs = _alignment_costs(
            candidates, indices, factor_a, factor_b, normalized_weights, denominator
        )
        best = int(np.argmin(costs))
        if float(costs[best]) <= best_cost:
            shift = candidates[best]
            best_cost = float(costs[best])
        step *= 0.5

    transformed = apply_origin_shift(factor_b, indices, shift)
    return OriginAlignment(
        shift=shift,
        similarity=float(np.clip(1.0 - best_cost, 0.0, 1.0)),
        structure_factor_b=transformed,
    )


def mismatch_disk(
    hkl,
    structure_factor_a,
    structure_factor_b,
    *,
    weights=None,
    origin_shift=None,
    optimize_origin: bool = False,
    epsilon: float | None = None,
    phase_threshold: float | None = None,
    origin_grid_size: int = 8,
    origin_refinement_steps: int = 10,
) -> MismatchDiskResult:
    """Calculate the bounded amplitude-phase mismatch disk for matched HKLs."""
    indices = np.asarray(hkl, dtype=np.int64)
    factor_a = np.asarray(structure_factor_a, dtype=np.complex128)
    factor_b = np.asarray(structure_factor_b, dtype=np.complex128)
    if indices.ndim != 2 or indices.shape[1:] != (3,):
        raise ValueError("hkl must have shape (n, 3)")
    if factor_a.shape != (len(indices),) or factor_b.shape != factor_a.shape:
        raise ValueError("both structure-factor arrays must have one value per hkl row")
    if len(indices) == 0:
        raise ValueError("at least one matched reflection is required")
    if optimize_origin and origin_shift is not None:
        raise ValueError("provide origin_shift or request optimization, not both")

    normalized_weights = _normalized_weights(weights, len(indices))
    scale = max(float(np.abs(factor_a).max()), float(np.abs(factor_b).max()), 1.0)
    regularizer = np.finfo(np.float64).eps * scale if epsilon is None else float(epsilon)
    if not np.isfinite(regularizer) or regularizer < 0:
        raise ValueError("epsilon must be finite and non-negative")

    if optimize_origin:
        alignment = align_relative_origin(
            indices,
            factor_a,
            factor_b,
            weights=normalized_weights,
            epsilon=regularizer,
            grid_size=origin_grid_size,
            refinement_steps=origin_refinement_steps,
        )
    else:
        shift = np.zeros(3) if origin_shift is None else np.asarray(origin_shift, dtype=float)
        transformed = apply_origin_shift(factor_b, indices, shift)
        denominator_for_cost = np.abs(factor_a) + np.abs(factor_b) + regularizer
        cost = np.sum(
            normalized_weights * np.abs(transformed - factor_a) ** 2 / denominator_for_cost**2
        )
        alignment = OriginAlignment(
            shift=shift % 1.0,
            similarity=float(np.clip(1.0 - cost, 0.0, 1.0)),
            structure_factor_b=transformed,
        )

    amplitude_a = np.abs(factor_a)
    amplitude_b = np.abs(alignment.structure_factor_b)
    denominator = amplitude_a + amplitude_b + regularizer
    phase_difference = np.angle(alignment.structure_factor_b * np.conj(factor_a))
    threshold = 1e-10 * scale if phase_threshold is None else float(phase_threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("phase_threshold must be finite and non-negative")
    phase_defined = (amplitude_a > threshold) & (amplitude_b > threshold)

    x = (amplitude_b - amplitude_a) / denominator
    y = 2.0 * np.sqrt(amplitude_a * amplitude_b) / denominator * np.sin(
        phase_difference / 2.0
    )
    radius = np.hypot(x, y)
    direct_radius_squared = (
        np.abs(alignment.structure_factor_b - factor_a) ** 2 / denominator**2
    )
    identity_error = float(np.max(np.abs(radius**2 - direct_radius_squared)))
    d_amplitude = float(np.sqrt(np.sum(normalized_weights * x**2)))
    d_phase = float(np.sqrt(np.sum(normalized_weights * y**2)))
    d_sf = float(np.hypot(d_amplitude, d_phase))

    identity_match = ReflectionMatch(
        hkl=indices.copy(),
        indices_a=np.arange(len(indices), dtype=np.int64),
        indices_b=np.arange(len(indices), dtype=np.int64),
    )
    return MismatchDiskResult(
        match=identity_match,
        alignment=alignment,
        structure_factor_a=factor_a,
        amplitude_a=amplitude_a,
        amplitude_b=amplitude_b,
        phase_difference=phase_difference,
        phase_defined=phase_defined,
        x=x,
        y=y,
        radius=radius,
        weights=normalized_weights,
        d_sf=d_sf,
        d_amplitude=d_amplitude,
        d_phase=d_phase,
        identity_error=identity_error,
        epsilon=regularizer,
        phase_threshold=threshold,
    )


def compare_calculators(
    calculator_a,
    calculator_b,
    *,
    domain: str = "q",
    optimize_origin: bool = True,
    weights=None,
    lattice_rtol: float = 1e-7,
    lattice_atol: float = 1e-8,
    **disk_options,
) -> MismatchDiskResult:
    """Compare two loaded calculators with the same lattice representation."""
    calculator_a._ensure_loaded()
    calculator_b._ensure_loaded()
    if calculator_a.mode != calculator_b.mode or not np.isclose(
        calculator_a.wavelength, calculator_b.wavelength
    ):
        raise ValueError("calculators must use the same radiation mode and wavelength")
    if not np.allclose(
        calculator_a._symm["lattice"],
        calculator_b._symm["lattice"],
        rtol=lattice_rtol,
        atol=lattice_atol,
    ):
        raise ValueError(
            "automatic cell transformations are not implemented; use the same lattice setting"
        )

    table_a = calculator_a.reflection_table(domain=domain)
    table_b = calculator_b.reflection_table(domain=domain)
    match = match_reflections(table_a.hkl, table_b.hkl)
    selected_weights = None if weights is None else np.asarray(weights)[match.indices_a]
    result = mismatch_disk(
        match.hkl,
        _as_numpy(table_a.structure_factor)[match.indices_a],
        _as_numpy(table_b.structure_factor)[match.indices_b],
        weights=selected_weights,
        optimize_origin=optimize_origin,
        **disk_options,
    )
    return MismatchDiskResult(
        match=match,
        alignment=result.alignment,
        structure_factor_a=result.structure_factor_a,
        amplitude_a=result.amplitude_a,
        amplitude_b=result.amplitude_b,
        phase_difference=result.phase_difference,
        phase_defined=result.phase_defined,
        x=result.x,
        y=result.y,
        radius=result.radius,
        weights=result.weights,
        d_sf=result.d_sf,
        d_amplitude=result.d_amplitude,
        d_phase=result.d_phase,
        identity_error=result.identity_error,
        epsilon=result.epsilon,
        phase_threshold=result.phase_threshold,
    )
